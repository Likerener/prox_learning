"""Self-contained pick-and-place pipeline demo.

A scripted, kinematic FR3 pick-and-place trajectory rendered through the
PLA collection schema. The point is to show end-to-end that:

  * a real MuJoCo robot model produces real RGB frames
  * the HDF5 schema is valid
  * `pla.viz.dataset_audit` produces meaningful plots from real data
  * a watchable MP4 video is generated

Caveats — be honest:

  * **Not a procthor house.** The scene is a simple wooden table + cube +
    target zone (the same scene that ships with the `franka_fr3`
    resource bundle). A real MolmoSpaces house-1 collection would
    randomise scene, object, and language.
  * **Kinematic, not dynamic.** We set `qpos` directly each step rather
    than running the physics. The cube does not actually get grasped —
    we mocap-track it with the EE during the carry phase. Good enough
    for visualising what the data looks like.
  * **ToF is structurally correct but synthesised.** Per-step we compute
    a min depth from the hand-to-object world-frame distance and produce
    an 8x8 grid centred on that value; other zones get noise. This
    gives the audit plots realistic temporal structure (depth dipping
    during approach) without requiring whole-body skin attachment.

Run::

    PYTHONPATH=. python -m pla.sim.demo_pnp \
        --out reports/demo_pnp \
        --n-episodes 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from pla.data.collect import _write_episode_h5

# Path to the FR3 scene from the molmo-spaces resource cache.
DEFAULT_SCENE = (
    "/home/jaydv/.cache/molmo-spaces-resources/"
    "robots/franka_fr3/20260303/scene_fr3.xml"
)

# FR3 'home' qpos (7 arm + 2 finger).
HOME_ARM = np.array([0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.7853])
FINGERS_OPEN = np.array([0.04, 0.04])
FINGERS_CLOSED = np.array([0.0, 0.0])

# Hand-tuned waypoints (arm only, fingers handled separately).
# Chosen so the EE traces approach -> grasp -> lift -> carry -> place -> retreat.
WAYPOINTS = [
    # (label, arm_qpos, fingers_target, dwell_steps)
    ("home",         HOME_ARM,                                              FINGERS_OPEN,   20),
    ("approach",     np.array([-0.12, -0.50,  0.05, -2.10, 0.0, 1.52, 0.7853]), FINGERS_OPEN,   30),
    ("descend",      np.array([-0.12, -0.20,  0.05, -2.40, 0.0, 2.10, 0.7853]), FINGERS_OPEN,   30),
    ("grasp",        np.array([-0.12, -0.20,  0.05, -2.40, 0.0, 2.10, 0.7853]), FINGERS_CLOSED, 15),
    ("lift",         np.array([-0.12, -0.55,  0.05, -2.05, 0.0, 1.55, 0.7853]), FINGERS_CLOSED, 25),
    ("carry",        np.array([ 0.20, -0.45,  0.05, -2.10, 0.0, 1.55, 0.7853]), FINGERS_CLOSED, 30),
    ("descend_place",np.array([ 0.20, -0.18,  0.05, -2.43, 0.0, 2.08, 0.7853]), FINGERS_CLOSED, 25),
    ("release",      np.array([ 0.20, -0.18,  0.05, -2.43, 0.0, 2.08, 0.7853]), FINGERS_OPEN,   15),
    ("retreat",      np.array([ 0.20, -0.55,  0.05, -2.05, 0.0, 1.55, 0.7853]), FINGERS_OPEN,   20),
    ("home_back",    HOME_ARM,                                              FINGERS_OPEN,   20),
]

ZNEAR_MM = 20.0
ZFAR_MM = 4000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=Path(DEFAULT_SCENE))
    p.add_argument("--out", type=Path, default=Path("reports/demo_pnp"))
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--n-sensors", type=int, default=8)
    p.add_argument("--rgb-h", type=int, default=224)
    p.add_argument("--rgb-w", type=int, default=224)
    p.add_argument("--video-fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def interpolate_trajectory(seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build a (T, 9) qpos trajectory + (T, 9) action chunk + per-step phase id."""
    rng = np.random.default_rng(seed)
    qpos_traj = []
    phase_ids = []
    prev_arm = WAYPOINTS[0][1].copy()
    prev_fin = WAYPOINTS[0][2].copy()
    for phase_id, (label, arm_target, fin_target, dwell) in enumerate(WAYPOINTS):
        # small per-episode noise on waypoints so each episode varies
        arm_target = arm_target + rng.normal(0, 0.01, size=arm_target.shape)
        for s in range(dwell):
            alpha = (s + 1) / dwell
            arm = (1 - alpha) * prev_arm + alpha * arm_target
            fin = (1 - alpha) * prev_fin + alpha * fin_target
            qpos_traj.append(np.concatenate([arm, fin]))
            phase_ids.append(phase_id)
        prev_arm = arm_target
        prev_fin = fin_target
    qpos_traj = np.asarray(qpos_traj, dtype=np.float32)
    actions = np.diff(qpos_traj, axis=0, prepend=qpos_traj[:1])[:, :7]
    return qpos_traj, actions.astype(np.float32), phase_ids


def make_renderer(model, height: int, width: int) -> mujoco.Renderer:
    return mujoco.Renderer(model, height=height, width=width)


def make_free_camera(distance: float = 1.6, azimuth: float = -135.0,
                     elevation: float = -25.0,
                     lookat=(0.55, 0.0, 0.35)) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = lookat
    return cam


def synthesize_tof(hand_pos_world: np.ndarray, object_pos_world: np.ndarray,
                   n_sensors: int, rng: np.random.Generator,
                   carrying: bool = False) -> np.ndarray:
    """Make a structurally-correct ToF [N, 8, 8] frame.

    The minimum reading per sensor tracks the hand-to-object distance
    (in mm). When the hand is far, depths are uniform in [200, 4000]
    mm; when the hand is within 10 cm of the object, the closest
    sensor reads down to 20 mm. Other sensors get less-tight readings
    sampled from a normal centred at min + offset.
    """
    dist_mm = float(np.linalg.norm(hand_pos_world - object_pos_world)) * 1000.0
    if carrying:
        # The hand "carries" the object — closest reading is small.
        base_min = 25.0 + rng.uniform(-3, 3)
    else:
        base_min = max(ZNEAR_MM, min(ZFAR_MM, dist_mm))
    out = np.zeros((n_sensors, 8, 8), dtype=np.float32)
    for i in range(n_sensors):
        # Each sensor sees the object at a slightly different distance
        offset = rng.uniform(0, 250) + 80 * (i / max(n_sensors - 1, 1))
        per_sensor_min = max(ZNEAR_MM, min(ZFAR_MM, base_min + offset))
        out[i] = rng.normal(per_sensor_min + 80, 60, size=(8, 8)).astype(np.float32)
        # The closest sensor (i=0) gets a clear "hot zone" near the object
        if i == 0:
            out[i, 3:5, 3:5] = per_sensor_min + rng.normal(0, 5, size=(2, 2))
    return np.clip(out, ZNEAR_MM, ZFAR_MM)


def render_one_episode(model, data, renderer, rgb_cam, qpos_traj, actions,
                       phase_ids, n_sensors, rng, video_writer=None,
                       overlay_text=True):
    """Step through the kinematic trajectory; render + collect."""
    T = qpos_traj.shape[0]
    rgb_buf = np.zeros((T, 3, model.vis.global_.offwidth // 1, model.vis.global_.offwidth // 1),
                       dtype=np.uint8)
    rgb_buf = np.zeros((T, 3, renderer.height, renderer.width), dtype=np.uint8)
    qpos_buf = np.zeros((T, 7), dtype=np.float32)
    tof_buf = np.zeros((T, n_sensors, 8, 8), dtype=np.float32)

    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object1")
    hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    for t in range(T):
        # Set qpos directly (kinematic).
        data.qpos[:7] = qpos_traj[t][:7]
        data.qpos[7:9] = qpos_traj[t][7:9]
        # During "carry" / "descend_place" / "release" we mocap the object
        # to the hand so the visual is consistent.
        carrying = phase_ids[t] in (4, 5, 6, 7)  # lift / carry / descend_place / release
        mujoco.mj_forward(model, data)
        hand_pos = data.xpos[hand_body].copy()
        if carrying and "object1_freejoint_addr" not in dir():
            # Get freejoint qpos address for object1
            jnt_id = model.body_jntadr[object_body]
            qadr = model.jnt_qposadr[jnt_id]
            # Place the object 5cm below the hand TCP-ish offset.
            data.qpos[qadr:qadr + 3] = hand_pos + np.array([0.0, 0.0, -0.05])
            # leave quaternion fixed
            mujoco.mj_forward(model, data)

        # Render RGB
        renderer.update_scene(data, camera=rgb_cam)
        img = renderer.render()  # [H, W, 3] uint8
        rgb_buf[t] = img.transpose(2, 0, 1)
        qpos_buf[t] = qpos_traj[t][:7]
        # Synthesise tof from hand-object distance.
        obj_pos = data.xpos[object_body].copy()
        tof_buf[t] = synthesize_tof(hand_pos, obj_pos, n_sensors, rng,
                                     carrying=carrying)

        if video_writer is not None:
            frame = img.copy()
            if overlay_text:
                # Light HUD text via direct pixel poke (simple, no font needed):
                # add a coloured strip across top with phase id encoded.
                strip = np.zeros((10, frame.shape[1], 3), dtype=np.uint8)
                strip[:, :, (phase_ids[t] % 3)] = 200
                strip[:, : (phase_ids[t] + 1) * (frame.shape[1] // 12)] = 255
                frame[:10, :, :] = strip
            video_writer.append_data(frame)

    success = bool(phase_ids[-1] == len(WAYPOINTS) - 1)  # got back home
    return {
        "tof": tof_buf,
        "rgb": rgb_buf,
        "qpos": qpos_buf,
    }, actions, success


def main() -> None:
    args = parse_args()
    if not args.scene.exists():
        raise SystemExit(
            f"Scene file not found: {args.scene}\n"
            f"Run a quick `find ~ -name 'scene_fr3.xml' 2>/dev/null` to locate it."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.scene}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    print(f"  nq={model.nq} nbody={model.nbody}")

    renderer = make_renderer(model, args.rgb_h, args.rgb_w)
    rgb_cam = make_free_camera()

    rng = np.random.default_rng(args.seed)

    for ep_idx in range(args.n_episodes):
        print(f"\n[{ep_idx+1}/{args.n_episodes}] generating episode")
        ep_rng = np.random.default_rng(args.seed + ep_idx)
        qpos_traj, actions, phase_ids = interpolate_trajectory(args.seed + ep_idx)
        T = qpos_traj.shape[0]
        print(f"  T={T} phases={len(WAYPOINTS)}")

        video_path = args.out / f"episode_{ep_idx:03d}.mp4"
        # Collect frames first; write as MP4 at the end via ffmpeg backend.
        frames = []

        class _FrameSink:
            def append_data(self, frame: np.ndarray) -> None:
                frames.append(frame)

        obs_seq, actions, success = render_one_episode(
            model, data, renderer, rgb_cam,
            qpos_traj, actions, phase_ids,
            args.n_sensors, ep_rng, video_writer=_FrameSink(),
        )
        iio.imwrite(video_path, np.stack(frames), fps=args.video_fps,
                    plugin="FFMPEG", codec="libx264")
        print(f"  video: {video_path}  ({video_path.stat().st_size/1e6:.2f} MB)")

        h5_path = raw_dir / f"episode_{ep_idx:06d}.h5"
        _write_episode_h5(
            h5_path,
            obs_seq=obs_seq,
            actions=actions,
            success=success,
            n_sensors=args.n_sensors,
            policy_phase=np.asarray(phase_ids, dtype=np.int32),
            extra_attrs={
                "demo": "kinematic_pnp",
                "language": "pick up the blue cube and place it on the green target",
            },
        )
        print(f"  h5:    {h5_path}  ({h5_path.stat().st_size/1e6:.2f} MB)")

        # Also dump a few key still frames as PNG.
        keyframe_idx = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        for k, t in enumerate(keyframe_idx):
            iio.imwrite(args.out / f"episode_{ep_idx:03d}_frame{k}_t{t:03d}.png",
                        obs_seq["rgb"][t].transpose(1, 2, 0))

    # Run the audit pipeline on the result.
    print("\n=== audit pipeline on the demo dataset ===")
    audit_dir = args.out / "audit"
    import subprocess
    subprocess.run(
        ["python", "-m", "pla.viz.dataset_audit",
         "--data-dir", str(raw_dir), "--out", str(audit_dir)],
        check=True, env={"PYTHONPATH": "."} | dict(__import__("os").environ),
    )

    # Run the deep verifier.
    print("\n=== deep verify on the demo dataset ===")
    subprocess.run(
        ["python", "-m", "pla.data.verify",
         "--data-dir", str(raw_dir),
         "--report", str(args.out / "verify.json")],
        check=True, env={"PYTHONPATH": "."} | dict(__import__("os").environ),
    )

    print(f"\nDone. Browse the output:\n  {args.out}/")
    print(f"  videos:  episode_*.mp4")
    print(f"  stills:  episode_*_frame*.png")
    print(f"  h5:      raw/episode_*.h5")
    print(f"  audit:   audit/INDEX.md")


if __name__ == "__main__":
    main()
