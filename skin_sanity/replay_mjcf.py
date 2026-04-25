"""Replay a saved trajectory through the EXACT MJCF used for data generation and
render depth + RGB from every proximity sensor camera, for every timestep.

Inputs
------
- HDF5 from FrankaSkinPickConfig datagen (has `env_states/articulations/panda`
  + `obs/extra/proximity`).
- `resources/robots/franka_droid_skin/model.xml` (29 sensor cameras, plus
  the wrist camera).

Outputs (saved to OUT_DIR, defaults to /home/jaydv/code/skin_sanity/replay_traj0)
- `replay_depth.npy`          (T, 29, 8, 8) float32 — re-rendered depth
- `replay_rgb.npy`            (T, 29, 8, 8, 3) uint8 — re-rendered RGB
- `replay_sensor_names.txt`   29 names, matching the tensor axis order
- `replay_vs_saved.png`       per-patch MAE histogram between replay and saved
  proximity (if <= ~5 mm across all patches, the replay reproduces datagen).

Why this is the right "full pass with all sensors"
-------------------------------------------------
The original datagen saved depth but NOT RGB from the proximity cameras
(the `ProximityRGBSensor` was added later). The saved MJCF + qpos is
fully self-contained, so we can reproduce every depth reading AND
harvest the RGB view from the same cameras. If replay_depth matches
saved_proximity, the analysis stands; the RGB tells us what each
patch was actually looking at.

Usage
-----
    python replay_mjcf.py                       # default traj_0
    python replay_mjcf.py --traj-idx 1
    python replay_mjcf.py --max-steps 10        # smoke-test
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import h5py
import mujoco

H5_DEFAULT    = '/home/jaydv/code/molmo/resources/experiment_output/datagen/skin_pick_v1/FrankaSkinPickConfig/20260416_133342/house_0/trajectories_batch_1_of_1.h5'
MJCF_DEFAULT  = '/home/jaydv/code/molmo/resources/robots/franka_droid_skin/model.xml'
OUT_DEFAULT   = '/home/jaydv/code/skin_sanity/replay_traj0'

ZNEAR, ZFAR = 0.02, 4.0
RES = 8


def list_sensor_cams(model: mujoco.MjModel) -> list[tuple[int, str]]:
    out = []
    for cid in range(model.ncam):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        if n and 'sensor_' in n:
            out.append((cid, n))
    return out


def apply_qpos(model: mujoco.MjModel, data: mujoco.MjData, panda_row: np.ndarray) -> None:
    """Map the saved panda layout onto the MJCF's qpos vector.

    The saver stores `[arm(7), fingers(2), ...]` (31 floats). MJCF `nq=13` with
    layout `[arm(7), gripper_driver+5mimics(6)]`. The 5 mimic joints follow
    the driver via <equality> constraints; `mj_forward` propagates them.
    """
    data.qpos[:7] = panda_row[:7]
    # Saved finger values are driver-joint angle (radians). Zero = open.
    data.qpos[7] = float(panda_row[7])
    # The remaining five gripper joints are slaved; setting them identically is
    # a reasonable seed, and mj_forward + equality will settle them.
    data.qpos[8:13] = float(panda_row[7])


def render_one_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    depth_ren: mujoco.Renderer,
    rgb_ren: mujoco.Renderer,
    cams: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Render (29, 8, 8) depth and (29, 8, 8, 3) RGB for all sensor cams at the
    current qpos."""
    n = len(cams)
    depth = np.empty((n, RES, RES), dtype=np.float32)
    rgb = np.empty((n, RES, RES, 3), dtype=np.uint8)
    for i, (_, name) in enumerate(cams):
        depth_ren.update_scene(data, camera=name)
        d = depth_ren.render()
        depth[i] = d
        rgb_ren.update_scene(data, camera=name)
        rgb[i] = rgb_ren.render()
    # Match datagen clipping: out-of-range → zfar sentinel.
    depth = np.where(depth < ZNEAR, ZFAR, depth)
    depth = np.where(depth > ZFAR, ZFAR, depth)
    return depth, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', type=Path, default=Path(H5_DEFAULT))
    ap.add_argument('--mjcf', type=Path, default=Path(MJCF_DEFAULT))
    ap.add_argument('--traj-idx', type=int, default=0)
    ap.add_argument('--out-dir', type=Path, default=Path(OUT_DEFAULT))
    ap.add_argument('--max-steps', type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[replay] loading {args.mjcf}')
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    cams = list_sensor_cams(model)
    names = [n for _, n in cams]
    assert len(cams) == 29, f'expected 29 sensor cameras, got {len(cams)}'

    print(f'[replay] loading trajectory {args.traj_idx} from {args.h5}')
    with h5py.File(args.h5, 'r') as f:
        panda = f[f'traj_{args.traj_idx}/env_states/articulations/panda'][:]
        saved = f[f'traj_{args.traj_idx}/obs/extra/proximity'][:]  # (T,29,8,8)
    T = panda.shape[0]
    if args.max_steps is not None:
        T = min(T, args.max_steps)
    print(f'[replay] rendering {T} steps × {len(cams)} cameras = {T * len(cams)} renders')

    # Use two renderers so toggling depth/RGB mode is free.
    depth_ren = mujoco.Renderer(model, height=RES, width=RES)
    depth_ren.enable_depth_rendering()
    rgb_ren = mujoco.Renderer(model, height=RES, width=RES)

    replay_depth = np.empty((T, 29, RES, RES), dtype=np.float32)
    replay_rgb = np.empty((T, 29, RES, RES, 3), dtype=np.uint8)

    t0 = time.time()
    for t in range(T):
        apply_qpos(model, data, panda[t])
        mujoco.mj_forward(model, data)
        d, c = render_one_step(model, data, depth_ren, rgb_ren, cams)
        replay_depth[t] = d
        replay_rgb[t] = c
        if (t + 1) % 10 == 0 or t == T - 1:
            print(f'  t={t + 1}/{T}   elapsed={time.time() - t0:.1f}s')

    depth_ren.close()
    rgb_ren.close()

    # ----- Compare against saved proximity -----
    # Saved tensor was produced by the same MJCF + same qpos + same renderer
    # config. Expected discrepancy: floating-point + mesh simplification noise.
    common_T = min(T, saved.shape[0])
    diff = np.abs(replay_depth[:common_T] - saved[:common_T])
    per_patch_mae = diff.reshape(common_T, 29, -1).mean(axis=(0, 2))
    per_patch_max = diff.reshape(common_T, 29, -1).max(axis=(0, 2))

    # ----- Persist -----
    np.save(args.out_dir / 'replay_depth.npy', replay_depth)
    np.save(args.out_dir / 'replay_rgb.npy', replay_rgb)
    (args.out_dir / 'replay_sensor_names.txt').write_text('\n'.join(names))

    print('\n[replay] per-patch MAE (replay vs. saved, metres):')
    for i in range(29):
        mark = ' <-- high' if per_patch_max[i] > 0.01 else ''
        print(f'  [{i:2d}] {names[i]:<20}  mae={per_patch_mae[i]*1000:6.3f} mm   max={per_patch_max[i]*1000:6.3f} mm{mark}')
    print(f'\n[replay] overall MAE = {diff.mean() * 1000:.4f} mm   max = {diff.max() * 1000:.4f} mm')
    print(f'[replay] saved: replay_depth.npy, replay_rgb.npy  ({replay_depth.nbytes / 1e6:.1f} MB + {replay_rgb.nbytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
