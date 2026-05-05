"""Rigorous tests + diagnostic plots for ToF → 3D pointcloud reconstruction.

Validates the projection math in :mod:`pla.viz.pointcloud_core` against
ground truth derived from mujoco's depth renderer. Runs five tests:

  T1. Pinhole intrinsics : taxel ray directions span the configured FOV
      and the central rays equal the optical axis.
  T2. Synthetic flat wall : a constant-depth image unprojects to a
      planar point cloud at the correct distance, with the expected
      side length given the FOV.
  T3. mujoco wall ground truth : place a real wall geom in front of one
      sensor, render mujoco depth, reconstruct world points, and check
      they lie on the wall plane (target rms < 1 mm in mujoco units).
  T4. Multi-sensor coverage : depth-render all 29 sensor cameras at the
      home pose against a simple scene, reconstruct, and verify the
      world point cloud is consistent with each sensor's own forward
      ray (i.e. no "ghost" points behind sensors).
  T5. Pose invariance : move the arm to two different qpos values; the
      reconstructed cloud of a static obstacle stays in the same world
      location (translation-invariant under arm motion).

Each test prints a PASS/FAIL line; failures cause exit code 1. Plots
land in ``reports/checks/pointcloud_tests/``.

Run:

    MUJOCO_GL=egl python -m pla.viz.pointcloud_tests
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pla.viz.pointcloud_core import (
    PinholeIntrinsics,
    taxel_directions,
    unproject_taxels,
    unproject_taxels_to_body,
    reconstruct_world_pts,
    fit_plane,
)
from pla.viz.sensor_overlay import (
    patch_mjcf,
    collect_sensors,
    LINK_COLOR,
    LINK_COUNT,
    HOME_QPOS,
    MJCF_SRC,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "reports/checks/pointcloud_tests"
INTR = PinholeIntrinsics(fovy_deg=45.0, res=8, znear_m=0.02, zfar_m=4.0)


# ---------------------------------------------------------------------------
# helpers


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


def run_check(name: str, fn) -> TestResult:
    try:
        msg = fn()
        passed = True
    except AssertionError as e:
        msg = str(e); passed = False
    except Exception as e:  # pragma: no cover — surface tracebacks
        import traceback; traceback.print_exc()
        msg = f"unexpected exception: {type(e).__name__}: {e}"; passed = False
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}: {msg}")
    return TestResult(name, passed, msg)


def _set_qpos(model, data, qpos):
    for i in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i+1}")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = qpos[i]
    mujoco.mj_forward(model, data)


def _render_depth(model, data, cam_name: str, res: int = 8) -> np.ndarray:
    """Render a (res, res) metric depth image for ``cam_name``."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise RuntimeError(f"missing camera {cam_name}")
    r = mujoco.Renderer(model, height=res, width=res)
    r.enable_depth_rendering()
    r.update_scene(data, camera=cam_name)
    depth = r.render().astype(np.float32).copy()
    r.close()
    return depth


def _patched_model_with_wall(out_dir: Path, wall_pos, wall_size, wall_quat=None):
    """Patch the FR3 skin XML and inject a static box obstacle ('wall')."""
    text_extra = (
        f'    <body name="probe_wall" pos="{wall_pos[0]:.4f} {wall_pos[1]:.4f} {wall_pos[2]:.4f}"'
    )
    if wall_quat is not None:
        text_extra += (
            f' quat="{wall_quat[0]:.4f} {wall_quat[1]:.4f} '
            f'{wall_quat[2]:.4f} {wall_quat[3]:.4f}"'
        )
    text_extra += ">\n"
    text_extra += (
        f'      <geom type="box" size="{wall_size[0]:.4f} {wall_size[1]:.4f} '
        f'{wall_size[2]:.4f}" rgba="0.8 0.3 0.3 1"/>\n'
        f"    </body>\n"
    )
    src = MJCF_SRC.read_text()
    src = src.replace("<worldbody>", "<worldbody>\n" + text_extra, 1)
    tmp = out_dir / "_fr3_skin_wall.xml"
    tmp.write_text(src)
    final = patch_mjcf(tmp, out_dir / "_fr3_skin_patched_wall.xml")
    return final


# ---------------------------------------------------------------------------
# tests


def test_pinhole_intrinsics() -> str:
    """T1: ray directions span the configured FOV; corner rays match tan(fovy/2)."""
    intr = PinholeIntrinsics(fovy_deg=45.0, res=8)
    dirs = taxel_directions(intr, frame="depth_axis")
    assert dirs.shape == (8, 8, 3), f"shape mismatch: {dirs.shape}"
    assert np.allclose(dirs[..., 2], 1.0), "depth_axis: z-component must be +1"
    cam_dirs = taxel_directions(intr, frame="camera")
    assert np.allclose(cam_dirs[..., 2], -1.0), "camera: z-component must be -1"
    half = intr.half_tan
    expected_corner_u = (1 - 1 / 8) * half  # 0.875 * half
    actual_corner_u = abs(dirs[0, -1, 0])
    rel_err = abs(actual_corner_u - expected_corner_u) / expected_corner_u
    assert rel_err < 1e-6, f"corner u off: got {actual_corner_u}, want {expected_corner_u}"
    cx_avg = dirs[3:5, 3:5, 0].mean()
    assert abs(cx_avg) < 1e-9, f"central rays should be near optical axis: {cx_avg}"
    return f"45° fovy, 8x8, half_tan={half:.4f}, corner u={actual_corner_u:.4f} (both frames OK)"


def test_synthetic_flat_wall() -> str:
    """T2: constant depth → coplanar points at correct distance, side length = 2·d·tan(fovy/2)."""
    d_true = 0.5
    depth = np.full((8, 8), d_true, dtype=np.float32)
    # camera-frame: z = -d (in front of camera)
    pts_cam, mask = unproject_taxels(depth, intr=INTR, frame="camera")
    assert mask.all(), f"all 64 taxels should be valid; got {mask.sum()}"
    assert pts_cam.shape == (64, 3)
    assert np.allclose(pts_cam[:, 2], -d_true), \
        f"camera z must equal -d_true; got mean {pts_cam[:, 2].mean()}"
    # depth_axis-frame: z = +d
    pts_axis, _ = unproject_taxels(depth, intr=INTR, frame="depth_axis")
    assert np.allclose(pts_axis[:, 2], d_true)
    expected_half = INTR.half_tan * d_true * (1 - 1 / INTR.res)
    rel = abs(pts_cam[:, 0].max() - expected_half) / expected_half
    assert rel < 1e-6, f"x extent off: got {pts_cam[:, 0].max():.4f}, want {expected_half:.4f}"
    centroid, normal, rms = fit_plane(pts_cam)
    assert rms < 1e-9, f"plane fit residual too high: {rms}"
    assert abs(abs(normal[2]) - 1) < 1e-9, f"normal not along z: {normal}"
    return f"d={d_true}m → 64 pts both frames, half-extent={expected_half*1000:.1f}mm rms={rms:.1e}"


def _pick_sensor_facing(sensors, model, data, target_world: np.ndarray):
    """Choose the sensor whose true camera looking-direction best aligns with ``target_world``.

    Reads ``data.cam_xmat`` so it is independent of the body-frame
    convention question. The mujoco camera looks down its -z, so the
    look direction in world is ``-cam_xmat[:, 2]``.
    """
    best_score, best_s, best_cid = -2.0, None, -1
    for s in sensors:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, s.name)
        if cid < 0:
            continue
        cam_R = data.cam_xmat[cid].reshape(3, 3)
        look = -cam_R[:, 2]
        score = float(look @ target_world)
        if score > best_score:
            best_score, best_s, best_cid = score, s, cid
    return best_s, best_cid


def test_mujoco_wall_ground_truth(out_dir: Path) -> tuple[str, dict]:
    """T3: render mujoco depth on a real flat wall, reconstruct, check rms < 1 mm.

    Uses the ground-truth camera pose (``data.cam_xpos`` /
    ``data.cam_xmat``) and ``frame="camera"`` so the math is the
    standard OpenGL inverse-pinhole — the only thing being tested is the
    mujoco depth pipeline + our pinhole intrinsics.
    """
    wall_xml = _patched_model_with_wall(
        out_dir,
        wall_pos=(0.55, 0.0, 0.50),
        wall_size=(0.005, 0.40, 0.40),  # thin, broad in YZ
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    _set_qpos(model, data, HOME_QPOS)
    sensors = collect_sensors(model, data)
    best_s, cid = _pick_sensor_facing(sensors, model, data, np.array([1.0, 0.0, 0.0]))
    assert best_s is not None and cid >= 0, "no sensor found"
    depth = _render_depth(model, data, best_s.name, res=INTR.res)
    valid_frac = float(((depth > INTR.znear_m + 1e-3) & (depth < INTR.zfar_m - 1e-3)).mean())
    assert valid_frac > 0.5, f"{best_s.name} saw only {valid_frac:.0%} taxels"
    cam_pos = data.cam_xpos[cid].copy()
    cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
    pts_world, mask = reconstruct_world_pts(depth, cam_pos, cam_R, intr=INTR, frame="camera")
    front_x = 0.55 - 0.005
    residuals = np.abs(pts_world[:, 0] - front_x)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_res = float(residuals.max())
    extras = dict(
        sensor=best_s.name, link=best_s.link, cam_id=int(cid),
        n_pts=int(len(pts_world)), valid_frac=valid_frac,
        rms_mm=rms * 1000, max_mm=max_res * 1000,
        depth_mean=float(depth[mask].mean()),
        cam_pos=[float(x) for x in cam_pos],
        front_x=front_x,
    )
    assert rms < 1e-3, f"wall rms = {rms*1000:.2f} mm > 1 mm tolerance"
    assert max_res < 5e-3, f"worst-case wall point off by {max_res*1000:.2f} mm > 5 mm"
    return (
        f"cam={best_s.name} n={len(pts_world)} valid={valid_frac:.0%} "
        f"rms={rms*1000:.3f} mm max={max_res*1000:.3f} mm"
    ), extras


def test_multi_sensor_coverage(out_dir: Path) -> tuple[str, dict]:
    """T4: every sensor's reconstructed points lie *in front of* the sensor (no ghosts behind it)."""
    wall_xml = _patched_model_with_wall(
        out_dir,
        wall_pos=(0.55, 0.0, 0.50),
        wall_size=(0.40, 0.40, 0.40),
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    _set_qpos(model, data, HOME_QPOS)
    sensors = collect_sensors(model, data)

    n_sensors_with_hits = 0
    n_total_pts = 0
    n_violations = 0
    per_link_pts = {lk: 0 for lk in LINK_COLOR}
    for s in sensors:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, s.name)
        depth = _render_depth(model, data, s.name, res=INTR.res)
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        pts_world, _ = reconstruct_world_pts(
            depth, cam_pos, cam_R, intr=INTR, frame="camera",
        )
        if len(pts_world) == 0:
            continue
        n_sensors_with_hits += 1
        n_total_pts += len(pts_world)
        per_link_pts[s.link] += len(pts_world)
        # Each reconstructed point must lie in the look direction half-space.
        # Camera looks down -z_cam, so look = -cam_R[:,2]. Points in front
        # have positive projection on look.
        look = -cam_R[:, 2]
        rel = pts_world - cam_pos
        proj = rel @ look
        n_violations += int((proj < -1e-4).sum())
    assert n_violations == 0, f"{n_violations} reconstructed points behind their sensor!"
    extras = dict(
        n_sensors_with_hits=n_sensors_with_hits,
        n_total_pts=n_total_pts,
        per_link=per_link_pts,
        n_violations=n_violations,
    )
    return f"{n_sensors_with_hits}/29 sensors had hits, {n_total_pts} points total, 0 ghosts", extras


def test_pose_invariance(out_dir: Path) -> tuple[str, dict]:
    """T5: a static wall reconstructs to the same world plane from two different arm poses."""
    wall_xml = _patched_model_with_wall(
        out_dir,
        wall_pos=(0.55, 0.0, 0.50),
        wall_size=(0.005, 0.40, 0.40),
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    poses = [HOME_QPOS, HOME_QPOS + np.array([0.3, 0.1, -0.2, 0.1, 0.0, 0.0, 0.0])]
    front_x = 0.55 - 0.005
    rms_per_pose, counts = [], []
    for qp in poses:
        _set_qpos(model, data, qp)
        sensors = collect_sensors(model, data)
        best_s, cid = _pick_sensor_facing(sensors, model, data, np.array([1.0, 0, 0]))
        assert best_s is not None
        depth = _render_depth(model, data, best_s.name, res=INTR.res)
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        pts_world, _ = reconstruct_world_pts(depth, cam_pos, cam_R, intr=INTR, frame="camera")
        if len(pts_world) == 0:
            rms_per_pose.append(np.nan); counts.append(0); continue
        rms = float(np.sqrt(np.mean((pts_world[:, 0] - front_x) ** 2)))
        rms_per_pose.append(rms); counts.append(len(pts_world))
    assert all(c > 0 for c in counts), f"some pose had no hits: {counts}"
    for i, r in enumerate(rms_per_pose):
        assert r < 5e-3, f"pose {i} reconstruction rms = {r*1000:.2f} mm > 5 mm"
    extras = dict(rms_mm=[r * 1000 for r in rms_per_pose], counts=counts)
    return (
        f"poses A/B align with wall: rms = {rms_per_pose[0]*1000:.2f} / "
        f"{rms_per_pose[1]*1000:.2f} mm  (n = {counts[0]} / {counts[1]})"
    ), extras


def test_legacy_convention_parity(out_dir: Path) -> tuple[str, dict]:
    """T6: legacy ``pla.viz.pointcloud`` convention sanity check.

    The legacy code reads body-frame poses (via pybullet) and unprojects
    with z=+d. To reach the same world point as the camera-frame
    reconstruction, the legacy pipeline would have to either (a) invert
    the X- and Y-flips introduced by the MJCF camera ``quat="0 1 0 0"``,
    or (b) match the *fixed* y-flip in its own ``-v`` lateral term.

    Empirically (this test) the legacy formula ``(u·half·d, -v·half·d,
    d)`` × R_body has a residual y-flip vs the camera-frame ground
    truth: errors scale linearly with depth and lateral pixel offset.
    A correct body-frame formula would be ``(u·half·d, +v·half·d, d)``.

    This test is a **regression report** (RMS in mm) rather than a hard
    pass/fail — the bug it surfaces is fixed in
    :func:`reconstruct_world_pts` when called with ``frame="camera"``.
    """
    wall_xml = _patched_model_with_wall(
        out_dir, wall_pos=(0.55, 0.0, 0.50), wall_size=(0.005, 0.40, 0.40),
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    _set_qpos(model, data, HOME_QPOS)
    sensors = collect_sensors(model, data)
    best_s, cid = _pick_sensor_facing(sensors, model, data, np.array([1.0, 0.0, 0.0]))
    depth = _render_depth(model, data, best_s.name, res=INTR.res)
    cam_pos = data.cam_xpos[cid].copy()
    cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
    body_pos = data.xpos[best_s.body_id].copy()
    body_R = data.xmat[best_s.body_id].reshape(3, 3).copy()
    pts_cam, _ = reconstruct_world_pts(depth, cam_pos, cam_R, intr=INTR, frame="camera")
    pts_legacy_buggy, _ = reconstruct_world_pts(
        depth, body_pos, body_R, intr=INTR, frame="depth_axis",
    )
    # Construct the corrected body-frame variant (+v instead of -v).
    pts_body_fixed, _ = unproject_taxels(depth, intr=INTR, frame="depth_axis")
    pts_body_fixed[:, 1] *= -1.0  # flip y back: -(-v) = +v
    pts_world_fixed = pts_body_fixed @ body_R.T + body_pos[None, :]

    diff_legacy = np.linalg.norm(pts_cam - pts_legacy_buggy, axis=1) * 1000
    diff_fixed = np.linalg.norm(pts_cam - pts_world_fixed, axis=1) * 1000
    rms_legacy = float(np.sqrt(np.mean(diff_legacy ** 2)))
    rms_fixed = float(np.sqrt(np.mean(diff_fixed ** 2)))
    max_legacy = float(diff_legacy.max())
    extras = dict(
        rms_legacy_mm=rms_legacy, max_legacy_mm=max_legacy,
        rms_fixed_mm=rms_fixed, n=int(len(pts_cam)),
    )
    # Hard requirement: the *corrected* body formula must agree with camera-frame.
    assert rms_fixed < 1e-3, \
        f"corrected body formula still off: rms={rms_fixed:.3f} mm — math is wrong"
    return (
        f"legacy(-v): rms={rms_legacy:.1f}mm  max={max_legacy:.1f}mm  ← bug in pla.viz.pointcloud  ||  "
        f"corrected(+v): rms={rms_fixed:.1e}mm ✓"
    ), extras


# ---------------------------------------------------------------------------
# diagnostic plots


def plot_synthetic_wall_pointcloud(out_path: Path):
    """Diagnostic: 5 wall distances → pointclouds (camera-frame, OpenGL)."""
    from pla.viz.pointcloud_core import unproject_taxels
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=110, subplot_kw={"projection": "3d"})
    distances = [0.10, 0.25, 0.5, 1.0, 2.0]
    for ax, d in zip(axes, distances):
        depth = np.full((8, 8), d, dtype=np.float32)
        pts, _ = unproject_taxels(depth, intr=INTR, frame="camera")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18, c=-pts[:, 2], cmap="viridis")
        ax.scatter([0], [0], [0], s=80, c="red", marker="X", label="sensor")
        ax.set_title(
            f"d = {d:.2f} m\n(64 pts, half-extent = {d * INTR.half_tan * 1000:.0f} mm)",
            fontsize=9,
        )
        ax.set_xlabel("x_cam"); ax.set_ylabel("y_cam"); ax.set_zlabel("z_cam")
        lim = max(d * INTR.half_tan * 1.3, 0.05)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-max(d * 1.1, 0.05), 0)
    fig.suptitle(
        "T2 — synthetic flat wall: pinhole unprojection (mujoco camera frame, looking -z)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mujoco_wall_diagnostic(out_path: Path, model, data, sensor, depth, pts_world, front_x):
    fig = plt.figure(figsize=(16, 5), dpi=110)
    # 1) depth heatmap
    ax = fig.add_subplot(1, 4, 1)
    im = ax.imshow(depth, cmap="viridis"); plt.colorbar(im, ax=ax, label="depth (m)")
    ax.set_title(f"depth ({sensor.name})\nmean={depth.mean():.3f} m")
    ax.set_xticks([]); ax.set_yticks([])
    # 2) reconstructed YZ (in the wall plane)
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(pts_world[:, 1], pts_world[:, 2], c=pts_world[:, 0] - front_x, cmap="coolwarm",
                s=42, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("world y (m)"); ax.set_ylabel("world z (m)")
    ax.set_title("reconstructed in wall plane (YZ)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    # 3) Δx vs y to expose any tilt
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(pts_world[:, 1], (pts_world[:, 0] - front_x) * 1000, s=22, c="#3a8df0",
                edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="red", ls="--", lw=0.8, label="true wall x")
    ax.set_xlabel("world y (m)"); ax.set_ylabel("Δx vs wall (mm)")
    ax.set_title("residual vs y (look for tilt)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # 4) histogram
    ax = fig.add_subplot(1, 4, 4)
    residuals_mm = (pts_world[:, 0] - front_x) * 1000
    ax.hist(residuals_mm, bins=20, color="#3bbf60", edgecolor="black")
    ax.set_xlabel("Δx vs true wall (mm)"); ax.set_ylabel("count")
    ax.set_title(f"residuals  rms={np.sqrt(np.mean(residuals_mm**2)):.3f} mm")
    ax.grid(alpha=0.3)
    fig.suptitle(
        f"T3 — mujoco wall reconstruction  ({sensor.name}, n={len(pts_world)})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_multi_sensor_coverage(out_path: Path, model, data, sensors):
    """Render world point cloud from all 29 sensors against the box obstacle."""
    all_pts, all_link, all_d = [], [], []
    for s in sensors:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, s.name)
        depth = _render_depth(model, data, s.name, res=INTR.res)
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        pts_world, _ = reconstruct_world_pts(
            depth, cam_pos, cam_R, intr=INTR, frame="camera",
        )
        if len(pts_world) == 0: continue
        all_pts.append(pts_world)
        all_link.extend([s.link] * len(pts_world))
        all_d.append(depth[depth < INTR.zfar_m - 1e-3])
    if not all_pts:
        return
    pts = np.concatenate(all_pts, axis=0)
    link_arr = np.array(all_link)
    # workspace clip: ~1.5 m around base
    inside = (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.5) & (pts[:, 2] < 1.5) & (pts[:, 2] > -0.05)
    fig = plt.figure(figsize=(16, 5), dpi=110)
    obstacle_box = dict(x=(0.15, 0.95), y=(-0.4, 0.4), z=(0.10, 0.90))
    titles = [("XY (top)", 0, 1), ("XZ (side)", 0, 2), ("YZ (front)", 1, 2)]
    for k, (tt, i, j) in enumerate(titles, 1):
        ax = fig.add_subplot(1, 3, k)
        for lk, color in LINK_COLOR.items():
            m = (link_arr == lk) & inside
            if m.any():
                ax.scatter(pts[m, i], pts[m, j], c=color, s=8, alpha=0.6, label=lk,
                           edgecolor="black", linewidth=0.2)
        # draw the obstacle bounding box
        keys = ["x", "y", "z"]
        xa, xb = obstacle_box[keys[i]]
        ya, yb = obstacle_box[keys[j]]
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya,
                                fill=False, edgecolor="red", lw=1.2, label="obstacle"))
        ax.set_xlabel(f"world {keys[i]} (m)"); ax.set_ylabel(f"world {keys[j]} (m)")
        ax.set_title(tt); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        if k == 1:
            ax.legend(fontsize=8, markerscale=2.0, loc="upper right")
    n_inside = int(inside.sum())
    fig.suptitle(
        f"T4 — 29-sensor reconstructed point cloud at home pose  "
        f"({n_inside} pts in 1.5 m workspace, {len(pts) - n_inside} far hits clipped)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pose_invariance(out_path: Path, model, data, qpos_a, qpos_b, front_x):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=110)
    for ax, qp, tag in zip(axes, [qpos_a, qpos_b], ["pose A (home)", "pose B (perturbed)"]):
        _set_qpos(model, data, qp)
        sensors = collect_sensors(model, data)
        best_s, cid = _pick_sensor_facing(sensors, model, data, np.array([1.0, 0, 0]))
        depth = _render_depth(model, data, best_s.name, res=INTR.res)
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        pts, _ = reconstruct_world_pts(depth, cam_pos, cam_R, intr=INTR, frame="camera")
        if len(pts) > 0:
            sc = ax.scatter(pts[:, 1], pts[:, 2], c=(pts[:, 0] - front_x) * 1000,
                             cmap="coolwarm", s=42, edgecolor="black", linewidth=0.4)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Δx vs wall (mm)")
            rms_mm = np.sqrt(np.mean((pts[:, 0] - front_x) ** 2)) * 1000
        else:
            rms_mm = float("nan")
        ax.set_xlabel("world y (m)"); ax.set_ylabel("world z (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{tag}: sensor={best_s.name}  n={len(pts)}  rms={rms_mm:.2f} mm")
    fig.suptitle("T5 — static wall reconstructed from two different arm poses (YZ in-plane)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"=== pointcloud reconstruction tests → {out} ===")
    results: list[TestResult] = []

    print("[ T1 ] pinhole intrinsics")
    results.append(run_check("T1 intrinsics", test_pinhole_intrinsics))

    print("[ T2 ] synthetic flat wall")
    results.append(run_check("T2 synthetic wall", test_synthetic_flat_wall))
    plot_synthetic_wall_pointcloud(out / "T2_synthetic_wall.png")
    print(f"    plot → T2_synthetic_wall.png")

    print("[ T3 ] mujoco wall ground truth (single sensor)")
    extras_t3 = {}
    def _t3():
        nonlocal extras_t3
        msg, extras = test_mujoco_wall_ground_truth(out)
        extras_t3.update(extras)
        return msg
    results.append(run_check("T3 mujoco wall", _t3))
    if extras_t3:
        # re-render for plot
        wall_xml = _patched_model_with_wall(
            out, wall_pos=(0.55, 0.0, 0.50), wall_size=(0.005, 0.40, 0.40),
        )
        model = mujoco.MjModel.from_xml_path(str(wall_xml))
        data = mujoco.MjData(model)
        _set_qpos(model, data, HOME_QPOS)
        sensors = collect_sensors(model, data)
        sensor = next(s for s in sensors if s.name == extras_t3["sensor"])
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, sensor.name)
        depth = _render_depth(model, data, sensor.name, res=INTR.res)
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        pts, _ = reconstruct_world_pts(depth, cam_pos, cam_R, intr=INTR, frame="camera")
        plot_mujoco_wall_diagnostic(
            out / "T3_mujoco_wall.png", model, data, sensor, depth, pts, extras_t3["front_x"],
        )
        print(f"    plot → T3_mujoco_wall.png")

    print("[ T4 ] multi-sensor coverage")
    extras_t4 = {}
    def _t4():
        nonlocal extras_t4
        msg, extras = test_multi_sensor_coverage(out)
        extras_t4.update(extras)
        return msg
    results.append(run_check("T4 multi-sensor", _t4))
    # plot
    wall_xml = _patched_model_with_wall(
        out, wall_pos=(0.55, 0.0, 0.50), wall_size=(0.40, 0.40, 0.40),
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    _set_qpos(model, data, HOME_QPOS)
    sensors = collect_sensors(model, data)
    plot_multi_sensor_coverage(out / "T4_multi_sensor.png", model, data, sensors)
    print(f"    plot → T4_multi_sensor.png")

    print("[ T6 ] legacy depth_axis ↔ camera-frame parity")
    extras_t6 = {}
    def _t6():
        nonlocal extras_t6
        msg, extras = test_legacy_convention_parity(out)
        extras_t6.update(extras)
        return msg
    results.append(run_check("T6 legacy parity", _t6))

    print("[ T5 ] pose invariance")
    extras_t5 = {}
    def _t5():
        nonlocal extras_t5
        msg, extras = test_pose_invariance(out)
        extras_t5.update(extras)
        return msg
    results.append(run_check("T5 pose invariance", _t5))
    # plot
    wall_xml = _patched_model_with_wall(
        out, wall_pos=(0.55, 0.0, 0.50), wall_size=(0.005, 0.40, 0.40),
    )
    model = mujoco.MjModel.from_xml_path(str(wall_xml))
    data = mujoco.MjData(model)
    poses = [HOME_QPOS, HOME_QPOS + np.array([0.3, 0.1, -0.2, 0.1, 0.0, 0.0, 0.0])]
    plot_pose_invariance(out / "T5_pose_invariance.png", model, data, poses[0], poses[1],
                          front_x=0.55 - 0.005)
    print(f"    plot → T5_pose_invariance.png")

    # summary
    n_pass = sum(r.passed for r in results)
    print(f"\n=== summary: {n_pass}/{len(results)} passed ===")
    # Write a JSON report
    import json
    report = {
        "results": [
            {"name": r.name, "passed": r.passed, "message": r.message}
            for r in results
        ],
        "extras": {"T3": extras_t3, "T4": extras_t4, "T5": extras_t5, "T6": extras_t6},
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  report → results.json")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
