"""Render the FR3 robot with all 29 PLA proximity sensors color-coded by link.

Produces annotated reference images that *explicitly* show where the
VL53L5CX patches sit on the robot. The unmodified renders in
`assets/reference_images/franka_skin_*.png` show the FR3 mesh but the
sensor sites are too small to spot — these annotated copies overlay a
colored disk per sensor (color = link, label = MJCF index).

Outputs
-------
`assets/reference_images/annotated/`:
  - `skin_overlay_az<DEG>_el<DEG>.png` — one per requested viewpoint.
  - `skin_overlay_legend.png` — combined 2x2 grid + legend strip.
  - `sensor_layout_table.csv` — index, name, link, body-frame xyz.

The mesh paths in the on-disk MJCF point to a deleted cache; we patch
them at runtime to the on-disk gentact + cached molmo-spaces meshes.

Run
---
    python -m pla.viz.sensor_overlay
    python -m pla.viz.sensor_overlay --azimuths 0,90,180,270 --width 800
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
MJCF_SRC = REPO / "assets/mjcf/fr3_skin.xml"
FR3_MESH_DIR = Path("/home/jaydv/code/molmo/molmospaces/assets/robots/franka_fr3/assets")
SKIN_MESH_DIR = Path(
    "/home/jaydv/blender_ws/install/gentact_ros_tools_hybrid/share/"
    "gentact_ros_tools_hybrid/meshes"
)
OUT_DIR = REPO / "assets/reference_images/annotated"

LINK_COLOR = {
    "link2": "#e84141",  # red
    "link3": "#ffa130",  # orange
    "link5": "#3bbf60",  # green
    "link6": "#3a8df0",  # blue
}
LINK_COUNT = {"link2": 7, "link3": 8, "link5": 6, "link6": 8}
TOTAL_SENSORS = sum(LINK_COUNT.values())  # 29

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


@dataclass
class Sensor:
    name: str
    link: str
    idx_in_link: int
    body_id: int
    world_pos: np.ndarray  # (3,)
    world_normal: np.ndarray  # (3,) outward (sensor +Z in body frame)


def patch_mjcf(src: Path, dst: Path, cameras: list[dict] | None = None) -> Path:
    """Rewrite mesh file paths so the on-disk MJCF actually loads.

    If ``cameras`` is supplied, also inject one fixed camera per dict
    (``name``, ``pos``, ``xyaxes``) into the worldbody. Using a fixed
    camera lets mujoco compute the camera frame for us, avoiding the
    free-camera azimuth/elevation math entirely.
    """
    text = src.read_text()
    for i in range(8):
        old = (
            "/home/jaydv/code/molmo/fr3_skin_mujoco/"
            f".mesh_cache/robot_arms/fr3/visual/link{i}.obj"
        )
        new = str(FR3_MESH_DIR / f"link{i}.obj")
        text = text.replace(old, new)
    for L in (2, 3, 5, 6):
        text = text.replace(
            f"skin/self-cap/link{L}_fancy.stl",
            str(SKIN_MESH_DIR / f"skin/self-cap/link{L}_fancy.stl"),
        )
    if cameras:
        cam_xml = ""
        for c in cameras:
            cam_xml += (
                f'    <camera name="{c["name"]}" mode="fixed"\n'
                f'            pos="{c["pos"][0]:.4f} {c["pos"][1]:.4f} {c["pos"][2]:.4f}"\n'
                f'            xyaxes="{c["xyaxes"][0]:.4f} {c["xyaxes"][1]:.4f} {c["xyaxes"][2]:.4f} '
                f'{c["xyaxes"][3]:.4f} {c["xyaxes"][4]:.4f} {c["xyaxes"][5]:.4f}"\n'
                f'            fovy="{c.get("fovy", 45.0)}"/>\n'
            )
        text = text.replace("<worldbody>", "<worldbody>\n" + cam_xml, 1)
    dst.write_text(text)
    return dst


def make_orbit_camera(name: str, azimuth_deg: float, elevation_deg: float,
                       distance: float, lookat: np.ndarray, fovy: float = 45.0) -> dict:
    """Build a fixed-camera dict for an orbit pose around ``lookat``.

    Convention here (independent of mujoco free-camera): azimuth is the
    yaw angle CCW around world Z measured from world +X. So az=0 places
    the camera on the +X side looking back toward -X.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    # camera position on a sphere around lookat
    offset = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ]) * distance
    cam_pos = lookat + offset
    forward = (lookat - cam_pos)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(forward, world_up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(-forward, x_axis)  # mujoco fixed cam: -z = forward
    y_axis /= np.linalg.norm(y_axis)
    return {
        "name": name,
        "pos": cam_pos.tolist(),
        "xyaxes": [*x_axis.tolist(), *y_axis.tolist()],
        "fovy": fovy,
    }


def collect_sensors(model: mujoco.MjModel, data: mujoco.MjData) -> list[Sensor]:
    out: list[Sensor] = []
    for link, n in LINK_COUNT.items():
        for i in range(n):
            name = f"{link}_sensor_{i}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"missing body {name}")
            R = data.xmat[bid].reshape(3, 3)
            # The MJCF cameras use quat="0 1 0 0" (180° about X), so the camera
            # looks down -Z_body and the body's *outward* face is along -Z_body.
            normal = -R[:, 2]
            out.append(Sensor(
                name, link, i, bid,
                data.xpos[bid].copy(),
                normal / (np.linalg.norm(normal) + 1e-9),
            ))
    if len(out) != TOTAL_SENSORS:
        raise RuntimeError(f"expected {TOTAL_SENSORS} sensors, got {len(out)}")
    return out


def project_world_to_pixel(
    p_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,  # 3x3, columns = [x_cam, y_cam, z_cam] in world
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    """Project a world point into image pixels for a mujoco camera.

    mujoco camera convention: -z is forward (looking direction),
    +y is up, +x is right (in camera frame).
    """
    rel = p_world - cam_pos
    p_cam = cam_mat.T @ rel  # world delta → camera-frame
    x, y, z = p_cam
    if z >= 0:  # behind camera
        return None
    depth = -z
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    u = width * 0.5 + f * (x / depth)
    v = height * 0.5 - f * (y / depth)
    return (float(u), float(v), float(depth))


def render_with_fixed_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Render with a named fixed camera; return image + cam_pos + cam_mat + fovy.

    Reads the camera pose directly from mj_kinematics-computed
    ``data.cam_xpos`` / ``data.cam_xmat`` so the projection math always
    matches the renderer.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise RuntimeError(f"camera not found: {cam_name}")
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    cam_pos = data.cam_xpos[cid].copy()
    cam_mat = data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = float(model.cam_fovy[cid])
    renderer.close()
    return img, cam_pos, cam_mat, fovy


def annotate_image(
    img: np.ndarray,
    sensors: list[Sensor],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    out_path: Path,
    title: str,
) -> int:
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=120)
    ax.imshow(img)
    n_visible = 0
    # depth-sort so back sensors render first
    items = []
    for s in sensors:
        proj = project_world_to_pixel(s.world_pos, cam_pos, cam_mat, fovy_deg, w, h)
        if proj is None:
            continue
        u, v, depth = proj
        if not (0 <= u < w and 0 <= v < h):
            continue
        items.append((depth, u, v, s))
    # camera viewing direction in world frame (camera looks down -z_cam)
    view_dir = -cam_mat[:, 2]
    items.sort(reverse=True, key=lambda t: t[0])  # far first
    for _, u, v, s in items:
        # Cull sensors whose outward normal faces away from the camera
        # (back-of-link patches we shouldn't see through the mesh).
        facing = float(s.world_normal @ (-view_dir))  # >0 = pointing toward camera
        front = facing > -0.05  # slight tolerance for grazing patches
        if not front:
            continue
        ax.scatter(
            [u], [v], s=130,
            facecolor=LINK_COLOR[s.link],
            edgecolor="black", linewidth=1.0, alpha=0.95, zorder=3,
        )
        ax.annotate(
            str(s.idx_in_link),
            (u, v), color="white", ha="center", va="center",
            fontsize=7, fontweight="bold", zorder=4,
        )
        n_visible += 1
    legend = [Patch(facecolor=c, edgecolor="black",
                    label=f"{lk}  ({LINK_COUNT[lk]} sensors)")
              for lk, c in LINK_COLOR.items()]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return n_visible


def write_layout_table(sensors: list[Sensor], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mjcf_index", "name", "link", "idx_in_link",
                      "world_x_m", "world_y_m", "world_z_m"])
        for i, s in enumerate(sensors):
            wr.writerow([i, s.name, s.link, s.idx_in_link,
                         f"{s.world_pos[0]:.4f}",
                         f"{s.world_pos[1]:.4f}",
                         f"{s.world_pos[2]:.4f}"])


def make_grid_figure(items: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=110)
    for ax, (title, img) in zip(axes.flat, items):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    legend = [Patch(facecolor=c, edgecolor="black",
                    label=f"{lk}  ({LINK_COUNT[lk]} sensors)")
              for lk, c in LINK_COLOR.items()]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "FR3 + GenTact whole-body proximity skin: 29 VL53L5CX 8x8 patches",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuths", default="-180,-90,0,90",
                    help="comma-sep camera azimuths in degrees (matches az suffix in original images)")
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=1.6)
    ap.add_argument("--lookat", default="0.30,0.0,0.55",
                    help="look-at point x,y,z in world frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--qpos", default=",".join(f"{q:.3f}" for q in HOME_QPOS),
                    help="7-DoF arm pose (rad)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    azimuths = [float(x) for x in args.azimuths.split(",")]
    qpos = np.array([float(x) for x in args.qpos.split(",")], dtype=np.float64)
    assert len(qpos) == 7, "qpos must be 7 values"
    lookat = np.array([float(x) for x in args.lookat.split(",")], dtype=np.float64)

    cameras = [
        make_orbit_camera(
            name=f"orbit_az{int(az):+04d}",
            azimuth_deg=az, elevation_deg=args.elevation,
            distance=args.distance, lookat=lookat,
        )
        for az in azimuths
    ]
    patched = patch_mjcf(
        MJCF_SRC, out_dir / "_fr3_skin_patched.xml", cameras=cameras,
    )
    model = mujoco.MjModel.from_xml_path(str(patched))
    data = mujoco.MjData(model)
    # set arm joints (first 7 hinges named fr3_joint1..7)
    for i in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i+1}")
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = qpos[i]
    mujoco.mj_forward(model, data)

    sensors = collect_sensors(model, data)
    write_layout_table(sensors, out_dir / "sensor_layout_table.csv")
    print(f"[ok] collected {len(sensors)} sensors @ home pose")

    grid_items = []
    for az, cam in zip(azimuths, cameras):
        img, cam_pos, cam_mat, fovy = render_with_fixed_camera(
            model, data, cam_name=cam["name"],
            width=args.width, height=args.height,
        )
        out_png = out_dir / f"skin_overlay_az{int(az):+04d}_el{int(args.elevation):+03d}.png"
        title = (f"FR3 + GenTact skin  |  azimuth = {az:+.0f}°  "
                 f"elevation = {args.elevation:+.0f}°  |  29 ToF patches")
        n_vis = annotate_image(img, sensors, cam_pos, cam_mat, fovy, out_png, title)
        print(f"[ok] {out_png.name}  visible={n_vis}/{len(sensors)}")
        grid_items.append((f"az = {az:+.0f}°", plt.imread(out_png)))

    make_grid_figure(grid_items, out_dir / "skin_overlay_legend.png")
    print(f"[ok] grid → {out_dir / 'skin_overlay_legend.png'}")


if __name__ == "__main__":
    main()
