"""
FR3 + self-cap skin in MuJoCo with SPAD-like proximity cameras.

Two stages:

1. `build_mjcf(...)` parses the URDF and emits an MJCF that
   - rebuilds the kinematic chain as nested bodies (revolute joints → hinges),
   - uses the collision STLs for rendering (MuJoCo can't load the .dae visuals),
   - adds one fixed camera + site per URDF sensor frame.
   Each sensor's joint origin (rpy + xyz) defines its frame, with body +Z
   taken as the outward pointing direction. The camera inside the sensor body
   is rotated 180° about X so that MuJoCo's default -Z view direction aligns
   with body +Z.

2. `render_depth(...)` loads the MJCF, sets an optional qpos, and renders an
   8×8 depth image per camera with `mujoco.Renderer` in depth mode. That mimics
   a VL53L5CX-class multi-zone SPAD proximity sensor.

Run:
    python -m pla.sim.build_mjcf \\
        --urdf assets/urdf/fr3_full_skin.urdf \\
        --meshdir /path/to/gentact_ros_tools/meshes \\
        --out assets/mjcf/fr3_skin.xml \\
        --view            # open interactive viewer
    python -m pla.sim.build_mjcf --out assets/mjcf/fr3_skin.xml --render  # depth dump
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np


PACKAGE_PREFIX = "package://gentact_ros_tools/meshes/"


def _vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=float)


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy (fixed-axis X→Y→Z) → MuJoCo quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = R.from_euler("xyz", rpy).as_quat()
    return np.array([w, x, y, z])


def _fmt(a: np.ndarray) -> str:
    return " ".join(f"{v:.8g}" for v in a)


def _mesh_filename(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    mesh = elem.find(".//mesh")
    if mesh is None:
        return None
    fn = mesh.get("filename", "").removeprefix(PACKAGE_PREFIX)
    # Some URDFs drop the `package://` prefix but still start with a bare
    # `meshes/` segment. The MJCF compiler's meshdir already points at the
    # gentact `meshes/` directory, so strip the duplicate here too.
    return fn.removeprefix("meshes/")


def _ensure_mujoco_mesh(
    rel_filename: str, meshdir: Path, cache_dir: Path
) -> str | None:
    """Return a path suitable for MJCF `<mesh file="...">`.

    `.stl`/`.obj`/`.msh` are returned unchanged (resolved against meshdir).
    `.dae` files are converted to `.obj` via trimesh and cached in
    `cache_dir` — the absolute path is returned, which MuJoCo accepts in
    place of a meshdir-relative path.

    Returns None if the source file is missing or conversion fails.
    """
    ext = Path(rel_filename).suffix.lower()
    if ext in (".stl", ".obj", ".msh"):
        return rel_filename
    if ext != ".dae":
        return None

    src = meshdir / rel_filename
    if not src.exists():
        return None
    cached = (cache_dir / rel_filename).with_suffix(".obj")
    if not cached.exists() or cached.stat().st_mtime < src.stat().st_mtime:
        try:
            import trimesh

            mesh = trimesh.load(str(src), force="mesh")
            cached.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(cached))
        except Exception as e:
            print(f"[warn] failed to convert {rel_filename}: {e}")
            return None
    return str(cached.resolve())


def build_mjcf(
    urdf_path: Path,
    meshdir: Path,
    out_path: Path,
    *,
    fovy_deg: float = 45.0,
    resolution: tuple[int, int] = (8, 8),
    znear: float = 0.02,
    zfar: float = 4.0,
) -> Path:
    """Generate an MJCF from the URDF and write it to ``out_path``."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot", f"expected <robot>, got <{root.tag}>"

    links = {l.get("name"): l for l in root.findall("link")}
    joints = list(root.findall("joint"))

    children_of: dict[str, list[ET.Element]] = {}
    parent_of: dict[str, str] = {}
    mimic_pairs: list[tuple[str, str, float]] = []  # (child_joint, parent_joint, multiplier)
    for j in joints:
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        children_of.setdefault(parent, []).append(j)
        parent_of[child] = parent
        mimic = j.find("mimic")
        if mimic is not None:
            mimic_pairs.append(
                (
                    j.get("name"),
                    mimic.get("joint"),
                    float(mimic.get("multiplier", "1")),
                )
            )

    root_links = [n for n in links if n not in parent_of]
    assert len(root_links) == 1, f"expected a single root link, got {root_links}"
    root_link = root_links[0]

    mj = ET.Element("mujoco", {"model": "fr3_skin"})
    ET.SubElement(
        mj,
        "compiler",
        {
            "angle": "radian",
            "autolimits": "true",
            "meshdir": str(meshdir),
            # URDF inertia frames are not always principal axes.
            "balanceinertia": "true",
            "boundmass": "1e-6",
            "boundinertia": "1e-9",
        },
    )
    option = ET.SubElement(
        mj, "option", {"integrator": "implicitfast", "timestep": "0.002"}
    )
    # Disable gravity: this model has no actuators or joint stiffness, so
    # under gravity the arm immediately collapses in the viewer. Since the
    # script exists to inspect a static pose and render depth, a frozen
    # kinematic model is what we want.
    ET.SubElement(option, "flag", {"gravity": "disable"})

    # --- default block -----------------------------------------------------
    default = ET.SubElement(mj, "default")
    ET.SubElement(
        default,
        "geom",
        {"contype": "1", "conaffinity": "1", "rgba": "0.75 0.75 0.78 1"},
    )
    skin_default = ET.SubElement(default, "default", {"class": "skin"})
    ET.SubElement(
        skin_default,
        "geom",
        {"rgba": "0.25 0.55 0.85 0.35", "contype": "0", "conaffinity": "0"},
    )

    # --- assets: meshes ----------------------------------------------------
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.3 0.3 0.35",
            "rgb2": "0.2 0.2 0.25",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "grid",
            "texture": "grid",
            "texrepeat": "5 5",
            "reflectance": "0.1",
        },
    )

    mesh_name_for: dict[str, str] = {}
    used: set[str] = set()
    mesh_cache_dir = out_path.parent / ".mesh_cache"

    def add_mesh(filename: str, name: str):
        if name in used:
            return
        used.add(name)
        ET.SubElement(asset, "mesh", {"name": name, "file": filename})

    for link_name, link in links.items():
        # Prefer the visual mesh (detailed .dae, converted to .obj if needed)
        # so skin geometry — which was authored against the visual shape —
        # fits snugly. Fall back to the collision STL if visual is missing
        # or conversion fails.
        vfn = _mesh_filename(link.find("visual"))
        fn: str | None = None
        if vfn is not None:
            fn = _ensure_mujoco_mesh(vfn, meshdir, mesh_cache_dir)
        if fn is None:
            fn = _mesh_filename(link.find("collision"))
        if fn is None:
            continue
        # Name by the source stem so skin vs robot meshes don't collide.
        mname = Path(vfn if vfn is not None else fn).stem
        add_mesh(fn, mname)
        mesh_name_for[link_name] = mname

    # --- worldbody + kinematic chain --------------------------------------
    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0 0 3", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.05",
            "material": "grid",
            "pos": "0 0 0",
        },
    )

    def is_sensor(name: str) -> bool:
        return "_sensor_" in name

    def is_skin(name: str) -> bool:
        return name.endswith("_skin")

    def add_body(
        parent_elem: ET.Element,
        link_name: str,
        pos: np.ndarray,
        quat: np.ndarray,
        parent_joint: ET.Element | None,
    ) -> None:
        attrs = {"name": link_name, "pos": _fmt(pos), "quat": _fmt(quat)}
        body = ET.SubElement(parent_elem, "body", attrs)

        # Revolute joint from parent → this link.
        if parent_joint is not None and parent_joint.get("type") == "revolute":
            jname = parent_joint.get("name")
            axis = _vec(parent_joint.find("axis").get("xyz"))
            lim = parent_joint.find("limit")
            jattrs = {
                "name": jname,
                "type": "hinge",
                "axis": _fmt(axis),
                "range": f"{lim.get('lower')} {lim.get('upper')}",
            }
            dyn = parent_joint.find("dynamics")
            if dyn is not None and dyn.get("damping"):
                jattrs["damping"] = dyn.get("damping")
            ET.SubElement(body, "joint", jattrs)

        # Inertial (skin and sensor dummies have none).
        link = links[link_name]
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            iorg = inertial.find("origin")
            ipos = _vec(iorg.get("xyz", "0 0 0")) if iorg is not None else np.zeros(3)
            ine = link.find("inertial/inertia")
            ixx = float(ine.get("ixx")); iyy = float(ine.get("iyy"))
            izz = float(ine.get("izz")); ixy = float(ine.get("ixy"))
            ixz = float(ine.get("ixz")); iyz = float(ine.get("iyz"))
            # URDF inertia is about the inertial origin frame; URDF rpy is 0 for
            # every FR3 link in this file (verified), so no reorientation needed.
            # `fullinertia` and an <inertial quat> are mutually exclusive in MJCF.
            # Skip degenerate inertials (e.g. robotiq_base_link has all zeros) —
            # MuJoCo will fall back to boundmass / boundinertia defaults.
            if max(ixx, iyy, izz) > 0.0:
                ET.SubElement(
                    body,
                    "inertial",
                    {
                        "pos": _fmt(ipos),
                        "mass": f"{mass:.8g}",
                        "fullinertia": f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}",
                    },
                )

        # Mesh geom.
        if link_name in mesh_name_for:
            geom_attrs = {"type": "mesh", "mesh": mesh_name_for[link_name]}
            if is_skin(link_name):
                geom_attrs["class"] = "skin"
            body.append(ET.Element("geom", geom_attrs))

        # Sensor-frame hardware: a small red site for visual debug + a fixed
        # camera flipped so -Z_camera = +Z_body (i.e. +Z_body is the ray dir).
        if is_sensor(link_name):
            ET.SubElement(
                body,
                "site",
                {
                    "name": link_name + "_site",
                    "type": "sphere",
                    "size": "0.004",
                    "rgba": "1 0.2 0.2 1",
                },
            )
            ET.SubElement(
                body,
                "camera",
                {
                    "name": link_name,
                    "mode": "fixed",
                    "pos": "0 0 0",
                    "quat": "0 1 0 0",  # 180° about X
                    "fovy": f"{fovy_deg}",
                    "resolution": f"{resolution[0]} {resolution[1]}",
                },
            )

        # Recurse into children.
        for cj in children_of.get(link_name, []):
            child_name = cj.find("child").get("link")
            corg = cj.find("origin")
            if corg is not None:
                cpos = _vec(corg.get("xyz", "0 0 0"))
                crpy = _vec(corg.get("rpy", "0 0 0"))
            else:
                cpos = np.zeros(3)
                crpy = np.zeros(3)
            cquat = _rpy_to_quat(crpy)
            add_body(body, child_name, cpos, cquat, cj)

    add_body(worldbody, root_link, np.zeros(3), np.array([1, 0, 0, 0]), None)

    # --- equality constraints from URDF <mimic> ---------------------------
    # Every revolute joint is emitted as a real MuJoCo hinge (so it owns a
    # qpos slot). URDF `<mimic>` is encoded as an equality/joint constraint:
    # child = multiplier * parent + offset (offset is 0 for all robotiq joints).
    if mimic_pairs:
        equality = ET.SubElement(mj, "equality")
        for child_j, parent_j, mult in mimic_pairs:
            ET.SubElement(
                equality,
                "joint",
                {
                    "joint1": child_j,
                    "joint2": parent_j,
                    # polycoef = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
                    # we want joint1 = mult * joint2 → a0=0, a1=mult.
                    "polycoef": f"0 {mult} 0 0 0",
                },
            )

    # --- keyframe: Franka "ready" pose ------------------------------------
    # Default qpos=0 puts several sensors looking at empty sky; this gives a
    # more interesting starting configuration to inspect in the viewer.
    # Pad with zeros for any gripper / extra joints beyond the 7 FR3 joints.
    home_fr3 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    n_revolute = sum(1 for j in joints if j.get("type") == "revolute")
    home_qpos = home_fr3 + [0.0] * max(0, n_revolute - len(home_fr3))
    keyframe = ET.SubElement(mj, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": " ".join(f"{v:.6g}" for v in home_qpos)},
    )

    # --- visual / renderer clip planes ------------------------------------
    visual = ET.SubElement(mj, "visual")
    # znear is a *fraction* of the model extent in MuJoCo. 0.005 × ~1.2 m ≈ 6 mm.
    ET.SubElement(visual, "map", {"znear": "0.005", "zfar": "10"})
    ET.SubElement(
        visual,
        "global",
        {"offwidth": "1280", "offheight": "960"},
    )

    # Save pretty-printed XML.
    rough = ET.tostring(mj, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    out_path.write_text(pretty)

    # Stash the user's requested SPAD clip range as a sidecar so the renderer
    # can clamp. (MuJoCo's per-camera near/far isn't honored by the renderer.)
    meta = {"znear": znear, "zfar": zfar, "resolution": list(resolution)}
    (out_path.with_suffix(".meta.json")).write_text(
        __import__("json").dumps(meta, indent=2)
    )
    return out_path


# ---------------------------------------------------------------------------
# Rendering / data collection
# ---------------------------------------------------------------------------


def render_depth(
    model_xml: Path,
    qpos: np.ndarray | None = None,
    keyframe: str | None = "home",
) -> dict[str, np.ndarray]:
    """Render an (H, W) depth image in metres for every `<camera>` in the model.

    If `qpos` is None and a keyframe with the given name exists, that keyframe
    is used. Pass `keyframe=None` to keep the default zero-pose.
    """
    import json

    import mujoco

    meta_path = model_xml.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    znear = float(meta.get("znear", 0.02))
    zfar = float(meta.get("zfar", 4.0))
    h, w = meta.get("resolution", [8, 8])

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[: len(qpos)] = qpos
    elif keyframe is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    readings: dict[str, np.ndarray] = {}
    with mujoco.Renderer(model, height=h, width=w) as renderer:
        renderer.enable_depth_rendering()
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            renderer.update_scene(data, camera=name)
            depth = renderer.render().copy()
            depth[depth < znear] = np.nan
            depth[depth > zfar] = np.nan
            readings[name] = depth
    return readings


def view(
    model_xml: Path,
    qpos: np.ndarray | None = None,
    keyframe: str | None = "home",
) -> None:
    """Launch the passive viewer so you can inspect the model + sensor sites."""
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[: len(qpos)] = qpos
    elif keyframe is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


def save_images(
    model_xml: Path,
    out_dir: Path,
    qpos: np.ndarray | None = None,
    keyframe: str | None = "home",
) -> None:
    """Save a third-person RGB render and a grid of per-sensor depth maps.

    Produces two PNGs in ``out_dir``:
    - ``fr3_skin_scene.png``: wide third-person shot of the robot in its pose,
      with the red sensor sites visible.
    - ``fr3_skin_depths.png``: grid of every sensor's 8×8 depth map.
    """
    import json

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mujoco

    meta_path = model_xml.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    znear = float(meta.get("znear", 0.02))
    zfar = float(meta.get("zfar", 4.0))

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    # Bump sensor-site markers so they are visible in the third-person render.
    # This only affects the in-memory model used for the scene shot; the
    # subsequent depth render reloads the model from disk.
    model.site_size[:, 0] *= 4
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[: len(qpos)] = qpos
    elif keyframe is not None:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Third-person scene render using the model's default free camera.
    # Capped at the MJCF's offscreen buffer size (offwidth=1280, offheight=960).
    scene_h, scene_w = 960, 1280
    with mujoco.Renderer(model, height=scene_h, width=scene_w) as renderer:
        renderer.update_scene(data)
        rgb = renderer.render()
    scene_path = out_dir / "fr3_skin_scene.png"
    plt.imsave(scene_path, rgb)
    print(f"[save] scene  → {scene_path}")

    # Per-sensor depth grid.
    readings = render_depth(model_xml, qpos=qpos, keyframe=keyframe)
    finite_vals = [d[np.isfinite(d)] for d in readings.values() if np.isfinite(d).any()]
    if finite_vals:
        all_finite = np.concatenate(finite_vals)
        vmin, vmax = float(all_finite.min()), float(all_finite.max())
    else:
        vmin, vmax = znear, zfar

    cmap = mpl.colormaps.get_cmap("viridis_r").copy()
    cmap.set_bad("lightgray")

    n = len(readings)
    ncols = 7
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.6, nrows * 1.7), squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, (name, depth) in zip(axes_flat, readings.items()):
        ax.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name.replace("_sensor_", "_s"), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(readings):]:
        ax.axis("off")
    fig.suptitle(
        f"Per-sensor depth (m). Range [{vmin:.2f}, {vmax:.2f}]. Grey = out of range.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    depths_path = out_dir / "fr3_skin_depths.png"
    fig.savefig(depths_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] depths → {depths_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_meshdir() -> Path:
    # User can override with --meshdir. Fallback to a neighbour `meshes/` dir.
    return Path(__file__).resolve().parent / "meshes"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", type=Path, default=Path(__file__).parent / "fr3_full_skin.urdf")
    p.add_argument("--meshdir", type=Path, default=_default_meshdir())
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "fr3_skin.xml")
    p.add_argument("--build", action="store_true", help="(re)generate MJCF from URDF")
    p.add_argument("--view", action="store_true", help="open the interactive viewer")
    p.add_argument("--render", action="store_true", help="render depth from every sensor camera")
    p.add_argument(
        "--save-images",
        action="store_true",
        help="save PNG renders: third-person RGB + grid of per-sensor depth maps",
    )
    p.add_argument("--fovy", type=float, default=45.0)
    p.add_argument("--res", type=int, nargs=2, default=[8, 8], metavar=("H", "W"))
    p.add_argument("--znear", type=float, default=0.02)
    p.add_argument("--zfar", type=float, default=4.0)
    args = p.parse_args()

    if args.build or not args.out.exists():
        print(f"[build] {args.urdf} → {args.out}  (meshdir={args.meshdir})")
        build_mjcf(
            args.urdf,
            args.meshdir,
            args.out,
            fovy_deg=args.fovy,
            resolution=tuple(args.res),
            znear=args.znear,
            zfar=args.zfar,
        )

    if args.view:
        view(args.out)

    if args.render:
        readings = render_depth(args.out)
        print(f"[render] {len(readings)} sensor cameras")
        for name, depth in readings.items():
            finite = np.isfinite(depth)
            if finite.any():
                mn, mx, mean = depth[finite].min(), depth[finite].max(), depth[finite].mean()
                print(f"  {name:<20}  min={mn:.3f}  max={mx:.3f}  mean={mean:.3f} m  ({finite.sum()}/{depth.size} valid)")
            else:
                print(f"  {name:<20}  (no returns in range)")

    if args.save_images:
        save_images(args.out, args.out.parent)


if __name__ == "__main__":
    main()
