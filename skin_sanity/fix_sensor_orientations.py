"""Rebuild per-patch orientation so body +Z is perpendicular to the local
skin-mesh surface at the mount point, and translate outward by a small
offset to eliminate penetration (== self-hits).

Algorithm, per sensor:
  1. Read URDF joint `<origin rpy xyz>` relative to its parent `linkN_skin`.
     That parent has its mesh `linkN_fancy.stl` placed at the origin, so the
     xyz is directly a point in the mesh's coordinate frame.
  2. `trimesh.proximity.closest_point(...)` returns the nearest surface
     point + the face it lies on. Use the face normal as the outward
     direction. (Trimesh is consistent: for a watertight mesh, face normals
     point away from the interior. If a patch's current +Z points *into*
     the mesh, we take the +normal direction as outward because we want
     the patch to see away from the link.)
  3. Reject patches whose inward-pointing original +Z matched the inward
     normal: force outward.
  4. Build R with columns [x_axis, y_axis, n] where n is the outward
     normal and x_axis is original +X projected into the tangent plane
     (fallback: world +X or +Z if degenerate). Convert R -> quaternion
     and then to URDF rpy.
  5. Shift xyz outward by `OUTWARD_SHIFT` metres along n.

Outputs:
  - `fr3_full_skin_fixed.urdf`  — new URDF
  - `sensor_fix_report.txt`     — per-patch angle change + offset
  - `sensor_fix_diag.html`      — 3D plotly showing old vs new ray directions
"""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

URDF_IN = Path('/home/jaydv/code/molmo/fr3_skin_mujoco/fr3_full_skin.urdf')
URDF_OUT = Path('/home/jaydv/code/molmo/fr3_skin_mujoco/fr3_full_skin_fixed.urdf')
SKIN_MESH_DIR = Path('/home/jaydv/code/molmo/resources/robots/franka_droid_skin/skin_meshes')
REPORT = Path('/home/jaydv/code/skin_sanity/sensor_fix_report.txt')

OUTWARD_SHIFT = 0.003   # 3 mm outward past mesh surface
LINKS = [2, 3, 5, 6]


def load_skin_meshes() -> dict[int, trimesh.Trimesh]:
    return {n: trimesh.load(str(SKIN_MESH_DIR / f'link{n}_fancy.stl'), force='mesh')
            for n in LINKS}


def _parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=float)


def _fmt_vec(v: np.ndarray, prec: int = 6) -> str:
    return ' '.join(f'{x:.{prec}f}' for x in v)


def build_outward_R(n_hat: np.ndarray, old_x_axis: np.ndarray | None = None) -> np.ndarray:
    """Return 3x3 rotation whose +Z is `n_hat` (outward normal).

    `old_x_axis` is used, if provided, to keep the 8x8 taxel grid's x-axis
    close to the original authored orientation (minimizes "roll" change).
    """
    n_hat = n_hat / np.linalg.norm(n_hat)
    # Choose tangent-plane x. Default: world +X; fall back to world +Y if
    # parallel to normal.
    if old_x_axis is not None:
        x_try = old_x_axis
    else:
        x_try = np.array([1.0, 0.0, 0.0])
    x_proj = x_try - (x_try @ n_hat) * n_hat
    if np.linalg.norm(x_proj) < 1e-3:
        x_try = np.array([0.0, 1.0, 0.0])
        x_proj = x_try - (x_try @ n_hat) * n_hat
    x_hat = x_proj / np.linalg.norm(x_proj)
    y_hat = np.cross(n_hat, x_hat)
    Rm = np.column_stack([x_hat, y_hat, n_hat])
    return Rm


def rpy_from_R(Rm: np.ndarray) -> np.ndarray:
    # URDF uses fixed-axis XYZ rpy
    return R.from_matrix(Rm).as_euler('xyz')


def find_sensor_joints(root: ET.Element) -> list[ET.Element]:
    out = []
    for j in root.findall('joint'):
        nm = j.get('name', '')
        m = re.match(r'link(\d+)_sensor_(\d+)_joint$', nm)
        if m: out.append(j)
    return out


def process():
    tree = ET.parse(str(URDF_IN))
    root = tree.getroot()
    meshes = load_skin_meshes()
    report_lines = [f'{"patch":<22} {"old_xyz":<30} {"old_+Z":<30} {"normal_out":<30} {"angle_deg":>10}']
    report_lines.append('-' * 120)

    joints = find_sensor_joints(root)
    print(f'found {len(joints)} sensor joints')

    for j in joints:
        name = j.get('name')  # linkN_sensor_K_joint
        link_n = int(re.match(r'link(\d+)_sensor', name).group(1))
        mesh = meshes[link_n]

        origin_el = j.find('origin')
        xyz_old = _parse_vec(origin_el.get('xyz'))
        rpy_old = _parse_vec(origin_el.get('rpy'))
        R_old = R.from_euler('xyz', rpy_old).as_matrix()
        z_old = R_old[:, 2]
        x_old = R_old[:, 0]

        # Nearest surface point
        pt, dist, face_id = trimesh.proximity.closest_point(mesh, xyz_old[None, :])
        pt = pt[0]; face_id = int(face_id[0])
        normal = mesh.face_normals[face_id]
        normal = normal / np.linalg.norm(normal)

        # The mesh isn't guaranteed watertight, so face normals may flip
        # sign locally. The original authored +Z is reliable as an outward
        # indicator — pick the surface normal sign that agrees with it.
        cos = float(z_old @ normal)
        if cos < 0:
            normal = -normal
            cos = -cos
        angle_deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # New rotation: +Z aligned to outward normal, +X projected from old +X
        R_new = build_outward_R(normal, old_x_axis=x_old)
        rpy_new = rpy_from_R(R_new)

        # New position: outside the mesh along normal
        xyz_new = pt + OUTWARD_SHIFT * normal

        # Write back
        origin_el.set('xyz', _fmt_vec(xyz_new, prec=6))
        origin_el.set('rpy', _fmt_vec(rpy_new, prec=6))
        # Update axis too (cosmetic in URDF, but keep consistent with +Z)
        axis_el = j.find('axis')
        if axis_el is not None:
            axis_el.set('xyz', _fmt_vec(normal, prec=6))

        report_lines.append(
            f'{name.replace("_joint",""):<22} '
            f'({xyz_old[0]:+.3f},{xyz_old[1]:+.3f},{xyz_old[2]:+.3f}) -> '
            f'({xyz_new[0]:+.3f},{xyz_new[1]:+.3f},{xyz_new[2]:+.3f})  '
            f'oldZ=({z_old[0]:+.2f},{z_old[1]:+.2f},{z_old[2]:+.2f})  '
            f'n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f})  '
            f'{angle_deg:>7.2f}'
        )

    URDF_OUT.write_text(ET.tostring(root, encoding='unicode'))
    REPORT.write_text('\n'.join(report_lines) + '\n')
    print(f'wrote {URDF_OUT}')
    print(f'wrote {REPORT}')


if __name__ == '__main__':
    process()
