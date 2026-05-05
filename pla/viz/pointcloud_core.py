"""Pure-NumPy core for ToF-depth → world-point reconstruction.

Factored out of :mod:`pla.viz.pointcloud` so it can be unit-tested
without requiring HDF5 datasets, pybullet, or any rendering. All
functions here are deterministic and side-effect free.

Convention
----------
Each VL53L5CX patch is a pinhole camera with:
  - field of view: ``fovy`` degrees, square aspect (8 x 8 taxels)
  - body frame: looks down +Z (depth axis), +X right, +Y down

(The +Y-down sign matches the original implementation's ``-v`` flip in
the unprojection — it is the standard image-plane convention where row
0 is the top of the image.)

To go from a depth image ``D[H, W]`` to a world point cloud:

    body_pts = unproject_taxels_to_body(D, fovy_deg, ...)
    world_pts = transform_points(body_pts, body_pos, body_R)

``transform_points`` applies the standard ``world = pos + R @ pt``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PinholeIntrinsics:
    """Square pinhole intrinsics for an 8x8 ToF taxel grid."""
    fovy_deg: float = 45.0
    res: int = 8
    znear_m: float = 0.02
    zfar_m: float = 4.0

    @property
    def half_tan(self) -> float:
        return float(np.tan(np.deg2rad(self.fovy_deg) * 0.5))


def taxel_directions(intr: PinholeIntrinsics, frame: str = "depth_axis") -> np.ndarray:
    """Return ray direction vectors per taxel, **before** scaling by depth.

    Two frames are supported:

    - ``"camera"`` — standard OpenGL pinhole. Output point at depth ``d``
      is ``(u·half·d, -v·half·d, -d)`` in mujoco camera frame
      (+x right, +y up, looking down -z).
    - ``"depth_axis"`` — non-standard "body-like" frame where depth is
      along **+z** (so ``z = +d``). This matches the legacy convention
      used by :mod:`pla.viz.pointcloud`. To go to world you must
      compose with the body-frame rotation (R_body), **not** R_cam.

    Output shape: ``(res, res, 3)``.
    """
    if frame not in ("camera", "depth_axis"):
        raise ValueError(f"frame must be 'camera' or 'depth_axis', got {frame!r}")
    n = intr.res
    half = intr.half_tan
    u = ((np.arange(n) + 0.5) / n * 2 - 1).astype(np.float64)
    v = ((np.arange(n) + 0.5) / n * 2 - 1).astype(np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    if frame == "camera":
        # OpenGL: y_cam_norm = -v_image, z_cam = -d → ray = (u*half, -v*half, -1)
        dirs = np.stack([uu * half, -vv * half, -np.ones_like(uu)], axis=-1)
    else:  # depth_axis: legacy body-frame convention (z = +d)
        dirs = np.stack([uu * half, -vv * half, np.ones_like(uu)], axis=-1)
    return dirs


def unproject_taxels(
    depth: np.ndarray,
    intr: PinholeIntrinsics | None = None,
    near_eps: float = 1e-3,
    far_eps: float = 1e-3,
    frame: str = "camera",
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject depth image into points in the chosen frame.

    Default ``frame="camera"`` returns standard OpenGL camera-frame
    points (use with ``R_cam`` to transform to world). ``frame="depth_axis"``
    returns the legacy convention (use with ``R_body``).
    """
    intr = intr or PinholeIntrinsics()
    if depth.shape != (intr.res, intr.res):
        raise ValueError(f"expected ({intr.res},{intr.res}) depth, got {depth.shape}")
    dirs = taxel_directions(intr, frame=frame)
    mask = (depth > intr.znear_m + near_eps) & (depth < intr.zfar_m - far_eps)
    d = depth[mask].astype(np.float64)
    if frame == "camera":
        # ray.z = -1, so multiplying by depth d gives z = -d (in front of cam)
        pts = dirs[mask] * d[:, None]
    else:
        pts = dirs[mask] * d[:, None]
    return pts, mask


# Back-compat alias used by pla.viz.pointcloud (legacy depth_axis convention).
def unproject_taxels_to_body(
    depth: np.ndarray,
    intr: PinholeIntrinsics | None = None,
    near_eps: float = 1e-3,
    far_eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    return unproject_taxels(depth, intr=intr, near_eps=near_eps, far_eps=far_eps,
                              frame="depth_axis")


def transform_points(
    pts_local: np.ndarray, pos: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Apply ``world = pos + R @ pts_local`` (right-multiply with ``R.T``)."""
    return pts_local @ R.T + pos[None, :]


def reconstruct_world_pts(
    depth: np.ndarray,
    sensor_pos: np.ndarray,
    sensor_R: np.ndarray,
    intr: PinholeIntrinsics | None = None,
    frame: str = "camera",
) -> tuple[np.ndarray, np.ndarray]:
    """Compose unproject + transform.

    Use ``frame="camera"`` with ``sensor_pos = data.cam_xpos[cid]`` and
    ``sensor_R = data.cam_xmat[cid]`` for ground-truth-correct output.

    Use ``frame="depth_axis"`` with body pose for legacy compatibility
    (this matches :mod:`pla.viz.pointcloud`).
    """
    pts_local, mask = unproject_taxels(depth, intr=intr, frame=frame)
    pts_world = transform_points(pts_local, sensor_pos, sensor_R)
    return pts_world, mask


def fit_plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane fit via SVD.

    Returns ``(centroid, normal, rms_residual)``. ``normal`` has unit
    length; sign is arbitrary.
    """
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points for a plane fit")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Smallest singular vector = plane normal
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    residuals = np.abs(centered @ normal)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return centroid, normal, rms
