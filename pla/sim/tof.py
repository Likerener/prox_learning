"""ToF observation extension.

Renders an 8x8 depth buffer per sensor camera at every simulation step,
clips to VL53L5CX physical range (20 mm – 4000 mm), and adds 5 mm Gaussian
noise. The result is stacked into ``obs['tof']`` with shape
``[N_sensors, 8, 8]``.

This mirrors the snippet in PROJECT.md §3.2. Call it once per step after the
underlying sim env has produced its base ``obs`` dict.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import mujoco
except ImportError:  # noqa: F401
    mujoco = None  # type: ignore[assignment]


# Defaults; override at call site if the MJCF uses different camera names.
DEFAULT_SENSOR_CAM_NAMES: tuple[str, ...] = ()


def extend_obs_with_tof(
    obs: dict,
    env,
    *,
    sensor_cam_names: Iterable[str] = DEFAULT_SENSOR_CAM_NAMES,
    renderer=None,
    noise_sigma_mm: float = 5.0,
    clip_min_mm: float = 20.0,
    clip_max_mm: float = 4000.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Add ``obs['tof']`` of shape [N_sensors, 8, 8] in millimetres.

    Args:
        obs: existing observation dict from the sim env (mutated in place).
        env: the sim env, must expose ``model`` (mjModel) and ``data`` (mjData).
        sensor_cam_names: ordered camera names matching the MJCF sensor cameras.
        renderer: a ``mujoco.Renderer`` configured for 8x8 output. If None, one
            is constructed lazily — pass an explicit one to avoid per-step alloc.
        noise_sigma_mm: Gaussian noise stddev in millimetres.
        clip_min_mm, clip_max_mm: VL53L5CX measurable range.
        rng: numpy Generator for deterministic noise. If None, uses default.

    Returns:
        The same ``obs`` dict, with a new key ``tof``.
    """
    if mujoco is None:
        raise RuntimeError("mujoco import failed — install mujoco>=3.0 to use this.")
    if rng is None:
        rng = np.random.default_rng()
    if renderer is None:
        renderer = mujoco.Renderer(env.model, height=8, width=8)

    readings = []
    for cam_name in sensor_cam_names:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        renderer.update_scene(env.data, camera=cam_id)
        renderer.enable_depth_rendering()
        depth_m = renderer.render()
        depth_mm = depth_m * 1000.0
        depth_mm = np.clip(depth_mm, clip_min_mm, clip_max_mm)
        depth_mm = depth_mm + rng.standard_normal((8, 8)) * noise_sigma_mm
        readings.append(depth_mm.astype(np.float32))

    obs["tof"] = np.stack(readings, axis=0)
    return obs
