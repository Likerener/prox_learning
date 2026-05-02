"""HDF5 schema validation for PLA trajectory datasets.

A valid trajectory file has the layout described in
docs/PROJECT.md §3.3 / pla/data/collect.py. ``validate(path)`` returns
``(ok, errors)``. Use it as a precondition in training loaders so a corrupt
shard fails loud and early.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REQUIRED_OBS_KEYS: dict[str, tuple[type, tuple[int, ...]]] = {
    # key: (dtype, expected suffix shape after the leading T axis)
    "tof": (np.floating, (8, 8)),  # plus N_sensors as middle axis
    "rgb": (np.unsignedinteger, (3, 224, 224)),
    "qpos": (np.floating, (7,)),
}


def validate(path: Path) -> tuple[bool, list[str]]:
    errs: list[str] = []
    with h5py.File(path, "r") as f:
        episodes = [k for k in f.keys() if k.startswith("episode_") or k.startswith("traj_")]
        if not episodes:
            return False, ["no episode_*/traj_* groups in file"]
        for ep in episodes:
            grp = f[ep]
            if "observations" not in grp:
                errs.append(f"{ep}: missing observations/")
                continue
            obs = grp["observations"]
            for key, (dtype, suffix) in REQUIRED_OBS_KEYS.items():
                if key not in obs:
                    errs.append(f"{ep}: missing observations/{key}")
                    continue
                arr = obs[key]
                if not np.issubdtype(arr.dtype, dtype):
                    errs.append(f"{ep}: {key} dtype {arr.dtype} not {dtype}")
                if arr.shape[-len(suffix):] != suffix:
                    errs.append(f"{ep}: {key} shape {arr.shape} suffix != {suffix}")
            if "actions" not in grp:
                errs.append(f"{ep}: missing actions/")
    return len(errs) == 0, errs


def proximity_informative_fraction(
    paths: Iterable[Path], threshold_mm: float = 200.0
) -> float:
    """Fraction of timesteps where any sensor reads below ``threshold_mm``.

    PROJECT.md §3.3 requires >= 30%. Run this before kicking off training.
    """
    n_steps = 0
    n_close = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            for ep in f.keys():
                if "observations" not in f[ep] or "tof" not in f[ep]["observations"]:
                    continue
                tof = f[ep]["observations"]["tof"][:]  # [T, N, 8, 8]
                n_steps += tof.shape[0]
                n_close += int(np.any(tof.reshape(tof.shape[0], -1) < threshold_mm, axis=1).sum())
    return n_close / max(n_steps, 1)
