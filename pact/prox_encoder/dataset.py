"""Torch Dataset over the cached proximity samples.

Returns:
  prox:  (W*4, 8, 8) fp32 normalized
  label: (3,) fp32, in normalized space
  meta:  dict with raw label, sensor_id, traj_id, t (kept for eval)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ProxWindowDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        indices: Optional[np.ndarray] = None,
        normalize_label: bool = True,
    ):
        self.path = Path(npz_path)
        d = np.load(self.path)
        # Eager-load — cache is small enough.
        self.prox = d["prox"]                    # (N, W, 4, 8, 8) fp16
        self.label = d["label"].astype(np.float32)  # (N, 3)
        self.sensor_id = d["sensor_id"]
        self.traj_id = d["traj_id"]
        self.t = d["t"]
        self.sensor_names = d["sensor_names"]
        self.prox_mean = d["prox_mean"].astype(np.float32)        # (4, 8, 8)
        self.prox_std = d["prox_std"].astype(np.float32)          # (4, 8, 8)
        self.label_mean = d["label_mean"].astype(np.float32)      # (3,)
        self.label_std = d["label_std"].astype(np.float32)        # (3,)
        self.window: int = int(d["window"])

        if indices is None:
            self.indices = np.arange(self.prox.shape[0])
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        self.normalize_label = normalize_label

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_names)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        i = int(self.indices[idx])
        prox = self.prox[i].astype(np.float32)         # (W, 4, 8, 8)
        # Normalize per (sub-step, row, col) channel using cached stats.
        prox = (prox - self.prox_mean[None]) / self.prox_std[None]
        # Flatten W and 4 sub-steps into a single time dimension.
        W, C, H, Wd = prox.shape
        prox = prox.reshape(W * C, H, Wd)              # (T, 8, 8) with T = W*4

        label = self.label[i].copy()
        if self.normalize_label:
            label_n = (label - self.label_mean) / self.label_std
        else:
            label_n = label

        meta = {
            "label_raw": torch.from_numpy(label.astype(np.float32)),
            "sensor_id": torch.tensor(int(self.sensor_id[i]), dtype=torch.long),
            "traj_id": torch.tensor(int(self.traj_id[i]), dtype=torch.long),
            "t": torch.tensor(int(self.t[i]), dtype=torch.long),
        }
        return (
            torch.from_numpy(prox.astype(np.float32)),
            torch.from_numpy(label_n.astype(np.float32)),
            meta,
        )


def split_by_trajectory(
    npz_path: str | Path,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val split that holds out whole trajectories."""
    d = np.load(npz_path)
    traj_id = d["traj_id"]
    unique_trajs = np.unique(traj_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_trajs)
    n_val = max(1, int(len(perm) * val_frac))
    val_trajs = set(int(x) for x in perm[:n_val])
    val_mask = np.array([int(tid) in val_trajs for tid in traj_id])
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx
