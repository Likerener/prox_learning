"""ProxAugmentedEpisodicDataset — ACT's EpisodicDataset plus a per-sensor proximity window.

Same sampling semantics as `submodules/act/utils.py:EpisodicDataset` (random
`start_ts` per `__getitem__`, action chunk of `num_queries` steps, image
loading + normalisation, qpos / action normalisation). Adds a 5th return
value `proximity_window` of shape `(N_sensors, W*4, 8, 8)`, ready to be fed
into the frozen prox-encoder.

The proximity window is sourced from the **original** datagen h5 via
`prox_mapping.json` built by `pact.act_prox.build_mapping` — the ACT episode
file itself does not carry proximity. Source h5 is opened lazily once per
worker and cached for the rest of the worker's lifetime.

Normalisation matches the encoder's training-time recipe exactly:

    raw_window: (W, 4, 8, 8) float32      # raw depth (metres)
    window     = (raw - prox_mean[None]) / prox_std[None]    # (4, 8, 8) buffers
    window     = window.reshape(W*4, 8, 8)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Re-use ACT's EpisodicDataset behaviour wholesale where possible.
_ACT_DIR = Path(__file__).resolve().parents[2] / "submodules" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
from utils import EpisodicDataset, get_norm_stats  # noqa: E402


class ProxAugmentedEpisodicDataset(Dataset):
    """EpisodicDataset clone that also yields a per-sensor proximity window.

    Returns: (image, qpos, action, is_pad, proximity_window) with
      image           : (n_cam, 3, H, W) float32 in [0,1]
      qpos            : (qpos_dim,) float32 normalised
      action          : (num_queries, action_dim) float32 normalised
      is_pad          : (num_queries,) bool
      proximity_window: (n_sensors, W*4, 8, 8) float32 z-scored
    """

    def __init__(
        self,
        episode_ids: np.ndarray,
        dataset_dir: str | os.PathLike,
        camera_names: List[str],
        norm_stats: Dict[str, np.ndarray],
        num_queries: int,
        prox_mapping: Dict,
        prox_mean: np.ndarray,       # (4, 8, 8)
        prox_std: np.ndarray,        # (4, 8, 8)
        window: int,                 # encoder window in control steps
    ):
        super().__init__()
        self.episode_ids = np.asarray(episode_ids)
        self.dataset_dir = str(dataset_dir)
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.num_queries = num_queries

        # Proximity-side state.
        self.mapping = prox_mapping
        self.sensor_names: List[str] = list(prox_mapping["sensor_names"])
        self.n_sensors: int = int(prox_mapping["n_sensors"])
        self.prox_mean = prox_mean.astype(np.float32)
        self.prox_std = prox_std.astype(np.float32)
        self.window = int(window)

        # Lazy-opened source h5 handles, keyed by absolute path. Each worker
        # populates its own dict so file handles stay private per process.
        self._src_handles: Dict[str, h5py.File] = {}

        # ACT's dataset infers `is_sim` from the first sample; we mirror that.
        self.is_sim: Optional[bool] = None
        # Touch one sample so downstream code can read `self.is_sim`.
        self.__getitem__(0)

    def __len__(self) -> int:
        return int(self.episode_ids.shape[0])

    # ---- helpers -----------------------------------------------------------

    def _open_source(self, path: str) -> h5py.File:
        h = self._src_handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._src_handles[path] = h
        return h

    def _load_proximity_window(self, episode_idx: int, start_ts: int) -> np.ndarray:
        """Return a (n_sensors, W*4, 8, 8) z-scored window ending at start_ts.

        Left-padded by repeating the first available substep when start_ts < W-1.
        """
        entry = self.mapping["episodes"][str(int(episode_idx))]
        src = self._open_source(entry["source_h5"])
        traj = src[entry["traj_key"]]
        W = self.window
        lo = max(0, start_ts - W + 1)
        hi = start_ts + 1                                        # exclusive
        n_real = hi - lo                                         # in [1, W]
        n_pad = W - n_real

        # Stack across sensors.
        sensor_windows = np.empty((self.n_sensors, W, 4, 8, 8), dtype=np.float32)
        for s_idx, sn in enumerate(self.sensor_names):
            full = traj[f"obs/proximity/{sn}"][lo:hi]            # (n_real, 4, 8, 8)
            if n_pad > 0:
                pad = np.repeat(full[:1], n_pad, axis=0)         # repeat earliest
                full = np.concatenate([pad, full], axis=0)
            sensor_windows[s_idx] = full

        # Channel-wise z-score: prox_mean/std broadcast across W.
        sensor_windows = (sensor_windows - self.prox_mean[None, None]) / self.prox_std[None, None]
        # Fold (W, 4) into one time dim for the encoder.
        sensor_windows = sensor_windows.reshape(self.n_sensors, W * 4, 8, 8)
        return sensor_windows

    # ---- main API ----------------------------------------------------------

    def __getitem__(self, index: int):
        episode_idx = int(self.episode_ids[index])
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_idx}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = bool(root.attrs.get("sim", False))
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            start_ts = int(np.random.choice(episode_len))

            qpos = root["/observations/qpos"][start_ts]
            image_dict: Dict[str, np.ndarray] = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim

        # Action padding to `num_queries`.
        padded_action = np.zeros((self.num_queries, original_action_shape[1]), dtype=np.float32)
        n_real = min(action_len, self.num_queries)
        padded_action[:n_real] = action[:n_real]
        is_pad = np.zeros(self.num_queries, dtype=bool)
        is_pad[n_real:] = True

        # Camera stack.
        cam_stack = np.stack([image_dict[c] for c in self.camera_names], axis=0)  # (n_cam, H, W, 3)
        image = torch.from_numpy(cam_stack).float().permute(0, 3, 1, 2) / 255.0
        qpos_t = torch.from_numpy(qpos).float()
        action_t = torch.from_numpy(padded_action).float()
        is_pad_t = torch.from_numpy(is_pad)

        # Normalise.
        action_t = (action_t - torch.as_tensor(self.norm_stats["action_mean"])) / torch.as_tensor(self.norm_stats["action_std"])
        qpos_t = (qpos_t - torch.as_tensor(self.norm_stats["qpos_mean"])) / torch.as_tensor(self.norm_stats["qpos_std"])

        # Proximity window (NOT through __init__'s recursive call, that one
        # also computes a proximity_window we ignore; it's only ~5 ms).
        prox = self._load_proximity_window(episode_idx, start_ts)        # (n_sensors, W*4, 8, 8) np
        prox_t = torch.from_numpy(prox)

        return image, qpos_t, action_t, is_pad_t, prox_t

    def __del__(self):
        for h in self._src_handles.values():
            try:
                h.close()
            except Exception:
                pass


# ---- factory ---------------------------------------------------------------


def load_norm_stats(dataset_dir: str | os.PathLike, num_episodes: int) -> Dict[str, np.ndarray]:
    """Thin re-export so callers don't have to import ACT's utils.py path hack."""
    return get_norm_stats(dataset_dir, num_episodes)


def make_prox_dataloaders(
    dataset_dir: str | os.PathLike,
    num_episodes: int,
    camera_names: List[str],
    batch_size_train: int,
    batch_size_val: int,
    num_queries: int,
    prox_mapping_json: str | os.PathLike,
    prox_ckpt_path: str | os.PathLike,
    val_ratio: float = 0.2,
    num_workers: int = 1,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray], bool, Dict]:
    """Construct train/val ProxAugmentedEpisodicDataset + DataLoaders.

    Returns (train_loader, val_loader, norm_stats, is_sim, prox_mapping).
    """
    # Load prox encoder ckpt for per-substep prox_mean/std and the window param.
    ckpt = torch.load(prox_ckpt_path, map_location="cpu", weights_only=False)
    prox_mean = np.asarray(ckpt["prox_mean"], dtype=np.float32)
    prox_std = np.asarray(ckpt["prox_std"], dtype=np.float32)
    window = int(ckpt["window"])

    # Load the mapping.
    with open(prox_mapping_json, "r") as f:
        mapping = json.load(f)
    if len(mapping["episodes"]) < num_episodes:
        raise RuntimeError(
            f"prox_mapping has {len(mapping['episodes'])} episodes, "
            f"but load_data was asked for {num_episodes}"
        )

    # Deterministic train/val split.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    n_train = int((1.0 - val_ratio) * num_episodes)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    norm_stats = get_norm_stats(str(dataset_dir), num_episodes)

    train_ds = ProxAugmentedEpisodicDataset(
        episode_ids=train_idx,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        num_queries=num_queries,
        prox_mapping=mapping,
        prox_mean=prox_mean,
        prox_std=prox_std,
        window=window,
    )
    val_ds = ProxAugmentedEpisodicDataset(
        episode_ids=val_idx,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        num_queries=num_queries,
        prox_mapping=mapping,
        prox_mean=prox_mean,
        prox_std=prox_std,
        window=window,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, norm_stats, bool(train_ds.is_sim), mapping
