"""Dataset builder for the Franka skin proximity CVAE.

Loads (T, 29, 8, 8) proximity tensors + conditioning (qpos arm(7) + tcp(7))
from every trajectory in a directory of trajectory HDF5 files, concatenates
them, normalizes depth to [0, 1] by dividing by zfar, and splits train/val.

Conditioning vector layout (14-dim):
    [q1..q7 (arm), tcp_x, tcp_y, tcp_z, tcp_qx, tcp_qy, tcp_qz, tcp_qw]

Proximity input is flattened to 1856 dims (= 29 * 8 * 8) and rescaled so
zfar=4.0 → 1.0. Values clip to [0, 1].

`load_all` returns (X, Y, meta_dict) where:
    X: (N, 1856) float32  -- normalized proximity
    Y: (N,   14) float32  -- conditioning
    meta_dict contains per-sample phase label, timestep idx, traj idx so
    plots can color by policy phase.
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

ZFAR = 4.0
PROX_DIM = 29 * 8 * 8  # 1856


def _read_traj(f: h5py.File, key: str) -> tuple[np.ndarray, ...] | None:
    """Return (prox, cond, phase) for one traj_N group or None on failure."""
    try:
        prox = f[f'{key}/obs/extra/proximity'][:]          # (T, 29, 8, 8)
        panda = f[f'{key}/env_states/articulations/panda'][:]  # (T, 31)
        tcp = f[f'{key}/obs/extra/tcp_pose'][:]            # (T, 7)
        phase = f[f'{key}/obs/extra/policy_phase'][:]      # (T,)
    except KeyError:
        return None
    T = prox.shape[0]
    arm = panda[:, :7]                                      # (T, 7)
    cond = np.concatenate([arm, tcp], axis=1).astype(np.float32)  # (T, 14)
    x = np.clip(prox, 0, ZFAR).reshape(T, -1).astype(np.float32) / ZFAR
    return x, cond, phase.astype(np.int64)


def load_all(h5_glob: str) -> dict:
    paths = sorted(glob.glob(h5_glob))
    Xs, Ys, phases, traj_ids, t_ids = [], [], [], [], []
    traj_ctr = 0
    for p in paths:
        with h5py.File(p, 'r') as f:
            keys = sorted([k for k in f.keys() if k.startswith('traj_')])
            for k in keys:
                r = _read_traj(f, k)
                if r is None: continue
                x, c, ph = r
                T = len(x)
                Xs.append(x); Ys.append(c); phases.append(ph)
                traj_ids.append(np.full(T, traj_ctr, dtype=np.int64))
                t_ids.append(np.arange(T, dtype=np.int64))
                traj_ctr += 1
    if not Xs:
        raise RuntimeError(f'no trajs in {h5_glob}')
    X = np.concatenate(Xs, 0)
    Y = np.concatenate(Ys, 0)
    phase = np.concatenate(phases, 0)
    traj = np.concatenate(traj_ids, 0)
    t_idx = np.concatenate(t_ids, 0)
    return dict(X=X, Y=Y, phase=phase, traj=traj, t=t_idx, n_traj=traj_ctr,
                h5_paths=paths)


class ProxCVAEDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_loaders(X, Y, batch_size=64, val_frac=0.15, seed=0):
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr = ProxCVAEDataset(X[tr_idx], Y[tr_idx])
    va = ProxCVAEDataset(X[val_idx], Y[val_idx])
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(va, batch_size=batch_size, shuffle=False),
        tr_idx, val_idx,
    )


import os as _os
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_GLOB = _os.environ.get(
    'PLA_CVAE_DATA_GLOB',
    str(_REPO_ROOT / 'data/skin_pick_fixed_v1/**/trajectories_batch_*.h5'),
)


if __name__ == '__main__':
    data = load_all(DATASET_GLOB)
    print(f'loaded {data["n_traj"]} trajectories   total T={len(data["X"])}')
    print(f'  X shape: {data["X"].shape}   Y shape: {data["Y"].shape}')
    print(f'  phase counts: {np.bincount(data["phase"])}')
    print(f'  depth stats (normalized): min={data["X"].min():.3f} mean={data["X"].mean():.3f} max={data["X"].max():.3f}')
