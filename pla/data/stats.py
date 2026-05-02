"""Compute per-dataset statistics that back up the README.

Dumps a markdown-formatted summary to stdout. Covers:
  - trajectory count, total timesteps, per-traj lengths
  - HDF5 file sizes + key shapes
  - qpos / qvel ranges per joint
  - TCP pose stats
  - proximity: % of taxels in each bin, per-patch median
  - RGB video file sizes
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
GLOB = os.environ.get(
    'PLA_PROX_DATA_GLOB',
    str(REPO_ROOT / 'data/skin_pick_fixed_v1/**/trajectories_batch_*.h5'),
)
DATASET_ROOT = os.environ.get(
    'PLA_PROX_DATA_ROOT',
    str(REPO_ROOT / 'data/skin_pick_fixed_v1'),
)


def main():
    paths = sorted(glob.glob(GLOB))
    tot_trajs = 0; tot_T = 0
    all_panda, all_prox, all_tcp, all_phase = [], [], [], []
    per_traj = []

    for p in paths:
        with h5py.File(p, 'r') as f:
            for k in sorted([k for k in f.keys() if k.startswith('traj_')]):
                try:
                    panda = f[f'{k}/env_states/articulations/panda'][:]
                    prox = f[f'{k}/obs/extra/proximity'][:]
                    tcp = f[f'{k}/obs/extra/tcp_pose'][:]
                    ph = f[f'{k}/obs/extra/policy_phase'][:]
                except KeyError:
                    continue
                T = prox.shape[0]
                per_traj.append((p, k, T))
                tot_trajs += 1
                tot_T += T
                all_panda.append(panda)
                all_prox.append(prox)
                all_tcp.append(tcp)
                all_phase.append(ph)

    panda = np.concatenate(all_panda, 0)
    prox = np.concatenate(all_prox, 0)
    tcp = np.concatenate(all_tcp, 0)
    phase = np.concatenate(all_phase, 0)

    total_h5_bytes = sum(Path(p).stat().st_size for p in paths)
    mp4_files = []
    for p in paths:
        d = Path(p).parent
        mp4_files.extend(d.glob('episode_*.mp4'))
    total_mp4_bytes = sum(f.stat().st_size for f in mp4_files)

    print(f'## Dataset statistics — `skin_pick_fixed_v1`')
    print(f'')
    print(f'- Trajectories: **{tot_trajs}** across {len(paths)} HDF5 files')
    print(f'- Total timesteps: **{tot_T}**')
    print(f'- Trajectory length range: {min(t for _,_,t in per_traj)}–{max(t for _,_,t in per_traj)}')
    print(f'- Trajectory mean length: {np.mean([t for _,_,t in per_traj]):.1f}')
    print(f'')
    print(f'### Storage')
    print(f'- HDF5 files: {total_h5_bytes/1e6:.1f} MB across {len(paths)} files')
    print(f'- RGB MP4 files: {total_mp4_bytes/1e6:.1f} MB across {len(mp4_files)} files ({len(mp4_files)} × 2 per episode: exo + wrist)')
    print(f'- Total on disk: {(total_h5_bytes + total_mp4_bytes)/1e6:.1f} MB')
    print(f'')
    print(f'### Per-trajectory table')
    print(f'| idx | path | timesteps |')
    print(f'|---|---|---|')
    for i, (pth, k, T) in enumerate(per_traj):
        try:
            rel = Path(pth).relative_to(Path(DATASET_ROOT))
        except ValueError:
            rel = Path(pth)
        print(f'| {i} | `{rel}::{k}` | {T} |')
    print(f'')

    print(f'### Phase coverage')
    phases, counts = np.unique(phase, return_counts=True)
    names = {0:'init', 1:'approach', 2:'pre-grasp', 3:'grasp', 4:'lift', 5:'place'}
    print(f'| phase | name | timesteps | % |')
    print(f'|---|---|---|---|')
    for ph, c in zip(phases, counts):
        print(f'| {int(ph)} | {names.get(int(ph), "?")} | {c} | {100*c/len(phase):.1f}% |')
    print(f'')

    print(f'### Arm joint positions (panda[:, 0:7])')
    print(f'| joint | min | max | mean | std |')
    print(f'|---|---|---|---|---|')
    for j in range(7):
        q = panda[:, j]
        print(f'| fr3_joint{j+1} | {q.min():+.3f} | {q.max():+.3f} | {q.mean():+.3f} | {q.std():.3f} |')
    print(f'')
    print(f'### Gripper (driver + 5 mimic joints saved as the same value)')
    q = panda[:, 7]
    print(f'| finger | min | max | mean | std |')
    print(f'|---|---|---|---|---|')
    print(f'| open-close (rad) | {q.min():.4f} | {q.max():.4f} | {q.mean():.4f} | {q.std():.4f} |')
    print(f'')

    print(f'### TCP pose (robot base frame) — obs/extra/tcp_pose')
    print(f'| component | min | max | mean | std |')
    print(f'|---|---|---|---|---|')
    labels = ['x (m)', 'y (m)', 'z (m)', 'qx', 'qy', 'qz', 'qw']
    for c, lab in enumerate(labels):
        v = tcp[:, c]
        print(f'| {lab} | {v.min():+.3f} | {v.max():+.3f} | {v.mean():+.3f} | {v.std():.3f} |')
    print(f'')

    print(f'### Proximity depth (m) — 29 patches × 8×8 = 1856 per step × {tot_T} steps = {tot_T*1856:,} values')
    print(f'- Overall min/max: {prox.min():.3f} / {prox.max():.3f}')
    print(f'- Overall mean: {prox.mean():.3f}')
    print(f'- % of values at zfar (no hit): {100*(prox >= 4.0-1e-3).mean():.2f}%')
    print(f'- % of values < 0.30 m (near contact): {100*(prox < 0.30).mean():.2f}%')
    print(f'- % of values < 0.10 m (very close): {100*(prox < 0.10).mean():.2f}%')
    print(f'')

    print(f'### Per-patch median depth (m)')
    names = ([f'link6_s{i}' for i in range(8)] + [f'link5_s{i}' for i in range(6)]
             + [f'link3_s{i}' for i in range(8)] + [f'link2_s{i}' for i in range(7)])
    med = np.median(prox, axis=(0, 2, 3))
    fracNear = (prox < 0.30).mean(axis=(0, 2, 3))
    std_t = prox.std(axis=0).mean(axis=(1, 2))
    print(f'| patch | median | fracNear | σ_t |')
    print(f'|---|---|---|---|')
    for i in range(29):
        print(f'| {names[i]} | {med[i]:.3f} | {fracNear[i]:.3f} | {std_t[i]:.3f} |')


if __name__ == '__main__':
    main()
