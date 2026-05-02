"""Data-level analysis plots for the NEW (fixed MJCF) dataset only.

Reads `skin_pick_fixed_v1/.../trajectories_batch_*.h5` and writes:
  - <RUN>/plots_data/dataset_overview.png
  - <RUN>/plots_data/per_patch_stats.png
  - <RUN>/plots_data/per_patch_depth_hist.png
  - <RUN>/plots_data/depth_vs_phase.png
  - <RUN>/plots_data/correlation_state_depth.png  (qpos/tcp vs fracNear)

Override the run dir / data glob with the env vars PLA_CVAE_RUN and
PLA_CVAE_DATA_GLOB.
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get('PLA_CVAE_RUN', str(REPO_ROOT / 'runs/cvae_v3')))
OUT = RUN / 'plots_data'
OUT.mkdir(parents=True, exist_ok=True)
ZFAR = 4.0
GLOB = os.environ.get(
    'PLA_CVAE_DATA_GLOB',
    str(REPO_ROOT / 'data/skin_pick_fixed_v1/**/trajectories_batch_*.h5'),
)

MJCF_ORDER = (
    [f'link6_s{i}' for i in range(8)] +
    [f'link5_s{i}' for i in range(6)] +
    [f'link3_s{i}' for i in range(8)] +
    [f'link2_s{i}' for i in range(7)]
)
PHASE_NAMES = {0: 'init', 1: 'approach', 2: 'pre-grasp', 3: 'grasp', 4: 'lift', 5: 'place'}


def load_all():
    rows = []
    for p in sorted(glob.glob(GLOB)):
        with h5py.File(p, 'r') as f:
            for k in sorted([k for k in f.keys() if k.startswith('traj_')]):
                try:
                    d = f[f'{k}/obs/extra/proximity'][:]
                    ph = f[f'{k}/obs/extra/policy_phase'][:]
                    panda = f[f'{k}/env_states/articulations/panda'][:]
                    tcp = f[f'{k}/obs/extra/tcp_pose'][:]
                except KeyError:
                    continue
                rows.append(dict(path=p, traj=k, T=d.shape[0], phase=ph, depth=d,
                                 panda=panda, tcp=tcp))
    return rows


def plot_overview(rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    lengths = [r['T'] for r in rows]
    ax = axes[0]
    ax.bar(np.arange(len(lengths)), lengths, color='seagreen')
    ax.set_xlabel('trajectory index'); ax.set_ylabel('timesteps')
    ax.set_title(f'Trajectory lengths — {len(rows)} trajs, {sum(lengths)} timesteps')
    ax.grid(alpha=0.3)

    phases_all = np.concatenate([r['phase'] for r in rows])
    ax = axes[1]
    phases = sorted(np.unique(phases_all).tolist())
    counts = [(phases_all == p).sum() for p in phases]
    ax.bar([PHASE_NAMES.get(p, f'p{p}') for p in phases], counts, color='seagreen')
    ax.set_ylabel('timesteps'); ax.set_title(f'Phase distribution  (N={len(phases_all)})')
    ax.tick_params(axis='x', rotation=30); ax.grid(alpha=0.3)

    d_all = np.concatenate([r['depth'].reshape(-1) for r in rows])
    ax = axes[2]
    bins = np.linspace(0, ZFAR, 80)
    ax.hist(d_all, bins=bins, color='seagreen', density=True)
    ax.axvline(0.30, color='red', ls='--', lw=0.8, alpha=0.7, label='0.30 m near threshold')
    ax.axvline(ZFAR, color='k', ls='--', lw=0.8, alpha=0.7, label='zfar = no hit')
    ax.set_xlabel('depth (m)'); ax.set_ylabel('density')
    ax.set_title(f'Global depth distribution  (N={len(d_all)/1e6:.2f}M samples)')
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(OUT / 'dataset_overview.png', dpi=140)
    plt.close(fig)
    print('wrote dataset_overview.png')


def plot_per_patch_stats(rows):
    d = np.concatenate([r['depth'] for r in rows], axis=0)
    T, N, H, W = d.shape
    median = np.median(d, axis=(0, 2, 3))
    mean = d.mean(axis=(0, 2, 3))
    p05 = np.percentile(d, 5, axis=(0, 2, 3))
    p95 = np.percentile(d, 95, axis=(0, 2, 3))
    frac_near = (d < 0.30).mean(axis=(0, 2, 3))
    frac_far = (d >= ZFAR - 1e-3).mean(axis=(0, 2, 3))
    std_t = d.std(axis=0).mean(axis=(1, 2))

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    ax = axes[0]
    ax.bar(np.arange(N), median, color='seagreen', label='median')
    ax.errorbar(np.arange(N), mean, yerr=[mean - p05, p95 - mean], fmt='none',
                ecolor='black', alpha=0.6, capsize=3, label='5–95 % spread')
    ax.scatter(np.arange(N), mean, s=18, color='crimson', zorder=5, label='mean')
    ax.set_ylabel('depth (m)'); ax.set_title('Per-patch depth (median, mean, 5–95 %)')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(N), frac_near, color='crimson', label='< 0.30 m')
    ax.set_ylabel('fraction'); ax.set_title('Fraction of taxels reading < 0.30 m')
    ax.set_ylim(0, max(frac_near.max() * 1.1, 0.05))
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[2]
    ax.bar(np.arange(N), frac_far, color='steelblue', label='= zfar (no hit)')
    ax.set_ylabel('fraction'); ax.set_title('Fraction of taxels at zfar (sensor sees nothing)')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[3]
    ax.bar(np.arange(N), std_t, color='purple')
    ax.set_ylabel('σ (m)'); ax.set_title('Mean per-taxel temporal σ  — how much each patch\'s reading varies over time')
    ax.grid(alpha=0.3)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(MJCF_ORDER, rotation=90, fontsize=7)
    ax.set_xlabel('patch (MJCF order)')

    fig.suptitle(f'Per-patch summary  —  new dataset  (T={T} timesteps)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'per_patch_stats.png', dpi=140)
    plt.close(fig)
    print('wrote per_patch_stats.png')


def plot_per_patch_hist(rows):
    d = np.concatenate([r['depth'] for r in rows], axis=0)
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    bins = np.linspace(0, ZFAR, 50)
    for i in range(32):
        ax = axes[i // 8, i % 8]
        ax.set_xticks([]); ax.set_yticks([])
        if i < 29:
            ax.hist(d[:, i].reshape(-1), bins=bins, color='seagreen', density=True)
            ax.set_title(MJCF_ORDER[i], fontsize=7)
            ax.set_yscale('log')
            ax.axvline(0.30, color='red', ls='--', lw=0.5, alpha=0.5)
            ax.axvline(ZFAR, color='k', ls='--', lw=0.5, alpha=0.5)
    fig.suptitle('Per-patch depth distribution  (log density)  —  new dataset', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'per_patch_depth_hist.png', dpi=140)
    plt.close(fig)
    print('wrote per_patch_depth_hist.png')


def plot_depth_vs_phase(rows):
    d = np.concatenate([r['depth'] for r in rows], axis=0)
    ph = np.concatenate([r['phase'] for r in rows], axis=0)
    phases = sorted(np.unique(ph).tolist())
    fig, axes = plt.subplots(5, 6, figsize=(15, 10), sharey=True)
    for i in range(30):
        ax = axes[i // 6, i % 6]
        ax.set_xticks([]); ax.set_yticks([])
        if i < 29:
            data_by_phase = [d[ph == p, i].reshape(-1) for p in phases]
            ax.boxplot(data_by_phase, positions=np.arange(len(phases)),
                       widths=0.6, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor='seagreen', alpha=0.6))
            ax.set_title(MJCF_ORDER[i], fontsize=7)
            ax.set_ylim(0, ZFAR)
            ax.axhline(0.3, color='red', ls='--', lw=0.5, alpha=0.5)
            ax.set_xticks(np.arange(len(phases)))
            ax.set_xticklabels([str(p) for p in phases], fontsize=6)
        else:
            ax.axis('off')
    fig.suptitle(
        f'Depth distribution per patch, split by policy phase '
        f'({", ".join(f"{p}={PHASE_NAMES[p]}" for p in phases)})',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / 'depth_vs_phase.png', dpi=140)
    plt.close(fig)
    print('wrote depth_vs_phase.png')


def plot_state_depth_correlation(rows):
    """For each patch, compute linear correlation of its min-depth timestep-wise
    against each of the 7 arm joint positions. High |r| means the patch's
    reading is strongly driven by robot kinematics — the conditioning in the
    CVAE should capture this."""
    d = np.concatenate([r['depth'] for r in rows], axis=0)
    panda = np.concatenate([r['panda'] for r in rows], axis=0)
    T, N = d.shape[:2]
    min_d = d.reshape(T, N, -1).min(axis=-1)                    # (T, 29)
    corr = np.zeros((N, 7))
    for j in range(7):
        q = panda[:, j]
        for i in range(N):
            # Pearson r
            x = q - q.mean(); y = min_d[:, i] - min_d[:, i].mean()
            denom = np.sqrt((x**2).sum() * (y**2).sum()) + 1e-12
            corr[i, j] = (x * y).sum() / denom

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([f'q{i+1}' for i in range(7)])
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(MJCF_ORDER, fontsize=7)
    ax.set_xlabel('arm joint position')
    ax.set_title('Pearson correlation: per-patch min-depth  vs.  arm joint position\n'
                 '(blue = negative, red = positive; strong |r| = this joint drives the reading)')
    for i in range(N):
        for j in range(7):
            c = 'white' if abs(corr[i, j]) > 0.6 else 'black'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=6, color=c)
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT / 'correlation_state_depth.png', dpi=140)
    plt.close(fig)
    print('wrote correlation_state_depth.png')


def main():
    rows = load_all()
    print(f'{len(rows)} trajectories, total {sum(r["T"] for r in rows)} timesteps')
    plot_overview(rows)
    plot_per_patch_stats(rows)
    plot_per_patch_hist(rows)
    plot_depth_vs_phase(rows)
    plot_state_depth_correlation(rows)
    print(f'outputs in {OUT}')


if __name__ == '__main__':
    main()
