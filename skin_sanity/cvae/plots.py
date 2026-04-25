"""Presentation plots for the CVAE.

Writes to `cvae/runs/<name>/plots/`:
  - loss_curves.png           train/val recon + KL over epochs
  - latent_scatter.png        2D projection of latents colored by phase
  - recon_samples.png         input vs. reconstruction tile grids for 4 steps
  - anomaly_over_time.png     anomaly score per-timestep per-trajectory + phase
  - prior_sample_diversity.png sample from prior, decode, show 29-tile grids
  - proximity_anomaly_map.png per-patch anomaly heatmap (patch x timestep)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def PCA(n_components: int):
    """Minimal PCA via numpy SVD (sklearn unavailable in this venv)."""
    class _PCA:
        def __init__(self, k): self.k = k
        def fit_transform(self, X):
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            return (U[:, :self.k] * S[:self.k])
    return _PCA(n_components)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from prox_learning.skin_sanity.cvae.model import CondVAE

RUN = Path('/home/jaydv/code/skin_sanity/cvae/runs/v1')
OUT = RUN / 'plots'
PHASE_NAMES = {0: 'init', 1: 'approach', 2: 'pre-grasp', 3: 'grasp', 4: 'lift', 5: 'place'}


def loss_curves(m: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(len(m['train_recon']))
    axes[0].plot(epochs, m['train_recon'], label='train', color='steelblue')
    axes[0].plot(epochs, m['val_recon'], label='val', color='crimson')
    axes[0].set_yscale('log'); axes[0].set_xlabel('epoch'); axes[0].set_ylabel('recon MSE (sum)')
    axes[0].set_title('Reconstruction loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, m['train_kl'], label='train KL', color='steelblue')
    axes[1].plot(epochs, m['val_kl'], label='val KL', color='crimson')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('KL')
    axes[1].set_title('KL divergence'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'loss_curves.png', dpi=140); plt.close(fig)


def latent_scatter(Z, phase, traj, val_mask, out: Path):
    # 2D PCA
    Z2 = PCA(n_components=2).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    phases = np.unique(phase)
    cmap = plt.get_cmap('tab10')
    for p in phases:
        m = phase == p
        ax.scatter(Z2[m, 0], Z2[m, 1], s=15, alpha=0.6,
                   color=cmap(int(p) % 10),
                   label=f'{int(p)}: {PHASE_NAMES.get(int(p), f"p{int(p)}")}')
    ax.set_title('CVAE latent μ, PCA-2D, colored by policy phase')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for ti in np.unique(traj):
        m = traj == ti
        marker = 'x' if val_mask[m].any() else 'o'
        ax.scatter(Z2[m, 0], Z2[m, 1], s=15, alpha=0.6, marker=marker,
                   label=f'traj {int(ti)} ({"val" if val_mask[m].any() else "train"})')
    ax.set_title('Latent μ colored by trajectory'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'latent_scatter.png', dpi=140); plt.close(fig)


def recon_samples(X, xhat, meta, out: Path, n_pick=4):
    # Pick a few interesting timesteps (varying depth intensity)
    intensity = X.mean(axis=1)
    idxs = np.argsort(-intensity)[[0, len(X)//4, len(X)//2, len(X)-1]][:n_pick]

    fig, axes = plt.subplots(2 * len(idxs), 8, figsize=(12, 2.6 * len(idxs)),
                             gridspec_kw=dict(hspace=0.6, wspace=0.05))
    for ri, i in enumerate(idxs):
        x_tiles = X[i].reshape(29, 8, 8)
        xh_tiles = xhat[i].reshape(29, 8, 8)
        for j in range(8):
            ax_in = axes[2 * ri,     j]
            ax_out = axes[2 * ri + 1, j]
            if j * 4 < 29:
                patch_idx = min(j * 4, 28)  # 0, 4, 8, 12, 16, 20, 24, 28
                ax_in.imshow(x_tiles[patch_idx], cmap='viridis_r', vmin=0, vmax=1)
                ax_out.imshow(xh_tiles[patch_idx], cmap='viridis_r', vmin=0, vmax=1)
                ax_in.set_title(f'p={patch_idx}', fontsize=6)
            for ax in (ax_in, ax_out):
                ax.set_xticks([]); ax.set_yticks([])
        axes[2*ri, 0].set_ylabel(f't={meta["t"][i]}\nphase={meta["phase"][i]}\nIN', fontsize=7)
        axes[2*ri+1, 0].set_ylabel('RECON', fontsize=7)
    fig.suptitle('CVAE input vs reconstruction (patches 0, 4, 8, 12, 16, 20, 24, 28)')
    fig.savefig(out / 'recon_samples.png', dpi=140); plt.close(fig)


def anomaly_over_time(anomaly, meta, out: Path):
    traj = meta['traj']; t = meta['t']; phase = meta['phase']; vm = meta['val_mask']
    n_traj = int(traj.max()) + 1
    fig, axes = plt.subplots(n_traj, 1, figsize=(12, 2 * n_traj), sharex=False)
    if n_traj == 1: axes = [axes]
    for ti in range(n_traj):
        m = traj == ti
        ax = axes[ti]
        ax.plot(t[m], anomaly[m], color='crimson', lw=1.2,
                label=('val' if vm[m].any() else 'train'))
        # phase shading
        ph = phase[m]; ts = t[m]
        for p in np.unique(ph):
            pm = ph == p
            if pm.any():
                ax.axvspan(ts[pm].min(), ts[pm].max(), alpha=0.05, color=plt.get_cmap('tab10')(int(p) % 10))
        ax.set_title(f'traj {ti} — anomaly score (|x−x̂|²) over time')
        ax.set_ylabel('recon err')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('timestep')
    fig.tight_layout()
    fig.savefig(out / 'anomaly_over_time.png', dpi=140); plt.close(fig)


def prior_sampling(model, Y, z_dim, out: Path, n_samples=6):
    device = next(model.parameters()).device
    # Condition on a few representative robot states
    idxs = np.linspace(0, len(Y) - 1, n_samples).astype(int)
    ys = torch.from_numpy(Y[idxs]).float().to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        xhat = model.decode(z, ys).cpu().numpy().reshape(n_samples, 29, 8, 8)

    fig, axes = plt.subplots(n_samples, 8, figsize=(12, 1.6 * n_samples),
                              gridspec_kw=dict(hspace=0.5, wspace=0.05))
    for r in range(n_samples):
        for c in range(8):
            p = min(c * 4, 28)
            ax = axes[r, c]
            ax.imshow(xhat[r, p], cmap='viridis_r', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'p={p}', fontsize=7)
        axes[r, 0].set_ylabel(f'z~N(0,I)\ny_idx={idxs[r]}', fontsize=7)
    fig.suptitle('Samples from the prior: different z, varied conditioning')
    fig.savefig(out / 'prior_sample_diversity.png', dpi=140); plt.close(fig)


def per_patch_anomaly(X, xhat, meta, out: Path):
    # per-patch MSE per timestep, reshape (T, 29)
    diff2 = (X - xhat) ** 2
    # Sort by traj then t
    order = np.lexsort((meta['t'], meta['traj']))
    diff2 = diff2[order]
    traj = meta['traj'][order]; t = meta['t'][order]; vm = meta['val_mask'][order]
    per_patch = diff2.reshape(len(X), 29, 64).mean(axis=2)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(per_patch.T, aspect='auto', cmap='hot', vmin=0, vmax=np.percentile(per_patch, 99))
    # traj boundaries
    changes = np.where(np.diff(traj) != 0)[0] + 0.5
    for c in changes:
        ax.axvline(c, color='cyan', linestyle='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('sample (concatenated trajs)')
    ax.set_ylabel('patch idx')
    ax.set_title('Per-patch reconstruction error across all trajectories\n(bright column = high anomaly at that timestep)')
    plt.colorbar(im, ax=ax, label='per-patch MSE')
    fig.tight_layout()
    fig.savefig(out / 'proximity_anomaly_map.png', dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=str(RUN))
    args = ap.parse_args()
    run = Path(args.run)
    out = run / 'plots'; out.mkdir(parents=True, exist_ok=True)

    m = np.load(run / 'metrics.npz')
    data = np.load(run / 'data_meta.npz')
    X = data['X']; Y = data['Y']; Z = data['Z']; xhat = data['xhat']
    anomaly = data['anomaly']
    meta = dict(t=data['t'], phase=data['phase'], traj=data['traj'],
                val_mask=data['val_mask'], tr_mask=data['tr_mask'])

    loss_curves(dict(m), out)
    print('wrote loss_curves.png')
    latent_scatter(Z, data['phase'], data['traj'], data['val_mask'], out)
    print('wrote latent_scatter.png')
    recon_samples(X, xhat, meta, out)
    print('wrote recon_samples.png')
    anomaly_over_time(anomaly, meta, out)
    print('wrote anomaly_over_time.png')
    per_patch_anomaly(X, xhat, meta, out)
    print('wrote proximity_anomaly_map.png')

    # Load model for prior sampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondVAE(x_dim=X.shape[1], y_dim=Y.shape[1], z_dim=Z.shape[1]).to(device)
    model.load_state_dict(torch.load(run / 'cvae.pt', map_location=device))
    prior_sampling(model, Y, z_dim=Z.shape[1], out=out)
    print('wrote prior_sample_diversity.png')


if __name__ == '__main__':
    main()
