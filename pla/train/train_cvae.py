"""Train the conditional VAE on skin proximity data.

Outputs to `runs/cvae_<run_name>/`:
- `cvae.pt`           trained weights
- `metrics.npz`       per-epoch train/val losses
- `data_meta.npz`     per-sample phase/traj/timestep labels for plotting
- `latent_{train,val}.npy`  per-sample latent means for plotting
"""
from __future__ import annotations
import argparse, time, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from pla.data.cvae_dataset import load_all, ProxCVAEDataset
from pla.models.cvae import CondVAE, elbo_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = os.environ.get(
    'PLA_CVAE_DATA_GLOB',
    str(REPO_ROOT / 'data/skin_pick_fixed_v1/**/trajectories_batch_*.h5'),
)
DEFAULT_OUT = REPO_ROOT / 'runs/cvae_v1'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] device={device}')

    meta = load_all(args.h5_glob)
    X, Y = meta['X'], meta['Y']
    N = len(X)

    # Train/val split by TRAJECTORY so val is unseen.
    rng = np.random.default_rng(args.seed)
    traj_perm = rng.permutation(meta['n_traj'])
    n_val_traj = max(1, int(meta['n_traj'] * args.val_frac))
    val_trajs = set(traj_perm[:n_val_traj].tolist())
    val_mask = np.array([t in val_trajs for t in meta['traj']])
    tr_mask = ~val_mask
    print(f'[train] N={N}  train={tr_mask.sum()}  val={val_mask.sum()}  val_trajs={sorted(val_trajs)}')

    tr_ds = ProxCVAEDataset(X[tr_mask], Y[tr_mask])
    va_ds = ProxCVAEDataset(X[val_mask], Y[val_mask])
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    model = CondVAE(x_dim=X.shape[1], y_dim=Y.shape[1], z_dim=args.z_dim,
                    hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    hist = {k: [] for k in ['train_loss', 'train_recon', 'train_kl',
                            'val_loss', 'val_recon', 'val_kl']}
    best_val = float('inf')
    best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        # anneal beta from 0 to target over first 20% of epochs
        beta = args.beta * min(1.0, ep / max(1, int(args.epochs * 0.2)))

        model.train()
        tr_meter = [0, 0, 0, 0]
        for x, y in tr_dl:
            x = x.to(device); y = y.to(device)
            xhat, mu, lv = model(x, y)
            losses = elbo_loss(xhat, x, mu, lv, beta=beta)
            opt.zero_grad(); losses['loss'].backward(); opt.step()
            n = len(x)
            tr_meter[0] += losses['loss'].item() * n
            tr_meter[1] += losses['recon'].item() * n
            tr_meter[2] += losses['kl'].item() * n
            tr_meter[3] += n
        tr_l, tr_r, tr_k = tr_meter[0]/tr_meter[3], tr_meter[1]/tr_meter[3], tr_meter[2]/tr_meter[3]

        model.eval()
        va_meter = [0, 0, 0, 0]
        with torch.no_grad():
            for x, y in va_dl:
                x = x.to(device); y = y.to(device)
                xhat, mu, lv = model(x, y)
                losses = elbo_loss(xhat, x, mu, lv, beta=beta)
                n = len(x)
                va_meter[0] += losses['loss'].item() * n
                va_meter[1] += losses['recon'].item() * n
                va_meter[2] += losses['kl'].item() * n
                va_meter[3] += n
        va_l, va_r, va_k = va_meter[0]/va_meter[3], va_meter[1]/va_meter[3], va_meter[2]/va_meter[3]

        hist['train_loss'].append(tr_l); hist['train_recon'].append(tr_r); hist['train_kl'].append(tr_k)
        hist['val_loss'].append(va_l); hist['val_recon'].append(va_r); hist['val_kl'].append(va_k)

        if va_r < best_val:
            best_val = va_r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f'ep {ep:4d}  β={beta:.4f}  train recon={tr_r:.4f} kl={tr_k:.2f}   val recon={va_r:.4f} kl={va_k:.2f}   [{time.time()-t0:.1f}s]')

    # Restore best model
    model.load_state_dict(best_state)

    # Compute latent embeddings for every sample
    model.eval()
    Z_all = np.empty((N, args.z_dim), dtype=np.float32)
    xhat_all = np.empty_like(X)
    anomaly = np.empty((N,), dtype=np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(Y).float().to(device)
        mu, lv = model.encode(x_t, y_t)
        Z_all = mu.cpu().numpy()
        xhat = model.decode(mu, y_t)
        xhat_all = xhat.cpu().numpy()
        anomaly = ((xhat - x_t) ** 2).sum(dim=-1).cpu().numpy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'cvae.pt')
    np.savez(out / 'metrics.npz', **hist)
    np.savez(out / 'data_meta.npz', phase=meta['phase'], traj=meta['traj'],
             t=meta['t'], val_mask=val_mask, tr_mask=tr_mask,
             X=X, Y=Y, Z=Z_all, xhat=xhat_all, anomaly=anomaly)
    print(f'[train] best val recon={best_val:.4f}')
    print(f'[train] saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5-glob', default=DEFAULT_GLOB)
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--z-dim', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=1e-2)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
