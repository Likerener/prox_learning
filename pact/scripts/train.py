"""Train the prox-encoder transformer.

Logs per-step train loss and per-eval-step val metrics (MSE + per-axis MAE in
metres, plus mean Euclidean error). Logs to wandb if --use_wandb is set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prox_encoder.dataset import ProxWindowDataset, split_by_trajectory
from prox_encoder.model import ProxEncoder, ProxEncoderConfig, num_params


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: ProxEncoder, loader: DataLoader, device: torch.device,
             label_mean: np.ndarray, label_std: np.ndarray) -> dict:
    model.eval()
    mse_sum = 0.0
    n = 0
    abs_err_m_per_axis = np.zeros(3)
    eucl_sq = []
    for prox, label_n, meta in loader:
        prox = prox.to(device)
        label_n = label_n.to(device)
        pred_n = model(prox)
        mse = nn.functional.mse_loss(pred_n, label_n, reduction="sum").item()
        mse_sum += mse
        # Denormalize.
        pred = pred_n.detach().cpu().numpy() * label_std + label_mean
        target = meta["label_raw"].cpu().numpy()
        diff = pred - target
        abs_err_m_per_axis += np.abs(diff).sum(axis=0)
        eucl_sq.extend(np.linalg.norm(diff, axis=-1).tolist())
        n += prox.shape[0]
    val_mse = mse_sum / (n * 3)  # over 3 dims
    mae_axes = abs_err_m_per_axis / max(1, n)
    eucl_arr = np.asarray(eucl_sq)
    return {
        "val_mse_normalized": float(val_mse),
        "val_mae_m_x": float(mae_axes[0]),
        "val_mae_m_y": float(mae_axes[1]),
        "val_mae_m_z": float(mae_axes[2]),
        "val_mae_m_mean": float(mae_axes.mean()),
        "val_eucl_m_mean": float(eucl_arr.mean()),
        "val_eucl_m_median": float(np.median(eucl_arr)),
        "val_eucl_m_p90": float(np.percentile(eucl_arr, 90)),
        "val_n": int(n),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="pact/outputs_prox/cache.npz")
    p.add_argument("--out_dir", default="pact/outputs_prox/runs/v1")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run_name", default="prox_encoder_v1")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="prox-encoder")
    p.add_argument("--smoke", action="store_true",
                   help="200-step smoke run with tiny eval cadence.")
    args = p.parse_args()

    if args.smoke:
        args.steps = 200
        args.val_every = 100
        args.ckpt_every = 200
        args.warmup = 20
        args.log_every = 20
        args.run_name = args.run_name + "_smoke"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_idx, val_idx = split_by_trajectory(args.cache, args.val_frac, args.seed)
    train_ds = ProxWindowDataset(args.cache, indices=train_idx)
    val_ds = ProxWindowDataset(args.cache, indices=val_idx)
    print(f"[train] N_train={len(train_ds)} N_val={len(val_ds)} "
          f"window={train_ds.window} sensors={train_ds.n_sensors}")
    print(f"[train] label_mean={train_ds.label_mean} label_std={train_ds.label_std}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Discover input T from one batch.
    prox, label_n, meta = next(iter(train_loader))
    T = prox.shape[1]
    print(f"[train] sample shapes: prox={tuple(prox.shape)} label={tuple(label_n.shape)} T={T}")

    cfg = ProxEncoderConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_T=max(T + 4, 64),
    )
    model = ProxEncoder(cfg).to(args.device)
    print(f"[train] model params: {num_params(model)/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=str(out_dir),
            config={**vars(args), "model_params": num_params(model),
                    "N_train": len(train_ds), "N_val": len(val_ds),
                    "T": T},
        )

    def cycle(loader):
        while True:
            for b in loader:
                yield b

    train_iter = cycle(train_loader)
    t0 = time.time()
    best_mae = float("inf")
    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        prox, label_n, _ = next(train_iter)
        prox = prox.to(args.device, non_blocking=True)
        label_n = label_n.to(args.device, non_blocking=True)

        model.train()
        pred = model(prox)
        loss = nn.functional.mse_loss(pred, label_n)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"[train] step {step+1:6d}/{args.steps} loss={loss.item():.4f} "
                  f"lr={lr:.2e} sps={sps:.1f}")
            if run is not None:
                run.log({"train/loss_mse_norm": float(loss.item()),
                         "train/lr": lr, "train/sps": sps}, step=step + 1)

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            v = evaluate(model, val_loader, torch.device(args.device),
                         train_ds.label_mean, train_ds.label_std)
            print(f"[val ] step {step+1:6d}: "
                  f"mse_n={v['val_mse_normalized']:.4f} "
                  f"mae_m=(x={v['val_mae_m_x']:.3f},y={v['val_mae_m_y']:.3f},"
                  f"z={v['val_mae_m_z']:.3f}) "
                  f"eucl_m_mean={v['val_eucl_m_mean']:.3f} "
                  f"median={v['val_eucl_m_median']:.3f}")
            if run is not None:
                run.log({**{f"val/{k}": vv for k, vv in v.items()}}, step=step + 1)
            if v["val_mae_m_mean"] < best_mae:
                best_mae = v["val_mae_m_mean"]
                torch.save({"model": model.state_dict(),
                            "cfg": cfg.__dict__,
                            "step": step + 1,
                            "label_mean": train_ds.label_mean,
                            "label_std": train_ds.label_std,
                            "prox_mean": train_ds.prox_mean,
                            "prox_std": train_ds.prox_std,
                            "window": train_ds.window,
                            "val": v},
                           out_dir / "ckpt_best.pt")
                print(f"[ckpt] new best mae_mean={best_mae:.3f} -> ckpt_best.pt")

        if (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "step": step + 1,
                        "label_mean": train_ds.label_mean,
                        "label_std": train_ds.label_std,
                        "prox_mean": train_ds.prox_mean,
                        "prox_std": train_ds.prox_std,
                        "window": train_ds.window},
                       out_dir / f"ckpt_step_{step+1:06d}.pt")

    print(f"[done] best val mae_mean = {best_mae:.3f} m")
    if run is not None:
        run.summary["best_val_mae_m_mean"] = best_mae
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
