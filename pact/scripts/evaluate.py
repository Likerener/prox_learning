"""Evaluate a trained prox-encoder checkpoint and produce plots.

Outputs (under --output_dir):
  evaluation_metrics.json    overall + per-sensor metrics
  scatter_xyz.png            pred vs gt per axis
  error_hist.png             histograms of per-axis errors
  euclidean_hist.png         distribution of ||pred - gt||
  scatter_3d.png             3D scatter of predicted vs gt positions
  per_sensor_mae.png         bar chart of per-sensor mean Euclidean error
  predictions.npz            raw arrays (preds, gts, sensor_ids, traj_ids, t)

If --use_wandb is set (with --run_id or --run_name), the same metrics + the
generated PNGs are logged to wandb so train and eval land in one run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prox_encoder.dataset import ProxWindowDataset, split_by_trajectory
from prox_encoder.model import ProxEncoder, ProxEncoderConfig


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    if ss_tot < 1e-8:
        return 0.0
    return max(-1.0, 1.0 - ss_res / ss_tot)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", default="pact/outputs_prox/cache.npz")
    p.add_argument("--output_dir", default=None,
                   help="If omitted, written next to the checkpoint as eval/.")
    p.add_argument("--split", choices=("val", "train", "all"), default="val")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="prox-encoder")
    p.add_argument("--wandb_run_name", default=None,
                   help="If set, log to a wandb run with this name (creating if needed).")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = split_by_trajectory(args.cache, args.val_frac, args.seed)
    all_idx = np.arange(np.load(args.cache)["traj_id"].shape[0])
    if args.split == "val":
        idx = val_idx
    elif args.split == "train":
        idx = train_idx
    else:
        idx = all_idx
    ds = ProxWindowDataset(args.cache, indices=idx)
    print(f"[eval] split={args.split} N={len(ds)}")

    cfg = ProxEncoderConfig(**ckpt["cfg"])
    model = ProxEncoder(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded ckpt step {ckpt.get('step','?')}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    preds_n = []
    gts_n = []
    gts_m = []
    sensor_ids = []
    traj_ids = []
    t_idxs = []
    with torch.no_grad():
        for prox, label_n, meta in loader:
            prox = prox.to(args.device, non_blocking=True)
            pred_n = model(prox).cpu().numpy()
            preds_n.append(pred_n)
            gts_n.append(label_n.numpy())
            gts_m.append(meta["label_raw"].numpy())
            sensor_ids.append(meta["sensor_id"].numpy())
            traj_ids.append(meta["traj_id"].numpy())
            t_idxs.append(meta["t"].numpy())
    preds_n = np.concatenate(preds_n, axis=0)
    gts_n = np.concatenate(gts_n, axis=0)
    gts_m = np.concatenate(gts_m, axis=0)
    sensor_ids = np.concatenate(sensor_ids, axis=0)
    traj_ids = np.concatenate(traj_ids, axis=0)
    t_idxs = np.concatenate(t_idxs, axis=0)

    # Denormalize predictions.
    preds_m = preds_n * ds.label_std + ds.label_mean

    diff = preds_m - gts_m
    abs_diff = np.abs(diff)
    eucl = np.linalg.norm(diff, axis=-1)

    mae_axes = abs_diff.mean(axis=0)
    rmse_axes = np.sqrt((diff ** 2).mean(axis=0))
    metrics = {
        "ckpt_step": int(ckpt.get("step", -1)),
        "split": args.split,
        "num_samples": int(preds_m.shape[0]),
        "mae_x": float(mae_axes[0]),
        "mae_y": float(mae_axes[1]),
        "mae_z": float(mae_axes[2]),
        "mae_mean": float(mae_axes.mean()),
        "rmse_x": float(rmse_axes[0]),
        "rmse_y": float(rmse_axes[1]),
        "rmse_z": float(rmse_axes[2]),
        "rmse_mean": float(rmse_axes.mean()),
        "r2_x": safe_r2(gts_m[:, 0], preds_m[:, 0]),
        "r2_y": safe_r2(gts_m[:, 1], preds_m[:, 1]),
        "r2_z": safe_r2(gts_m[:, 2], preds_m[:, 2]),
        "eucl_mean": float(eucl.mean()),
        "eucl_median": float(np.median(eucl)),
        "eucl_p90": float(np.percentile(eucl, 90)),
        "eucl_p99": float(np.percentile(eucl, 99)),
        "eucl_min": float(eucl.min()),
        "eucl_max": float(eucl.max()),
    }
    print(json.dumps(metrics, indent=2))

    # Per-sensor breakdown.
    per_sensor = {}
    for sid in np.unique(sensor_ids):
        mask = sensor_ids == sid
        per_sensor[str(ds.sensor_names[sid])] = {
            "n": int(mask.sum()),
            "mae_mean": float(abs_diff[mask].mean()),
            "eucl_mean": float(eucl[mask].mean()),
            "eucl_median": float(np.median(eucl[mask])),
        }
    metrics["per_sensor"] = per_sensor

    (out_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "predictions.npz",
        preds_m=preds_m, gts_m=gts_m, sensor_ids=sensor_ids,
        traj_ids=traj_ids, t=t_idxs,
        sensor_names=ds.sensor_names,
    )

    # ---- Plots -------------------------------------------------------------
    axes_lbl = ("x", "y", "z")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for a in range(3):
        ax = axs[a]
        ax.scatter(gts_m[:, a], preds_m[:, a], s=4, alpha=0.3)
        lo = float(min(gts_m[:, a].min(), preds_m[:, a].min()))
        hi = float(max(gts_m[:, a].max(), preds_m[:, a].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"gt_{axes_lbl[a]} [m]")
        ax.set_ylabel(f"pred_{axes_lbl[a]} [m]")
        ax.set_title(f"{axes_lbl[a]}: MAE={mae_axes[a]:.3f} m, R²={metrics[f'r2_{axes_lbl[a]}']:.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Prox-encoder pred vs gt ({args.split}, N={preds_m.shape[0]})")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_xyz.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for a in range(3):
        ax = axs[a]
        ax.hist(diff[:, a], bins=60, alpha=0.85, color="C0")
        ax.axvline(0, color="r", lw=1)
        ax.set_title(f"{axes_lbl[a]} error [m]  (mean={diff[:, a].mean():.3f})")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eucl, bins=60, color="C2")
    ax.axvline(metrics["eucl_mean"], color="r", lw=1, label=f"mean {metrics['eucl_mean']:.3f}")
    ax.axvline(metrics["eucl_median"], color="orange", lw=1, label=f"median {metrics['eucl_median']:.3f}")
    ax.set_xlabel("‖pred - gt‖ [m]")
    ax.set_ylabel("samples")
    ax.set_title("Euclidean error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "euclidean_hist.png", dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    n_show = min(2000, preds_m.shape[0])
    rs = np.random.default_rng(0).choice(preds_m.shape[0], size=n_show, replace=False)
    ax.scatter(gts_m[rs, 0], gts_m[rs, 1], gts_m[rs, 2], c="C0", s=6, alpha=0.4, label="gt")
    ax.scatter(preds_m[rs, 0], preds_m[rs, 1], preds_m[rs, 2], c="C3", s=6, alpha=0.4, label="pred")
    ax.set_xlabel("x_sensor")
    ax.set_ylabel("y_sensor")
    ax.set_zlabel("z_sensor")
    ax.set_title(f"3D positions (random {n_show} of {preds_m.shape[0]})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_3d.png", dpi=130)
    plt.close(fig)

    sensor_order = sorted(per_sensor.keys(),
                          key=lambda k: per_sensor[k]["eucl_mean"])
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(sensor_order)), 4))
    vals = [per_sensor[s]["eucl_mean"] for s in sensor_order]
    ns = [per_sensor[s]["n"] for s in sensor_order]
    ax.bar(range(len(sensor_order)), vals, color="C0")
    ax.set_xticks(range(len(sensor_order)))
    ax.set_xticklabels(sensor_order, rotation=45, ha="right")
    for i, n in enumerate(ns):
        ax.text(i, vals[i] + 0.005, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean ‖pred - gt‖ [m]")
    ax.set_title("Per-sensor Euclidean error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_sensor_mae.png", dpi=130)
    plt.close(fig)

    if args.use_wandb:
        import wandb
        run_name = args.wandb_run_name or ckpt_path.parent.name + "_eval"
        run = wandb.init(project=args.wandb_project, name=run_name,
                         dir=str(out_dir), reinit=True)
        run.log({**{f"eval_{args.split}/{k}": v
                    for k, v in metrics.items() if not isinstance(v, dict)},
                 f"eval_{args.split}/scatter_xyz": wandb.Image(str(out_dir / "scatter_xyz.png")),
                 f"eval_{args.split}/error_hist": wandb.Image(str(out_dir / "error_hist.png")),
                 f"eval_{args.split}/euclidean_hist": wandb.Image(str(out_dir / "euclidean_hist.png")),
                 f"eval_{args.split}/scatter_3d": wandb.Image(str(out_dir / "scatter_3d.png")),
                 f"eval_{args.split}/per_sensor_mae": wandb.Image(str(out_dir / "per_sensor_mae.png"))})
        run.summary[f"eval_{args.split}_eucl_mean_m"] = metrics["eucl_mean"]
        run.summary[f"eval_{args.split}_mae_mean_m"] = metrics["mae_mean"]
        run.finish()

    print(f"[eval] wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
