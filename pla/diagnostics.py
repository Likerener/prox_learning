"""Sanity-check plots for the franka_skin pick-and-place dataset.

Run this whenever a new dataset run lands to confirm:
- proximity actually has signal (max > 0, distribution non-degenerate)
- per-sensor saturation rates are reasonable
- trajectory lengths are in expected range
- action/qpos distributions look like joint trajectories (bounded, smooth)
- language coverage (number of unique task descriptions, frequency)

Outputs a folder of PNGs and a `summary.json`. Designed to be the
gatekeeper: if these plots look wrong, do not start training.

Usage:
  python -m pla.diagnostics --root <dataset_root> --out <out_dir>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pla.dataset import (
    SENSOR_NAMES,
    _decode_jsonrow,
    _extract_task_description,
)


def _gather_trajs(root: Path) -> list[tuple[str, str, int]]:
    """Return list of (h5_path, traj_key, T) for all successful trajectories."""
    out = []
    for h5 in sorted(root.glob("house_*/trajectories_batch_*.h5")):
        try:
            with h5py.File(h5, "r") as f:
                for tk in f.keys():
                    if not tk.startswith("traj_"):
                        continue
                    t = f[tk]
                    if not bool(t["success"][-1]):
                        continue
                    out.append((str(h5), tk, int(t["success"].shape[0])))
        except OSError:
            continue
    return out


def proximity_summary(trajs: list, out_dir: Path) -> dict:
    """Per-sensor depth stats across a sample of timesteps; saturation map."""
    rng = random.Random(0)
    sample = trajs if len(trajs) < 200 else rng.sample(trajs, 200)

    per_sensor = {s: {"min": np.inf, "max": -np.inf, "sum": 0.0, "sqsum": 0.0,
                       "n_zero": 0, "n_total": 0} for s in SENSOR_NAMES}
    histogram_bins = np.linspace(0, 5, 51)
    histogram_counts = np.zeros(len(histogram_bins) - 1, dtype=np.int64)

    for h5_path, tk, T in sample:
        with h5py.File(h5_path, "r") as f:
            for s in SENSOR_NAMES:
                arr = f[f"{tk}/obs/proximity/{s}"][:]  # (T, n_substeps, 8, 8)
                arr = arr.mean(axis=1)                 # (T, 8, 8)
                ps = per_sensor[s]
                ps["min"] = float(min(ps["min"], arr.min()))
                ps["max"] = float(max(ps["max"], arr.max()))
                ps["sum"] += float(arr.sum())
                ps["sqsum"] += float((arr.astype(np.float64) ** 2).sum())
                ps["n_zero"] += int((arr == 0).sum())
                ps["n_total"] += int(arr.size)
                histogram_counts += np.histogram(arr, bins=histogram_bins)[0].astype(np.int64)

    # finalize stats
    summary = {}
    for s, st in per_sensor.items():
        n = st["n_total"]
        if n == 0:
            continue
        mu = st["sum"] / n
        var = max(0.0, st["sqsum"] / n - mu * mu)
        summary[s] = {
            "min": st["min"], "max": st["max"], "mean": mu, "std": var ** 0.5,
            "zero_frac": st["n_zero"] / n,
        }

    # plot 1: depth histogram (across all sensors)
    fig, ax = plt.subplots(figsize=(8, 4))
    centers = 0.5 * (histogram_bins[1:] + histogram_bins[:-1])
    ax.bar(centers, histogram_counts, width=histogram_bins[1] - histogram_bins[0])
    ax.set_yscale("log")
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Proximity depth distribution ({len(sample)} sampled trajectories)")
    fig.tight_layout()
    fig.savefig(out_dir / "01_proximity_depth_histogram.png", dpi=120)
    plt.close(fig)

    # plot 2: per-sensor mean depth (ordered by sensor name)
    sensors = list(summary.keys())
    means = [summary[s]["mean"] for s in sensors]
    zero_fracs = [summary[s]["zero_frac"] for s in sensors]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].bar(range(len(sensors)), means, color="steelblue")
    axes[0].set_ylabel("mean depth (m)")
    axes[0].set_title("Per-sensor mean depth")
    axes[1].bar(range(len(sensors)), zero_fracs, color="firebrick")
    axes[1].set_ylabel("fraction of pixels == 0")
    axes[1].set_xticks(range(len(sensors)))
    axes[1].set_xticklabels(sensors, rotation=90, fontsize=7)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Zero-pixel fraction (1.0 = no signal at all)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_proximity_per_sensor_stats.png", dpi=120)
    plt.close(fig)

    return summary


def trajectory_length_plot(trajs: list, out_dir: Path) -> dict:
    Ts = [T for _, _, T in trajs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(Ts, bins=40, color="seagreen")
    ax.set_xlabel("trajectory length (timesteps)")
    ax.set_ylabel("count")
    ax.set_title(f"Episode length distribution (N={len(Ts)})")
    fig.tight_layout()
    fig.savefig(out_dir / "03_episode_length_hist.png", dpi=120)
    plt.close(fig)
    return {"min": int(min(Ts)), "max": int(max(Ts)),
            "mean": float(np.mean(Ts)), "median": float(np.median(Ts)),
            "n": len(Ts)}


def action_qpos_distributions(trajs: list, out_dir: Path) -> dict:
    """7 arm joints: histogram of qpos and joint_pos action across timesteps."""
    rng = random.Random(0)
    sample = trajs if len(trajs) < 100 else rng.sample(trajs, 100)
    qpos_buf, action_buf = [], []
    for h5_path, tk, T in sample:
        with h5py.File(h5_path, "r") as f:
            for t in range(0, T, max(1, T // 30)):
                try:
                    q = _decode_jsonrow(f[f"{tk}/obs/agent/qpos"][t]).get("arm", [])
                    a = _decode_jsonrow(f[f"{tk}/actions/joint_pos"][t]).get("arm", [])
                    if len(q) == 7:
                        qpos_buf.append(q)
                    if len(a) == 7:
                        action_buf.append(a)
                except Exception:
                    continue
    qpos_arr = np.asarray(qpos_buf)
    act_arr = np.asarray(action_buf)

    fig, axes = plt.subplots(2, 7, figsize=(18, 5), sharey="row")
    for j in range(7):
        axes[0, j].hist(qpos_arr[:, j], bins=40, color="steelblue")
        axes[0, j].set_title(f"qpos[{j}]")
        axes[1, j].hist(act_arr[:, j], bins=40, color="darkorange")
        axes[1, j].set_title(f"action[{j}]")
        axes[1, j].set_xlabel("rad")
    axes[0, 0].set_ylabel("qpos count")
    axes[1, 0].set_ylabel("action count")
    fig.suptitle(f"qpos / action joint distributions (N_qpos={len(qpos_buf)}, N_act={len(action_buf)})")
    fig.tight_layout()
    fig.savefig(out_dir / "04_qpos_action_distribution.png", dpi=120)
    plt.close(fig)
    return {
        "qpos_n": len(qpos_buf), "action_n": len(action_buf),
        "qpos_min": qpos_arr.min(axis=0).tolist() if len(qpos_arr) else [],
        "qpos_max": qpos_arr.max(axis=0).tolist() if len(qpos_arr) else [],
        "action_min": act_arr.min(axis=0).tolist() if len(act_arr) else [],
        "action_max": act_arr.max(axis=0).tolist() if len(act_arr) else [],
    }


def language_coverage(trajs: list, out_dir: Path) -> dict:
    descriptions = []
    for h5_path, tk, _ in trajs:
        with h5py.File(h5_path, "r") as f:
            scene = f[tk]["obs_scene"][()]
            if isinstance(scene, np.ndarray):
                scene = scene.tobytes()
            descriptions.append(_extract_task_description(scene))
    counter = Counter(descriptions)
    top = counter.most_common(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), [c for _, c in top], color="purple")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([(d or "<empty>")[:80] for d, _ in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("episode count")
    ax.set_title(f"Top-20 task descriptions (unique: {len(counter)} / {len(descriptions)})")
    fig.tight_layout()
    fig.savefig(out_dir / "05_language_top_descriptions.png", dpi=120)
    plt.close(fig)
    return {"n_episodes": len(descriptions), "n_unique": len(counter),
            "empty_count": counter.get("", 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="Dataset root containing house_*/trajectories_batch_*.h5")
    p.add_argument("--out", type=str, required=True,
                   help="Output dir for plots + summary.json")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = _gather_trajs(root)
    print(f"[diag] {len(trajs)} successful trajectories under {root}")
    if not trajs:
        print("[diag] no trajectories found; aborting.")
        return

    summary = {
        "n_trajectories": len(trajs),
        "lengths": trajectory_length_plot(trajs, out_dir),
        "actions_qpos": action_qpos_distributions(trajs, out_dir),
        "language": language_coverage(trajs, out_dir),
        "proximity": proximity_summary(trajs, out_dir),
    }

    # Top-line gating signal: is proximity actually nonzero?
    prox_max_overall = max(s["max"] for s in summary["proximity"].values())
    summary["proximity_overall_max"] = prox_max_overall
    summary["proximity_signal_ok"] = prox_max_overall > 0.001

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[diag] wrote {out_dir/'summary.json'}")
    print(f"[diag] proximity_overall_max = {prox_max_overall:.4f} "
          f"({'OK' if summary['proximity_signal_ok'] else 'BROKEN — re-collect data'})")


if __name__ == "__main__":
    main()
