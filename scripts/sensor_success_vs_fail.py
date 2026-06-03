"""Compare per-sensor activity in successful vs failed trajectories.

For each phase, plot mean sensor activity in success vs fail, with the
difference highlighted. Tells us whether failures correlate with degraded
proximity signal in any phase.

Outputs (in --output_dir):
  sensor_success_vs_fail_phase.png   — heatmap of mean activity per phase
  sensor_success_vs_fail_bars.png    — top-K diff per phase
  summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sensor_usage_timeline import (
    classify_phases, compute_sensor_activity,
)   # type: ignore  # noqa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True)
    p.add_argument("--prox_mapping_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_trajs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sensor_names = list(json.load(open(args.prox_mapping_json))["sensor_names"])
    N = len(sensor_names)
    phases_all = ["approach", "pregrasp", "grasp_lift", "transit", "place"]

    # success_sum[phase, sensor], success_count[phase], fail_sum[phase, sensor], fail_count[phase]
    succ_sum = np.zeros((len(phases_all), N), dtype=np.float64)
    fail_sum = np.zeros((len(phases_all), N), dtype=np.float64)
    succ_count = np.zeros(len(phases_all), dtype=np.int64)
    fail_count = np.zeros(len(phases_all), dtype=np.int64)
    succ_trajs = 0
    fail_trajs = 0

    run_dirs = sorted([p for p in Path(args.agg_dir).iterdir()
                       if p.is_dir() and p.name.startswith("run_")])
    for rd in run_dirs[: args.max_trajs]:
        h5p = rd / "house_1" / "trajectories_batch_1_of_1.h5"
        if not h5p.exists():
            continue
        try:
            with h5py.File(h5p, "r") as f:
                if "traj_0" not in f:
                    continue
                traj = f["traj_0"]
                phases, _ = classify_phases(traj)
                activity = compute_sensor_activity(traj, sensor_names)
                success = bool(traj["success"][-1])
        except Exception as e:
            print(f"skip {rd}: {e}")
            continue

        if success:
            succ_trajs += 1
            for i, ph in enumerate(phases_all):
                m = (phases == ph)
                if m.any():
                    succ_sum[i] += activity[m].sum(axis=0)
                    succ_count[i] += int(m.sum())
        else:
            fail_trajs += 1
            for i, ph in enumerate(phases_all):
                m = (phases == ph)
                if m.any():
                    fail_sum[i] += activity[m].sum(axis=0)
                    fail_count[i] += int(m.sum())

    succ_mean = np.where(succ_count[:, None] > 0, succ_sum / np.maximum(succ_count[:, None], 1), 0)
    fail_mean = np.where(fail_count[:, None] > 0, fail_sum / np.maximum(fail_count[:, None], 1), 0)
    diff = succ_mean - fail_mean

    print(f"[svf] processed {succ_trajs + fail_trajs} trajectories  "
          f"({succ_trajs} success, {fail_trajs} fail)")

    ph_used = [i for i, c in enumerate(succ_count + fail_count) if c > 0]

    # -- Plot 1: side-by-side phase heatmap
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.5), sharey=True)
    vmax = max(succ_mean.max(), fail_mean.max())
    h0 = axes[0].imshow(succ_mean[ph_used].T, aspect="auto", cmap="viridis",
                        vmin=0, vmax=vmax, origin="lower")
    axes[0].set_title(f"Success (n={succ_trajs})")
    h1 = axes[1].imshow(fail_mean[ph_used].T, aspect="auto", cmap="viridis",
                        vmin=0, vmax=vmax, origin="lower")
    axes[1].set_title(f"Failure (n={fail_trajs})")
    dmax = max(abs(diff).max(), 0.001)
    h2 = axes[2].imshow(diff[ph_used].T, aspect="auto", cmap="RdBu_r",
                        vmin=-dmax, vmax=dmax, origin="lower")
    axes[2].set_title("Success − Failure")

    for ax in axes:
        ax.set_xticks(range(len(ph_used)))
        ax.set_xticklabels([phases_all[i] for i in ph_used], rotation=15, ha="right",
                           fontsize=9)
        for boundary in [7, 15, 21]:
            ax.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.5)
    axes[0].set_yticks(range(N))
    axes[0].set_yticklabels(sensor_names, fontsize=7)
    plt.colorbar(h0, ax=axes[0], label="mean activity")
    plt.colorbar(h1, ax=axes[1], label="mean activity")
    plt.colorbar(h2, ax=axes[2], label="Δ activity (succ−fail)")
    fig.suptitle("Per-sensor activity: success vs failure trajectories")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "sensor_success_vs_fail_phase.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'sensor_success_vs_fail_phase.png'}")

    # -- Plot 2: top diff per phase
    K = 5
    n_panels = len(ph_used)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 4.5), sharey=False)
    if n_panels == 1:
        axes = [axes]
    for ax, ph_i in zip(axes, ph_used):
        d = diff[ph_i]
        # Top by absolute diff
        top = np.argsort(np.abs(d))[-K:][::-1]
        names_top = [sensor_names[j] for j in top]
        vals_top = d[top]
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals_top]
        ax.barh(range(K), vals_top[::-1], color=colors[::-1],
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(K))
        ax.set_yticklabels(names_top[::-1], fontsize=8)
        ax.set_title(f"{phases_all[ph_i]}\nsucc_n={succ_count[ph_i]}, fail_n={fail_count[ph_i]}",
                     fontsize=10)
        ax.set_xlabel("Δ activity (succ − fail)")
        ax.axvline(0, color="black", linewidth=0.5)
    fig.suptitle(f"Top sensors by success-vs-failure activity difference (n=50 total)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "sensor_success_vs_fail_bars.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'sensor_success_vs_fail_bars.png'}")

    # Summary
    summary = {
        "n_success": succ_trajs, "n_fail": fail_trajs,
        "phases": phases_all,
        "succ_mean_activity": succ_mean.tolist(),
        "fail_mean_activity": fail_mean.tolist(),
        "diff_succ_fail": diff.tolist(),
        "top_diff_per_phase": {
            phases_all[i]: [
                {"sensor": sensor_names[j], "diff": float(diff[i, j]),
                 "succ": float(succ_mean[i, j]), "fail": float(fail_mean[i, j])}
                for j in np.argsort(np.abs(diff[i]))[-5:][::-1]
            ] for i in ph_used
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[svf] outputs in {out}")


if __name__ == "__main__":
    main()
