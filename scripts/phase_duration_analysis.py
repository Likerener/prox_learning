"""Phase-duration analysis: how long does each manipulation phase last?

For each h5 in the eval dir, classify steps into phases, then compute the
duration of each phase per trajectory. Aggregates:
  * Mean phase duration (success vs fail)
  * Phase entry/exit timings (normalised)
  * Box plot of durations per phase

Outputs:
  phase_durations_box.png
  phase_entry_exit.png
  phase_duration_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_usage_timeline import classify_phases  # type: ignore


PHASES = ["approach", "pregrasp", "grasp_lift", "transit", "place"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_trajs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Per-trajectory: { phase: duration (frac of episode), entry_t, exit_t, success }
    per_traj_data = []
    succ_durations = {p: [] for p in PHASES}
    fail_durations = {p: [] for p in PHASES}

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
                T = len(phases)
                success = bool(traj["success"][-1])
        except Exception as e:
            print(f"skip {rd}: {e}")
            continue
        durations = {p: float((phases == p).sum() / T) for p in PHASES}
        per_traj_data.append({"run": rd.name, "success": success, "T": T,
                              **{f"dur_{p}": durations[p] for p in PHASES}})
        for p in PHASES:
            (succ_durations if success else fail_durations)[p].append(durations[p])

    # Box plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.35
    positions_s = np.arange(len(PHASES)) - width/2
    positions_f = np.arange(len(PHASES)) + width/2
    box_s = ax.boxplot([succ_durations[p] for p in PHASES], positions=positions_s,
                       widths=width, patch_artist=True,
                       boxprops=dict(facecolor="#1f77b4", edgecolor="black"))
    box_f = ax.boxplot([fail_durations[p] for p in PHASES], positions=positions_f,
                       widths=width, patch_artist=True,
                       boxprops=dict(facecolor="#d62728", edgecolor="black"))
    ax.set_xticks(np.arange(len(PHASES)))
    ax.set_xticklabels(PHASES)
    ax.set_ylabel("fraction of episode in phase")
    ax.set_title(f"Phase-duration distribution: success vs failure  "
                 f"(succ n={len(succ_durations['approach'])}, fail n={len(fail_durations['approach'])})")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="success"),
               plt.Rectangle((0, 0), 1, 1, color="#d62728", label="failure")]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "phase_durations_box.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'phase_durations_box.png'}")

    # Mean durations table
    summary = {
        "phases": PHASES,
        "n_success": len(succ_durations["approach"]),
        "n_fail": len(fail_durations["approach"]),
        "succ_mean_duration": {p: float(np.mean(succ_durations[p])) if succ_durations[p] else 0
                               for p in PHASES},
        "fail_mean_duration": {p: float(np.mean(fail_durations[p])) if fail_durations[p] else 0
                               for p in PHASES},
        "succ_median_duration": {p: float(np.median(succ_durations[p])) if succ_durations[p] else 0
                                 for p in PHASES},
        "fail_median_duration": {p: float(np.median(fail_durations[p])) if fail_durations[p] else 0
                                 for p in PHASES},
    }
    with open(out / "phase_duration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[phase-dur] mean phase fraction:")
    print(f"  {'phase':>11s}  {'succ':>8s}  {'fail':>8s}  {'diff':>8s}")
    for p in PHASES:
        s = summary["succ_mean_duration"][p]
        f = summary["fail_mean_duration"][p]
        print(f"  {p:>11s}  {s:>8.3f}  {f:>8.3f}  {s-f:>+8.3f}")


if __name__ == "__main__":
    main()
