"""Action variance per phase: where is the policy doing the most work?

For each trajectory, compute the step-to-step action delta norm in each phase.
Aggregates per-phase mean delta-norm across successful vs failed trajectories.

Outputs:
  action_delta_per_phase.png
  action_summary.json
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

    succ_deltas = {p: [] for p in PHASES}
    fail_deltas = {p: [] for p in PHASES}

    run_dirs = sorted([p for p in Path(args.agg_dir).iterdir()
                       if p.is_dir() and p.name.startswith("run_")])
    n_succ = 0
    n_fail = 0
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
                # Get commanded actions from JSON
                ca = traj["actions/commanded_action"][:]   # (T, ?) bytes
                # Parse JSON
                actions = []
                for row in ca:
                    try:
                        s = bytes(row).decode("utf-8", errors="ignore").split("\x00")[0].strip()
                        d = json.loads(s) if s else {}
                        arm = np.asarray(d.get("arm", [0]*7), dtype=np.float32)
                        grip = np.asarray(d.get("gripper", [0]), dtype=np.float32)
                        actions.append(np.concatenate([arm, grip]))
                    except Exception:
                        actions.append(np.zeros(8, dtype=np.float32))
                actions = np.array(actions)               # (T, 8)
                success = bool(traj["success"][-1])
        except Exception:
            continue
        # Step deltas
        deltas = np.linalg.norm(actions[1:] - actions[:-1], axis=1)   # (T-1,)
        # Assign to phases (skip last step)
        for i in range(len(deltas)):
            ph = phases[i]
            if ph in PHASES:
                (succ_deltas if success else fail_deltas)[ph].append(float(deltas[i]))
        if success:
            n_succ += 1
        else:
            n_fail += 1

    print(f"[action-var] processed {n_succ} success, {n_fail} fail trajectories")

    # Plot: box plot of action delta norms per phase
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.35
    pos_s = np.arange(len(PHASES)) - width/2
    pos_f = np.arange(len(PHASES)) + width/2
    succ_data = [succ_deltas[p] if succ_deltas[p] else [np.nan] for p in PHASES]
    fail_data = [fail_deltas[p] if fail_deltas[p] else [np.nan] for p in PHASES]
    ax.boxplot(succ_data, positions=pos_s, widths=width, patch_artist=True,
               boxprops=dict(facecolor="#1f77b4", edgecolor="black"),
               showfliers=False)
    ax.boxplot(fail_data, positions=pos_f, widths=width, patch_artist=True,
               boxprops=dict(facecolor="#d62728", edgecolor="black"),
               showfliers=False)
    ax.set_xticks(np.arange(len(PHASES)))
    ax.set_xticklabels(PHASES)
    ax.set_ylabel("||a_{t+1} − a_t||₂ per step")
    ax.set_title(f"Per-step action-delta norm by phase  (succ n={n_succ}, fail n={n_fail})")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="success"),
               plt.Rectangle((0, 0), 1, 1, color="#d62728", label="failure")]
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "action_delta_per_phase.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'action_delta_per_phase.png'}")

    summary = {
        "n_success": n_succ,
        "n_fail": n_fail,
        "phases": PHASES,
        "succ_mean_delta": {p: float(np.mean(succ_deltas[p])) if succ_deltas[p] else 0
                            for p in PHASES},
        "fail_mean_delta": {p: float(np.mean(fail_deltas[p])) if fail_deltas[p] else 0
                            for p in PHASES},
        "succ_median_delta": {p: float(np.median(succ_deltas[p])) if succ_deltas[p] else 0
                              for p in PHASES},
        "fail_median_delta": {p: float(np.median(fail_deltas[p])) if fail_deltas[p] else 0
                              for p in PHASES},
    }
    with open(out / "action_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out/'action_summary.json'}")
    print()
    print(f"  mean delta by phase (succ vs fail):")
    for p in PHASES:
        print(f"    {p:>11s}: succ={summary['succ_mean_delta'][p]:.4f}, "
              f"fail={summary['fail_mean_delta'][p]:.4f}")


if __name__ == "__main__":
    main()
