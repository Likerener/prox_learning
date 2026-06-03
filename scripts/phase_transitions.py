"""Plot when phases START and END across rollouts.

For each trajectory, find the first step where each phase begins and ends.
Plot a "gantt-chart-like" visualization showing the typical timeline of a
successful manipulation.

Outputs:
  phase_gantt_succ_vs_fail.png
  phase_transition_summary.json
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
PHASE_COLORS = {"approach": "#deebf7", "pregrasp": "#fed9a6",
                "grasp_lift": "#fdae6b", "transit": "#bcbddc",
                "place": "#a6dba0"}


def first_step_in_phase(phases: np.ndarray, target: str) -> int | None:
    idx = np.where(phases == target)[0]
    return int(idx[0]) if len(idx) else None


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

    rows: list[dict] = []
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
        except Exception:
            continue
        row = {"run": rd.name, "success": success, "T": T,
               "phases": phases.tolist()}
        for p in PHASES:
            ts = first_step_in_phase(phases, p)
            row[f"first_{p}"] = (ts / T if ts is not None else None)
        rows.append(row)

    succ_rows = [r for r in rows if r["success"]]
    fail_rows = [r for r in rows if not r["success"]]

    # Gantt-like: y-axis = trajectory index, x-axis = normalised time
    fig, (axs, axf) = plt.subplots(1, 2, figsize=(15, 8), sharey=False)

    def draw(ax, rs, title):
        ys = []
        for i, r in enumerate(rs):
            T = r["T"]
            ph = r["phases"]
            curr_ph, curr_start = ph[0], 0
            for t_idx in range(1, T):
                if ph[t_idx] != curr_ph:
                    ax.barh(i, (t_idx - curr_start) / T, left=curr_start / T,
                            height=0.85, color=PHASE_COLORS.get(curr_ph, "#eee"),
                            edgecolor="white", linewidth=0)
                    curr_ph = ph[t_idx]
                    curr_start = t_idx
            ax.barh(i, (T - curr_start) / T, left=curr_start / T,
                    height=0.85, color=PHASE_COLORS.get(curr_ph, "#eee"),
                    edgecolor="white", linewidth=0)
            ys.append(i)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(rs) - 0.5)
        ax.set_xlabel("normalised episode time")
        ax.set_yticks(ys)
        ax.set_yticklabels([r["run"] for r in rs], fontsize=7)
        ax.set_title(title)
        ax.invert_yaxis()

    draw(axs, succ_rows, f"Success (n={len(succ_rows)})")
    draw(axf, fail_rows, f"Failure (n={len(fail_rows)})")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS.values()]
    fig.legend(handles, list(PHASE_COLORS.keys()), loc="lower center",
               ncol=len(PHASE_COLORS), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-trajectory phase Gantt chart")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out / "phase_gantt_succ_vs_fail.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out/'phase_gantt_succ_vs_fail.png'}")

    # Phase entry timing distribution
    entries = {p: {"succ": [], "fail": []} for p in PHASES}
    for r in succ_rows:
        for p in PHASES:
            t = r.get(f"first_{p}")
            if t is not None:
                entries[p]["succ"].append(t)
    for r in fail_rows:
        for p in PHASES:
            t = r.get(f"first_{p}")
            if t is not None:
                entries[p]["fail"].append(t)

    # Box plot of entry times
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    positions_s = np.arange(len(PHASES)) - width/2
    positions_f = np.arange(len(PHASES)) + width/2
    succ_data = [entries[p]["succ"] for p in PHASES]
    fail_data = [entries[p]["fail"] for p in PHASES]
    succ_data = [d if d else [np.nan] for d in succ_data]
    fail_data = [d if d else [np.nan] for d in fail_data]
    ax.boxplot(succ_data, positions=positions_s, widths=width, patch_artist=True,
               boxprops=dict(facecolor="#1f77b4", edgecolor="black"))
    ax.boxplot(fail_data, positions=positions_f, widths=width, patch_artist=True,
               boxprops=dict(facecolor="#d62728", edgecolor="black"))
    ax.set_xticks(np.arange(len(PHASES)))
    ax.set_xticklabels(PHASES)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("entry time (fraction of episode)")
    ax.set_title("First entry time into each phase")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="success"),
               plt.Rectangle((0, 0), 1, 1, color="#d62728", label="failure")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "phase_entry_times.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'phase_entry_times.png'}")

    summary = {
        "n_success": len(succ_rows),
        "n_fail": len(fail_rows),
        "phase_entry_times": {
            p: {
                "succ_mean": float(np.mean(entries[p]["succ"])) if entries[p]["succ"] else None,
                "succ_median": float(np.median(entries[p]["succ"])) if entries[p]["succ"] else None,
                "fail_mean": float(np.mean(entries[p]["fail"])) if entries[p]["fail"] else None,
                "fail_median": float(np.median(entries[p]["fail"])) if entries[p]["fail"] else None,
                "succ_n_with_entry": len(entries[p]["succ"]),
                "fail_n_with_entry": len(entries[p]["fail"]),
            }
            for p in PHASES
        },
    }
    with open(out / "phase_transition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[phase-trans] done")


if __name__ == "__main__":
    main()
