"""Visualise the TCP path in space, colour-coded by phase.

For each trajectory, project TCP trajectory into world frame and plot xy and xz
projections, coloured by phase. Overlay object start position.

Produces:
  tcp_paths_success.png — overlay of 8 successful TCP paths
  tcp_paths_failure.png — overlay of 8 failed TCP paths
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
from sensor_usage_timeline import classify_phases, _tcp_world  # type: ignore


PHASE_COLORS = {"approach": "#7dafde", "pregrasp": "#ff9933",
                "grasp_lift": "#cc3300", "transit": "#9d4edd",
                "place": "#3a9b5a"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agg_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_each", type=int, default=8)
    return p.parse_args()


def collect_paths(agg_dir: Path, want_success: bool, n_max: int):
    paths = []
    for rd in sorted(p for p in agg_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
        if len(paths) >= n_max:
            break
        h5p = rd / "house_1" / "trajectories_batch_1_of_1.h5"
        if not h5p.exists():
            continue
        try:
            with h5py.File(h5p, "r") as f:
                if "traj_0" not in f:
                    continue
                traj = f["traj_0"]
                success = bool(traj["success"][-1])
                if success != want_success:
                    continue
                T = traj["obs/extra/tcp_pose"].shape[0]
                tcp_world = np.empty((T, 3), dtype=np.float32)
                for t in range(T):
                    tcp_world[t] = _tcp_world(traj["obs/extra/tcp_pose"][t],
                                              traj["obs/extra/robot_base_pose"][t])
                phases, _ = classify_phases(traj)
                obj_xyz = traj["obs/extra/obj_start"][0, :3].astype(np.float32)
                paths.append({"run": rd.name, "tcp": tcp_world,
                              "phases": phases, "obj_xyz": obj_xyz})
        except Exception:
            continue
    return paths


def plot_paths(paths: list, title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (xi, yi, xl, yl) in zip(axes,
                                    [(0, 1, "world x (m)", "world y (m)"),
                                     (0, 2, "world x (m)", "world z (m)")]):
        for p in paths:
            tcp = p["tcp"]
            phases = p["phases"]
            T = len(phases)
            # Plot segments coloured by phase
            for t in range(T - 1):
                ph = phases[t]
                color = PHASE_COLORS.get(ph, "#cccccc")
                ax.plot(tcp[t:t+2, xi], tcp[t:t+2, yi], color=color, alpha=0.55, linewidth=1.0)
            # Star at object xy
            obj = p["obj_xyz"]
            ax.scatter([obj[xi]], [obj[yi]], marker="*", s=150,
                       color="black", edgecolor="white", linewidth=0.8, zorder=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(("xy" if yi == 1 else "xz") + " projection")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    # Legend
    handles = [plt.Line2D([0], [0], color=c, label=p, linewidth=3) for p, c in PHASE_COLORS.items()]
    handles.append(plt.Line2D([0], [0], marker="*", color="black", linestyle="",
                              markersize=12, label="object start"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    succ_paths = collect_paths(Path(args.agg_dir), want_success=True, n_max=args.n_each)
    fail_paths = collect_paths(Path(args.agg_dir), want_success=False, n_max=args.n_each)

    print(f"[tcp-paths] {len(succ_paths)} success / {len(fail_paths)} fail paths")

    if succ_paths:
        plot_paths(succ_paths, f"TCP world-frame path — successful rollouts (n={len(succ_paths)})",
                   out / "tcp_paths_success.png")
        print(f"  wrote {out/'tcp_paths_success.png'}")
    if fail_paths:
        plot_paths(fail_paths, f"TCP world-frame path — failed rollouts (n={len(fail_paths)})",
                   out / "tcp_paths_failure.png")
        print(f"  wrote {out/'tcp_paths_failure.png'}")


if __name__ == "__main__":
    main()
