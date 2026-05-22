"""Compute and lock the clutter-bin assignments for the eval holdout houses.

Bins are computed FROM PLANNER DATA (privileged-state demonstrator), NOT from
any learned policy's rollouts — so they are independent of training outcomes
and stable across experiments. Output: a single JSON mapping house_id → bin
name, intended to be passed to `pla.eval_harness --clutter_bins_path <path>`
on every subsequent eval run.

Usage:
    python -m scripts.lock_clutter_bins \\
        --planner_root assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/FrankaSkinPickAndPlacePilotEvalHoldoutConfig/20260511_021228 \\
        --out analysis_output/eval_medium_v1/clutter_bins.json

The default `--planner_root` points to the existing eval-holdout planner data
(houses 11–20), which is what we have on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Re-use the harness's metrics + binning logic so the bins exactly match
# what the harness would compute internally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pla.eval_harness import metrics_from_traj, compute_clutter_bins  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--planner_root",
        type=str,
        default="assets/datagen/pick_and_place_skin_pilot_eval_holdout_v1/"
                "FrankaSkinPickAndPlacePilotEvalHoldoutConfig/20260511_021228",
        help="Root containing house_*/trajectories_batch_*.h5 from the planner demonstrator.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="analysis_output/eval_medium_v1/clutter_bins.json",
        help="Destination for the locked clutter_bins.json.",
    )
    p.add_argument(
        "--quantiles",
        type=str,
        default="0.333,0.667",
        help="Comma-separated lower,upper quantiles for the low|medium|high cuts.",
    )
    args = p.parse_args()

    root = Path(args.planner_root)
    if not root.exists():
        print(f"ERROR: planner_root does not exist: {root}", file=sys.stderr)
        return 1

    lo_q, hi_q = (float(x) for x in args.quantiles.split(","))

    metrics = []
    for h5p in sorted(root.glob("house_*/trajectories_batch_*.h5")):
        house = int(h5p.parent.name.removeprefix("house_"))
        with h5py.File(h5p, "r") as f:
            for tk in sorted(f.keys()):
                if not tk.startswith("traj_"):
                    continue
                metrics.append(
                    metrics_from_traj(f[tk], model="planner", seed=0, house=house, traj_key=tk)
                )
    if not metrics:
        print(f"ERROR: no trajectories found under {root}", file=sys.stderr)
        return 1

    print(f"[lock_clutter_bins] {len(metrics)} planner trajectories from "
          f"{len({m.house for m in metrics})} houses")

    bins = compute_clutter_bins(metrics, reference_model="planner", quantiles=(lo_q, hi_q))

    # Also stash the per-house clutter_signed values so the locked file is
    # self-documenting.
    from collections import defaultdict
    house_clutter = defaultdict(list)
    for m in metrics:
        house_clutter[m.house].append(m.clutter_signed)
    house_mean = {h: float(np.mean(vs)) for h, vs in house_clutter.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): v for k, v in sorted(bins.items())}
    # Add a metadata block under a reserved key (starts with underscore so
    # eval_harness ignores it when loading via int(k)).
    payload["__meta__"] = {
        "source_root": str(root),
        "n_houses": len(bins),
        "n_trajectories": len(metrics),
        "quantiles": [lo_q, hi_q],
        "per_house_clutter_signed": {str(k): v for k, v in sorted(house_mean.items())},
        "bin_thresholds": {
            "low_max_quantile": lo_q,
            "high_min_quantile": hi_q,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[lock_clutter_bins] bins: {bins}")
    print(f"[lock_clutter_bins] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
