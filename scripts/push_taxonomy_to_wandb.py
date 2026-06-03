"""Push the failure taxonomy result (and aggregate plots) to wandb.

Run after `scripts/failure_taxonomy.py`:

    /opt/conda/envs/mlspaces/bin/python scripts/push_taxonomy_to_wandb.py \\
        --tax_dir eval_output/exp3_failure_taxonomy \\
        --project pact-paper-corl2026
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tax_dir", required=True)
    p.add_argument("--project", default="pact-paper-corl2026")
    p.add_argument("--entity", default=None)
    p.add_argument("--run_name", default="exp3_failure_taxonomy_n50")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tax = Path(args.tax_dir).resolve()
    chi = json.loads((tax / "chi_square.json").read_text())
    classifications = list(csv.DictReader(open(tax / "classifications.csv")))

    # Aggregate counts per condition.
    cats = chi["categories"]
    counts = {c: {cat: 0 for cat in cats + ["success"]} for c in ("baseline", "pact")}
    for r in classifications:
        c = r["condition"]
        cat = "success" if r["success"] == "True" else r["category"]
        counts[c][cat] = counts[c].get(cat, 0) + 1

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "n_per_condition": sum(counts["baseline"].values()),
            "categories": cats + ["success"],
        },
        tags=["pact", "exp3", "failure_taxonomy", "corl2026"],
    )
    print(f"[exp3-wandb] run url: {run.url}")

    # Summary metrics.
    run.summary["chi2"] = float(chi["chi2"])
    run.summary["chi2_p"] = float(chi["p"])
    run.summary["cramers_v"] = float(chi["cramers_v"])
    for cond in ("baseline", "pact"):
        for cat, n in counts[cond].items():
            run.summary[f"{cond}/n_{cat}"] = int(n)
        n_tot = sum(counts[cond].values())
        n_succ = counts[cond].get("success", 0)
        run.summary[f"{cond}/success_rate"] = float(n_succ / n_tot) if n_tot else 0.0

    # Failure rates as a wandb Table.
    cat_rows = []
    for cat in cats:
        cat_rows.append([cat, counts["baseline"].get(cat, 0),
                         counts["pact"].get(cat, 0)])
    failure_table = wandb.Table(columns=["category", "baseline", "pact"], data=cat_rows)
    run.log({"failure_taxonomy_table": failure_table})

    # Plot.
    plot_p = tax / "failure_taxonomy.png"
    if plot_p.exists():
        run.log({"failure_taxonomy_plot": wandb.Image(str(plot_p))})

    # Per-trajectory classifications.
    rows = [[r["condition"], int(r["run_idx"]), r["success"], r["category"],
             float(r["min_tcp_obj_dist"]),
             int(r["max_held_steps"]),
             float(r["held_lift_height"]),
             (None if r["min_position_error"] == "inf"
              else float(r["min_position_error"])),
             r["notes"]] for r in classifications]
    classif_table = wandb.Table(
        columns=["condition", "run_idx", "success", "category",
                 "min_tcp_obj_dist_m", "max_held_steps", "held_lift_height_m",
                 "min_position_error_m", "notes"],
        data=rows,
    )
    run.log({"per_trajectory_classifications": classif_table})

    run.finish()
    print("[exp3-wandb] done.")


if __name__ == "__main__":
    main()
