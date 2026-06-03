"""Push the aggregate Exp 1 + Exp 2 + Exp 3 results (including plots) to wandb
as a single CORL2026 paper-figure run.

Run after `scripts/plot_mask_experiments.py` and `scripts/failure_taxonomy.py`:

    /opt/conda/envs/mlspaces/bin/python scripts/push_exp_aggregate_to_wandb.py \\
        --plots_dir eval_output/exp_plots \\
        --tax_dir eval_output/exp3_failure_taxonomy \\
        --project pact-paper-corl2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plots_dir", required=True,
                   help="Output of scripts/plot_mask_experiments.py "
                        "(contains exp1_bar.png, exp2_bar.png, all_rates.json)")
    p.add_argument("--tax_dir", required=True,
                   help="Output of scripts/failure_taxonomy.py "
                        "(contains chi_square.json, failure_taxonomy.png)")
    p.add_argument("--epoch_sweep_dir", default="",
                   help="Optional: output of scripts/run_pact_epoch_sweep.py")
    p.add_argument("--project", default="pact-paper-corl2026")
    p.add_argument("--entity", default=None)
    p.add_argument("--run_name", default="paper_aggregate_2026_05_26")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plots = Path(args.plots_dir).resolve()
    tax = Path(args.tax_dir).resolve()

    cfg: dict = {}
    if (plots / "all_rates.json").exists():
        cfg["rate_table"] = json.loads((plots / "all_rates.json").read_text())
    if (plots / "exp1_significance.json").exists():
        cfg["exp1_significance"] = json.loads((plots / "exp1_significance.json").read_text())
    if (tax / "chi_square.json").exists():
        cfg["exp3_chi_square"] = json.loads((tax / "chi_square.json").read_text())

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        tags=["pact", "paper", "corl2026", "aggregate"],
        config=cfg,
    )
    print(f"[push] wandb url: {run.url}")

    # Plots.
    for label, path in (
        ("exp1_bar", plots / "exp1_bar.png"),
        ("exp2_bar", plots / "exp2_bar.png"),
        ("exp3_failure_taxonomy", tax / "failure_taxonomy.png"),
    ):
        if path.exists():
            run.log({label: wandb.Image(str(path))})
            print(f"[push] uploaded {label}")

    # Optional epoch sweep.
    if args.epoch_sweep_dir:
        eps = Path(args.epoch_sweep_dir).resolve()
        if (eps / "epoch_sweep.png").exists():
            run.log({"epoch_sweep_plot": wandb.Image(str(eps / "epoch_sweep.png"))})
        if (eps / "best_epoch.json").exists():
            run.config.update({"best_epoch": json.loads((eps / "best_epoch.json").read_text())})

    # Headline summary numbers.
    if "rate_table" in cfg:
        for entry in cfg["rate_table"].get("exp1", []):
            short = entry["name"].replace(" ", "_").replace("(", "").replace(")", "")
            run.summary[f"exp1/{short}/rate"] = entry["rate"]
            run.summary[f"exp1/{short}/n_succ"] = entry["successes"]
            run.summary[f"exp1/{short}/n_tot"] = entry["total"]
        for entry in cfg["rate_table"].get("exp2", []):
            short = entry["name"].replace(" ", "_").replace("(", "").replace(")", "")
            run.summary[f"exp2/{short}/rate"] = entry["rate"]
            run.summary[f"exp2/{short}/n_succ"] = entry["successes"]
            run.summary[f"exp2/{short}/n_tot"] = entry["total"]

    if "exp3_chi_square" in cfg:
        run.summary["exp3/chi2"] = cfg["exp3_chi_square"]["chi2"]
        run.summary["exp3/p"] = cfg["exp3_chi_square"]["p"]
        run.summary["exp3/cramers_v"] = cfg["exp3_chi_square"]["cramers_v"]

    # Bundle artifact.
    art = wandb.Artifact(name=f"pact_paper_aggregate_{run.id}", type="paper-aggregate")
    for p in plots.iterdir():
        if p.is_file() and p.suffix in (".png", ".json"):
            art.add_file(str(p), name=f"plots/{p.name}")
    for p in tax.iterdir():
        if p.is_file() and p.suffix in (".png", ".json", ".csv"):
            art.add_file(str(p), name=f"taxonomy/{p.name}")
    run.log_artifact(art)

    run.finish()
    print("[push] done.")


if __name__ == "__main__":
    main()
