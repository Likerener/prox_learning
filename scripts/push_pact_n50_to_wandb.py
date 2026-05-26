"""One-shot wandb upload: final n=50 P+ACT vs Vanilla ACT headline numbers.

Logs to a single new run in `act-pla-house1-eval`:
  * summary scalars: per-arm succ/n/rate, Δ, Wilson + bootstrap CIs, Newcombe CI,
                     z-test stats, Fisher exact, bootstrap p-value
  * artifacts: comparison_plot.png, both summary.json, significance.json,
               both results.csv
  * a summary table (baseline vs P+ACT vs Δ) for quick scanning in the UI

Usage:
  /opt/conda/envs/mlspaces/bin/python scripts/push_pact_n50_to_wandb.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_root", default=str(REPO / "eval_output/act_house1_mug_random_v1_aggregate_n50"))
    p.add_argument("--pact_root",     default=str(REPO / "eval_output/act_prox_mug_v1_aggregate_n50"))
    p.add_argument("--sig_path",      default=str(REPO / "eval_output/act_prox_mug_v1_aggregate_n50/significance.json"))
    p.add_argument("--plot_path",     default=str(REPO / "eval_output/act_prox_mug_v1_aggregate_n50/comparison_plot.png"))
    p.add_argument("--project",       default="act-pla-house1-eval")
    p.add_argument("--entity",        default=None)
    p.add_argument("--run_name",      default=f"pact_vs_baseline_n50_{int(time.time())}")
    args = p.parse_args()

    base = json.loads((Path(args.baseline_root) / "summary.json").read_text())
    pact = json.loads((Path(args.pact_root)     / "summary.json").read_text())
    sig  = json.loads(Path(args.sig_path).read_text())

    config = {
        "n_per_arm":          base["total_episodes"],
        "baseline_ckpt":      "submodules/act/ckpts/act_house1_mug_random_v1/policy_best.ckpt",
        "pact_ckpt":          "runs/act_prox_mug_v1/policy_best.ckpt",
        "task_config":        "FrankaSkinPickAndPlacePilotMediumConfig",
        "house":              "house_1",
        "object":             "mug",
        "samples_per_house":  1,
        "task_horizon":       300,
        "chunk_size":         100,
        "temporal_ensemble":  True,
    }

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        tags=["pact", "n50", "headline", "vs-baseline"],
        config=config,
        notes="Final n=50 head-to-head: Vanilla ACT vs P+ACT. Same task config / house / horizon. "
              "Significance via Fisher exact (primary), z-test (asymptotic), and 20k bootstrap.",
        job_type="aggregate-eval",
    )

    # Summary scalars - one screen of numbers
    b_succ, b_n, b_rate = base["total_successes"], base["total_episodes"], base["pooled_success_rate"]
    p_succ, p_n, p_rate = pact["total_successes"], pact["total_episodes"], pact["pooled_success_rate"]
    delta_pp = 100.0 * (p_rate - b_rate)

    for k, v in {
        # per-arm
        "baseline/successes":      b_succ,
        "baseline/n":              b_n,
        "baseline/success_rate":   b_rate,
        "baseline/wilson_95_lo":   base["wilson_95_ci"][0],
        "baseline/wilson_95_hi":   base["wilson_95_ci"][1],
        "baseline/bootstrap_95_lo": sig["baseline"]["bootstrap95"][0],
        "baseline/bootstrap_95_hi": sig["baseline"]["bootstrap95"][1],
        "pact/successes":          p_succ,
        "pact/n":                  p_n,
        "pact/success_rate":       p_rate,
        "pact/wilson_95_lo":       pact["wilson_95_ci"][0],
        "pact/wilson_95_hi":       pact["wilson_95_ci"][1],
        "pact/bootstrap_95_lo":    sig["pact"]["bootstrap95"][0],
        "pact/bootstrap_95_hi":    sig["pact"]["bootstrap95"][1],
        # delta
        "delta/pp":                delta_pp,
        "delta/newcombe_95_lo_pp": 100.0 * sig["newcombe_95_for_diff"][0],
        "delta/newcombe_95_hi_pp": 100.0 * sig["newcombe_95_for_diff"][1],
        "delta/bootstrap_95_lo_pp": 100.0 * sig["bootstrap_95_for_diff"][0],
        "delta/bootstrap_95_hi_pp": 100.0 * sig["bootstrap_95_for_diff"][1],
        # significance tests
        "test/fisher_or":             sig["fisher_exact"]["odds_ratio"],
        "test/fisher_p_one_sided":    sig["fisher_exact"]["p_one_sided_greater"],
        "test/fisher_p_two_sided":    sig["fisher_exact"]["p_two_sided"],
        "test/z_pooled":              sig["two_prop_z"]["z_pooled"],
        "test/z_pooled_p_one_sided":  sig["two_prop_z"]["p_pooled_one_sided_greater"],
        "test/z_pooled_p_two_sided":  sig["two_prop_z"]["p_pooled_two_sided"],
        "test/z_unpooled":            sig["two_prop_z"]["z_unpooled"],
        "test/z_unp_p_one_sided":     sig["two_prop_z"]["p_unpooled_one_sided_greater"],
        "test/z_unp_p_two_sided":     sig["two_prop_z"]["p_unpooled_two_sided"],
        "test/bootstrap_p_one_sided": sig["bootstrap_p_one_sided_greater"],
        "test/bootstrap_B":           sig["bootstrap_B"],
        # flags
        "headline/fisher_one_sig_05": int(sig["fisher_exact"]["p_one_sided_greater"]   < 0.05),
        "headline/fisher_two_sig_05": int(sig["fisher_exact"]["p_two_sided"]           < 0.05),
        "headline/z_pool_two_sig_05": int(sig["two_prop_z"]["p_pooled_two_sided"]      < 0.05),
        "headline/z_unp_two_sig_05":  int(sig["two_prop_z"]["p_unpooled_two_sided"]    < 0.05),
        "headline/boot_one_sig_05":   int(sig["bootstrap_p_one_sided_greater"]         < 0.05),
    }.items():
        run.summary[k] = v

    # Pretty table for the run page
    table = wandb.Table(
        columns=["arm", "successes", "n", "rate_pct", "wilson_lo_pct", "wilson_hi_pct"],
        data=[
            ["Vanilla ACT", b_succ, b_n, 100*b_rate, 100*base["wilson_95_ci"][0], 100*base["wilson_95_ci"][1]],
            ["P + ACT",     p_succ, p_n, 100*p_rate, 100*pact["wilson_95_ci"][0], 100*pact["wilson_95_ci"][1]],
        ],
    )
    run.log({"headline_table": table})

    # Plot
    plot = Path(args.plot_path)
    if plot.exists():
        run.log({"comparison_plot": wandb.Image(str(plot), caption=f"P+ACT vs Vanilla ACT  —  n={b_n} per arm")})
    else:
        print(f"[wandb] WARNING: plot not found at {plot}; skipping image")

    # Artifact: raw files (rename to avoid basename collisions across arms)
    art = wandb.Artifact(name=f"pact_n50_aggregates_{run.id}", type="eval-aggregate")
    file_specs = [
        (Path(args.baseline_root) / "summary.json", "baseline__summary.json"),
        (Path(args.baseline_root) / "results.csv",  "baseline__results.csv"),
        (Path(args.pact_root)     / "summary.json", "pact__summary.json"),
        (Path(args.pact_root)     / "results.csv",  "pact__results.csv"),
        (Path(args.sig_path),                       "significance.json"),
        (plot,                                      "comparison_plot.png"),
    ]
    for path, name in file_specs:
        if path.exists():
            art.add_file(str(path), name=name)
        else:
            print(f"[wandb] skipping missing {path}")
    run.log_artifact(art)

    run.finish()
    print(f"[wandb] logged: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
