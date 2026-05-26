"""Push the 3-condition visrand ablation summary to wandb as a single
"headline" run containing: summary scalars (one per condition × per arm),
the summary table, the multi-condition plot, the per-condition Wilson CIs,
and the writeup markdown as an artifact.
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
    p.add_argument("--matrix_json", default=str(REPO / "eval_output/visrand_ablation_summary.json"))
    p.add_argument("--plot_png",    default=str(REPO / "eval_output/visrand_ablation_summary.png"))
    p.add_argument("--writeup_md",  default=str(REPO / "eval_output/visrand_ablation_writeup.md"))
    p.add_argument("--project",     default="act-pla-house1-eval")
    p.add_argument("--run_name",    default=f"visrand_ablation_summary_{int(time.time())}")
    args = p.parse_args()

    matrix = json.loads(Path(args.matrix_json).read_text())

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        tags=["pact", "visrand", "ablation-summary", "headline"],
        notes="Multi-condition visual-randomization ablation: Vanilla ACT vs P+ACT under "
              "(a) lighting only [in-distribution n=50], (b) +textures [OOD-moderate n=50], "
              "(c) +textures_all [OOD-severe n=10]. Significant negative result at (b).",
        job_type="ablation-summary",
    )

    # Flat summary scalars per condition × per arm + delta + p-values
    for row in matrix:
        c = row["condition"].lower().replace(" ", "_").replace("+", "plus")
        run.summary[f"{c}/baseline/succ"] = row["baseline"]["succ"]
        run.summary[f"{c}/baseline/n"]    = row["baseline"]["n"]
        run.summary[f"{c}/baseline/rate"] = row["baseline"]["rate"]
        run.summary[f"{c}/baseline/wilson_lo"] = row["baseline"]["wilson_95"][0]
        run.summary[f"{c}/baseline/wilson_hi"] = row["baseline"]["wilson_95"][1]
        run.summary[f"{c}/pact/succ"]     = row["pact"]["succ"]
        run.summary[f"{c}/pact/n"]        = row["pact"]["n"]
        run.summary[f"{c}/pact/rate"]     = row["pact"]["rate"]
        run.summary[f"{c}/pact/wilson_lo"] = row["pact"]["wilson_95"][0]
        run.summary[f"{c}/pact/wilson_hi"] = row["pact"]["wilson_95"][1]
        run.summary[f"{c}/delta_pp"]      = row["delta_pp"]
        run.summary[f"{c}/fisher_p_two"]  = row["fisher"]["p_two_sided"]
        run.summary[f"{c}/sig_05"]        = int(row["fisher"]["p_two_sided"] < 0.05)

    # One nice table view
    table = wandb.Table(
        columns=["condition", "baseline succ/n", "baseline %", "P+ACT succ/n", "P+ACT %",
                 "Δ pp", "Fisher 2-sided p", "sig α=0.05"],
        data=[[
            r["condition"],
            f"{r['baseline']['succ']}/{r['baseline']['n']}",
            round(100 * r["baseline"]["rate"], 1),
            f"{r['pact']['succ']}/{r['pact']['n']}",
            round(100 * r["pact"]["rate"], 1),
            round(r["delta_pp"], 1),
            round(r["fisher"]["p_two_sided"], 4),
            "yes" if r["fisher"]["p_two_sided"] < 0.05 else "no",
        ] for r in matrix],
    )
    run.log({"ablation_matrix": table})

    if Path(args.plot_png).exists():
        run.log({"visrand_ablation_plot": wandb.Image(args.plot_png,
                 caption="Visrand ablation: Vanilla ACT vs P+ACT (3 conditions, Wilson 95% CIs)")})

    art = wandb.Artifact(name=f"visrand_ablation_bundle_{run.id}", type="ablation-summary")
    for path in [args.matrix_json, args.plot_png, args.writeup_md]:
        p = Path(path)
        if p.exists():
            art.add_file(str(p), name=p.name)
        else:
            print(f"[wandb] skipping missing {p}")
    # Bundle the per-condition raw aggregates too so reviewers can recompute
    for path_str in [
        "eval_output/act_house1_mug_random_v1_aggregate_n50/summary.json",
        "eval_output/act_house1_mug_random_v1_aggregate_n50/results.csv",
        "eval_output/act_prox_mug_v1_aggregate_n50/summary.json",
        "eval_output/act_prox_mug_v1_aggregate_n50/results.csv",
        "eval_output/act_house1_mug_visrand_mod_n50/summary.json",
        "eval_output/act_house1_mug_visrand_mod_n50/results.csv",
        "eval_output/act_prox_mug_visrand_mod_n50/summary.json",
        "eval_output/act_prox_mug_visrand_mod_n50/results.csv",
        "eval_output/act_prox_mug_visrand_mod_n50/significance.json",
        "eval_output/act_house1_mug_visrand_severe_n10/summary.json",
        "eval_output/act_house1_mug_visrand_severe_n10/results.csv",
        "eval_output/act_prox_mug_visrand_severe_n10/summary.json",
        "eval_output/act_prox_mug_visrand_severe_n10/results.csv",
    ]:
        p = REPO / path_str
        if p.exists():
            art.add_file(str(p), name=path_str.replace("/", "__"))
    run.log_artifact(art)

    run.finish()
    print(f"[wandb] summary run logged: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
