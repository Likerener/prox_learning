"""Push the CoRL paper analysis (n=50 conditions + Exp 1/3 + figures) to wandb.

Single run under project pact-paper-corl2026 that logs:
  * the four success-rate conditions (baseline, PACT, mask=zero, mask=mean)
  * Exp 3 blinded failure classification counts + agreement
  * key figures as wandb Images
  * the markdown writeups as wandb Artifacts

Run:
  /opt/conda/envs/mlspaces/bin/python scripts/push_paper_analysis_to_wandb.py \\
    --run_name paper_analysis_2026_05_27
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import wandb

REPO = Path("/home/jaydv/code/prox_learning")
PA = REPO / "eval_output/paper_analysis_2026_05_27"


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="pact-paper-corl2026")
    ap.add_argument("--run_name", default="paper_analysis_2026_05_27")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    base = load(REPO/"eval_output/act_house1_mug_random_v1_aggregate_n50/summary.json")
    pact = load(REPO/"eval_output/best_epoch_headline_n50/summary.json")
    mzero = load(REPO/"eval_output/best_epoch_mask_zero_n50/summary.json")
    mmean = load(REPO/"eval_output/mask_mean_n50/summary.json")
    blinded = load(PA/"failure_classification_blinded/summary.json") \
        or load(REPO/"eval_output/failure_classification_blinded/summary.json")

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                     job_type="paper-analysis", reinit=True)

    # ---- Success-rate conditions ----
    def rate(d): return None if d is None else d["pooled_success_rate"]
    def succ(d): return None if d is None else d["total_successes"]
    summary = {
        "baseline/rate": rate(base), "baseline/succ": succ(base),
        "pact/rate": rate(pact), "pact/succ": succ(pact),
        "mask_zero/rate": rate(mzero), "mask_zero/succ": succ(mzero),
        "mask_mean/rate": rate(mmean), "mask_mean/succ": succ(mmean),
    }
    if pact and mzero:
        summary["drop_zero_pp"] = (pact["pooled_success_rate"]-mzero["pooled_success_rate"])*100
    if pact and mmean:
        summary["drop_mean_pp"] = (pact["pooled_success_rate"]-mmean["pooled_success_rate"])*100
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    # Bar table of conditions
    cond_rows = []
    for name, d in [("vanilla ACT", base), ("PACT", pact),
                    ("PACT mask=zero", mzero), ("PACT mask=mean", mmean)]:
        if d:
            cond_rows.append([name, d["total_successes"], d["total_episodes"],
                              d["pooled_success_rate"],
                              d["wilson_95_ci"][0], d["wilson_95_ci"][1]])
    run.log({"success_conditions": wandb.Table(
        columns=["condition", "succ", "N", "rate", "ci_lo", "ci_hi"],
        data=cond_rows)})

    # ---- Exp 3 blinded ----
    if blinded:
        run.summary.update({
            "blinded/pregrasp_baseline_pct": blinded["pregrasp_share"]["baseline"],
            "blinded/pregrasp_pact_pct": blinded["pregrasp_share"]["pact"],
            "blinded/chi2": blinded["chi_square"]["chi2"],
            "blinded/chi2_p": blinded["chi_square"]["p"],
        })
        if blinded.get("agreement_vs_existing"):
            run.summary["blinded/agreement_rate"] = blinded["agreement_vs_existing"]["rate"]
        cats = blinded["chi_square"]["categories"]
        tbl = blinded["chi_square"]["table"]
        run.log({"blinded_failure_counts": wandb.Table(
            columns=["category", "baseline", "pact"],
            data=[[c, tbl[0][i], tbl[1][i]] for i, c in enumerate(cats)])})

    # ---- Figures ----
    figs = {
        "fig/paper_figure": PA/"paper_figure_v2.png",
        "fig/modality_weights": PA/"modality_weight_comparison.png",
        "fig/blinded_taxonomy": (PA/"failure_classification_blinded/failure_taxonomy_blinded.png"
                                 if (PA/"failure_classification_blinded").exists()
                                 else REPO/"eval_output/failure_classification_blinded/failure_taxonomy_blinded.png"),
        "fig/sensor_activity": PA/"sensor_usage_v2/sensor_phase_heatmap.png",
        "fig/succ_vs_fail": PA/"sensor_succ_vs_fail_v2/sensor_success_vs_fail_bars.png",
        "fig/attn_vs_activity": PA/"attn_vs_activity_v2/attn_vs_activity_per_phase.png",
    }
    log_imgs = {k: wandb.Image(str(v)) for k, v in figs.items() if Path(v).exists()}
    if log_imgs:
        run.log(log_imgs)

    # ---- Markdown artifacts ----
    art = wandb.Artifact("paper_analysis_docs", type="analysis")
    for md in [PA/"_paper_table_v2.md", PA/"README.md",
               (REPO/"eval_output/failure_classification_blinded/REPORT.md")]:
        if Path(md).exists():
            art.add_file(str(md))
    run.log_artifact(art)

    print("[wandb-push] logged conditions:", {k: v for k, v in summary.items() if v is not None})
    print("[wandb-push] figures:", list(log_imgs.keys()))
    print(f"[wandb-push] run url: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
