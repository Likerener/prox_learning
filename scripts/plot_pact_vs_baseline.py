"""Headline comparison plot: baseline ACT vs P+ACT success rate.

Reads two summary.json files plus their results.csv (for per-run dots),
draws bars with Wilson 95% CIs, overlays per-run successes, and annotates
the bars with the Fisher one-sided p-value and odds ratio.

Usage:
  /opt/conda/envs/mlspaces/bin/python scripts/plot_pact_vs_baseline.py \
      --baseline_root eval_output/act_house1_mug_random_v1_aggregate \
      --pact_root     eval_output/act_prox_mug_v1_aggregate \
      --out          eval_output/act_prox_mug_v1_aggregate/comparison_plot.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, barnard_exact


def _load(root: Path) -> tuple[dict, list[int]]:
    summary = json.loads((root / "summary.json").read_text())
    runs: list[int] = []
    with open(root / "results.csv") as f:
        for row in csv.DictReader(f):
            runs.append(int(row["success"]))
    return summary, runs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_root", required=True)
    p.add_argument("--pact_root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base_sum, base_runs = _load(Path(args.baseline_root))
    pact_sum, pact_runs = _load(Path(args.pact_root))

    rates = [base_sum["pooled_success_rate"], pact_sum["pooled_success_rate"]]
    succ = [base_sum["total_successes"], pact_sum["total_successes"]]
    total = [base_sum["total_episodes"], pact_sum["total_episodes"]]
    cis = [base_sum["wilson_95_ci"], pact_sum["wilson_95_ci"]]
    err_lo = [rates[i] - cis[i][0] for i in range(2)]
    err_hi = [cis[i][1] - rates[i] for i in range(2)]

    table = [[succ[1], total[1] - succ[1]], [succ[0], total[0] - succ[0]]]
    odds, p_fisher_one = fisher_exact(table, alternative="greater")
    _, p_fisher_two = fisher_exact(table, alternative="two-sided")
    p_barnard_one = barnard_exact(table, alternative="greater").pvalue

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    x = [0, 1]
    bars = ax.bar(
        x,
        [r * 100 for r in rates],
        yerr=[[e * 100 for e in err_lo], [e * 100 for e in err_hi]],
        capsize=6,
        width=0.55,
        color=["#888a91", "#3b7dd8"],
        edgecolor="black",
        linewidth=0.6,
    )

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + (err_hi[i] * 100) + 3.0,
            f"{succ[i]}/{total[i]}  ({rates[i]*100:.0f} %)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    delta_pp = (rates[1] - rates[0]) * 100
    ax.annotate(
        "",
        xy=(1, rates[1] * 100 + (err_hi[1] * 100) + 16),
        xytext=(0, rates[1] * 100 + (err_hi[1] * 100) + 16),
        arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1),
    )
    ax.text(
        0.5,
        rates[1] * 100 + (err_hi[1] * 100) + 18,
        f"Δ = +{delta_pp:.0f} pp",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Vanilla ACT", "P + ACT"],
        fontsize=10,
    )
    ax.set_ylabel("Pick-and-place success rate (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "P+ACT vs baseline ACT  —  FrankaSkinPickAndPlacePilotMediumConfig, house_1, n=10",
        fontsize=10.5,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    # footer = (
    #     f"Wilson 95 % CI shown.  "
    #     f"Fisher one-sided p = {p_fisher_one:.3f}  (two-sided p = {p_fisher_two:.3f}),  "
    #     f"Barnard one-sided p = {p_barnard_one:.3f},  odds ratio = {odds:.1f}."
    # )
    # fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=8, color="#444")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    print(
        f"[plot] baseline {succ[0]}/{total[0]} ({rates[0]*100:.0f} %)  "
        f"P+ACT {succ[1]}/{total[1]} ({rates[1]*100:.0f} %)  "
        f"Δ +{delta_pp:.0f} pp  Fisher 1-sided p={p_fisher_one:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
