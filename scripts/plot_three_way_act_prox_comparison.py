"""Side-by-side bar plot: VLM-only ACT vs ACT+prox K=1 vs ACT+prox K=6.

Reads pre-existing aggregate summary.json files (no rollouts needed) and plots
pooled success rate with 95% Wilson CIs. Designed to compare the three model
variants on the same in-distribution mug_house1_random eval.

Note: K=6 is at n=10 (very wide CI) while VLM-only and K=1 are at n=50.
The n=10 number is statistically inconclusive on its own — the CI overlaps
both the K=1 mean and a much-lower one.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

CONDITIONS = [
    # (label, summary_path, color)
    ("ACT (no prox)\nRGB + qpos",
     REPO / "eval_output" / "act_house1_mug_random_v1_aggregate_n50" / "summary.json",
     "#7f7f7f"),
    ("ACT + prox  K=1\nRGB + qpos + prox",
     REPO / "eval_output" / "act_prox_mug_v1_aggregate_n50" / "summary.json",
     "#1f77b4"),
    ("ACT + prox  K=6\nRGB + qpos + prox",
     REPO / "eval_output" / "act_prox_mug_v1_K6_aggregate" / "summary.json",
     "#2ca02c"),
]


def main() -> None:
    rows = []
    for label, path, color in CONDITIONS:
        if not path.exists():
            raise SystemExit(f"missing summary: {path}")
        with open(path) as f:
            s = json.load(f)
        rows.append({
            "label": label,
            "color": color,
            "pooled": s["pooled_success_rate"],
            "ci_lo": s["wilson_95_ci"][0],
            "ci_hi": s["wilson_95_ci"][1],
            "n_eps": s["total_episodes"],
            "succ": s["total_successes"],
        })

    xs = np.arange(len(rows))
    means = np.array([r["pooled"] for r in rows])
    lo = np.array([r["ci_lo"] for r in rows])
    hi = np.array([r["ci_hi"] for r in rows])
    errs_low = means - lo
    errs_high = hi - means

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(
        xs, means,
        yerr=[errs_low, errs_high],
        color=[r["color"] for r in rows],
        edgecolor="black", linewidth=0.8,
        capsize=10, error_kw={"linewidth": 1.4, "ecolor": "black"},
    )

    for x, r in zip(xs, rows):
        ax.text(
            x, r["pooled"] + 0.025,
            f"{r['succ']}/{r['n_eps']}  =  {r['pooled']:.0%}\n"
            f"95% CI [{r['ci_lo']:.0%}, {r['ci_hi']:.0%}]",
            ha="center", va="bottom", fontsize=9.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_ylabel("Pooled success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "ACT on house1 mug pick-and-place — in-distribution eval\n"
        "(seed=2026, samples_per_house=1, randomize_lighting=True, "
        "z-offset uniform ±U(0,1))",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Caveat for the noisy K=6
    ax.text(
        0.5, -0.18,
        "Caveat: K=6 is at n=10 (Wilson CI covers 17-69%); K=1 and no-prox are at n=50.",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=9, color="#555",
    )

    fig.tight_layout()
    out_path = REPO / "eval_output" / "three_way_act_prox_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    for r in rows:
        print(f"  {r['label'].replace(chr(10),' '):>40s}  n={r['n_eps']:>2d}  "
              f"{r['succ']:>2d}/{r['n_eps']:>2d} = {r['pooled']:.2%}  "
              f"CI [{r['ci_lo']:.2%}, {r['ci_hi']:.2%}]")


if __name__ == "__main__":
    main()
