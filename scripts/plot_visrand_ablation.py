"""Multi-condition ablation bar chart: Vanilla ACT vs P+ACT under three
visual-randomization conditions.

Reads:
  * lighting-only  (existing n=50 aggregates from prior session)
  * +textures      (phase 1b+2 aggregate)
  * +textures_all  (phase 1a aggregate)

Writes:
  * eval_output/visrand_ablation_summary.png  — grouped bar with Wilson 95% CIs
  * eval_output/visrand_ablation_summary.json — machine-readable matrix
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact, norm


CONDITIONS = [
    {
        "label": "Lighting only",
        "baseline": "eval_output/act_house1_mug_random_v1_aggregate_n50",
        "pact":     "eval_output/act_prox_mug_v1_aggregate_n50",
    },
    {
        "label": "+ Textures",
        "baseline": "eval_output/act_house1_mug_visrand_mod_n50",
        "pact":     "eval_output/act_prox_mug_visrand_mod_n50",
    },
    {
        "label": "+ Textures all",
        "baseline": "eval_output/act_house1_mug_visrand_severe_n10",
        "pact":     "eval_output/act_prox_mug_visrand_severe_n10",
    },
]


def wilson_95(s: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(0.975)
    p = s / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(root: Path) -> dict:
    s = json.loads((root / "summary.json").read_text())
    return {"succ": s["total_successes"], "n": s["total_episodes"], "rate": s["pooled_success_rate"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out_png",  default="eval_output/visrand_ablation_summary.png")
    p.add_argument("--out_json", default="eval_output/visrand_ablation_summary.json")
    args = p.parse_args()

    rows = []
    matrix = []
    for cond in CONDITIONS:
        b_root = Path(cond["baseline"]); p_root = Path(cond["pact"])
        if not (b_root / "summary.json").exists() or not (p_root / "summary.json").exists():
            print(f"[plot] missing summary for {cond['label']} — skipping")
            continue
        b = load(b_root); pa = load(p_root)
        b_lo, b_hi = wilson_95(b["succ"], b["n"])
        p_lo, p_hi = wilson_95(pa["succ"], pa["n"])
        # Fisher one + two sided for P+ACT vs baseline
        table = [[pa["succ"], pa["n"] - pa["succ"]], [b["succ"], b["n"] - b["succ"]]]
        odds, p1 = fisher_exact(table, alternative="greater")
        _, p2 = fisher_exact(table, alternative="two-sided")
        rows.append({"label": cond["label"], "baseline": b, "pact": pa,
                     "baseline_wilson": [b_lo, b_hi], "pact_wilson": [p_lo, p_hi],
                     "delta_pp": 100*(pa["rate"] - b["rate"]),
                     "fisher_or": odds, "fisher_p_one": p1, "fisher_p_two": p2})
        matrix.append({"condition": cond["label"],
                       "baseline": {"succ": b["succ"], "n": b["n"], "rate": b["rate"],
                                    "wilson_95": [b_lo, b_hi]},
                       "pact":     {"succ": pa["succ"], "n": pa["n"], "rate": pa["rate"],
                                    "wilson_95": [p_lo, p_hi]},
                       "delta_pp": 100*(pa["rate"] - b["rate"]),
                       "fisher": {"odds_ratio": odds, "p_one_sided": p1, "p_two_sided": p2}})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(matrix, indent=2))

    # Plot
    n_cond = len(rows)
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * n_cond + 2.6), 5.0), dpi=140)
    x = np.arange(n_cond)
    w = 0.36
    b_rates = [100 * r["baseline"]["rate"] for r in rows]
    p_rates = [100 * r["pact"]["rate"]     for r in rows]
    b_err_lo = [b_rates[i] - 100 * r["baseline_wilson"][0] for i, r in enumerate(rows)]
    b_err_hi = [100 * r["baseline_wilson"][1] - b_rates[i] for i, r in enumerate(rows)]
    p_err_lo = [p_rates[i] - 100 * r["pact_wilson"][0]     for i, r in enumerate(rows)]
    p_err_hi = [100 * r["pact_wilson"][1] - p_rates[i]     for i, r in enumerate(rows)]

    ax.bar(x - w/2, b_rates, w, yerr=[b_err_lo, b_err_hi], capsize=5,
           color="#888a91", edgecolor="black", linewidth=0.5, label="Vanilla ACT")
    ax.bar(x + w/2, p_rates, w, yerr=[p_err_lo, p_err_hi], capsize=5,
           color="#3b7dd8", edgecolor="black", linewidth=0.5, label="P + ACT")

    for i, r in enumerate(rows):
        ax.text(x[i] - w/2, b_rates[i] + b_err_hi[i] + 2.5,
                f"{r['baseline']['succ']}/{r['baseline']['n']}\n({b_rates[i]:.0f}%)",
                ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w/2, p_rates[i] + p_err_hi[i] + 2.5,
                f"{r['pact']['succ']}/{r['pact']['n']}\n({p_rates[i]:.0f}%)",
                ha="center", va="bottom", fontsize=9)
        dpp = r["delta_pp"]
        color = "#3b7dd8" if dpp > 0 else ("#888a91" if dpp == 0 else "#d62728")
        ax.text(x[i], max(b_rates[i] + b_err_hi[i], p_rates[i] + p_err_hi[i]) + 14,
                f"Δ {dpp:+.0f} pp\nFisher 2-sided p={r['fisher_p_two']:.3f}",
                ha="center", va="bottom", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=11)
    ax.set_ylabel("Pick-and-place success rate (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Visual-randomization ablation  —  Vanilla ACT vs P+ACT",
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"[plot] wrote {args.out_png}")
    print(f"[plot] wrote {args.out_json}")
    for r in rows:
        print(f"  {r['label']:20s}  baseline {r['baseline']['succ']}/{r['baseline']['n']} ({100*r['baseline']['rate']:.1f}%)  "
              f"P+ACT {r['pact']['succ']}/{r['pact']['n']} ({100*r['pact']['rate']:.1f}%)  "
              f"Δ {r['delta_pp']:+.0f}pp  Fisher 2-sided p={r['fisher_p_two']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
