"""Aggregate plotting for Exp 1 (full mask) and Exp 2 (phase mask).

Loads `summary.json` from each per-condition output dir and produces:

- exp1_bar.png         — bars: baseline (vanilla ACT) | P+ACT none | P+ACT mean | P+ACT zero
- exp2_bar.png         — bars per phase, with horizontal lines for ceiling (mask=none)
                         and full-mask (mask=zero, phase=none).
- exp1_significance.json
- exp2_significance.json

Statistical test: Fisher 2-sided for Exp 1 head-to-heads;
chi-squared 2x5 across phases for Exp 2.

Run:
    /opt/conda/envs/mlspaces/bin/python scripts/plot_mask_experiments.py \\
        --baseline_dir eval_output/act_house1_mug_random_v1_aggregate_n50 \\
        --pact_none_dir eval_output/act_prox_mug_v1_aggregate_n50 \\
        --exp1_root eval_output \\
        --exp2_root eval_output \\
        --n 50 \\
        --output_dir eval_output/exp_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(p: Path) -> dict:
    if not p.exists():
        return {"total_episodes": 0, "total_successes": 0,
                "pooled_success_rate": 0.0,
                "wilson_95_ci": [0.0, 0.0], "missing": True,
                "path": str(p)}
    with open(p, "r") as f:
        s = json.load(f)
    s["missing"] = False
    s["path"] = str(p)
    return s


def fisher_two_sided(s_a: int, n_a: int, s_b: int, n_b: int) -> float:
    """2x2 Fisher exact (two-sided) using scipy if available."""
    try:
        from scipy.stats import fisher_exact
        table = [[s_a, n_a - s_a], [s_b, n_b - s_b]]
        _, p = fisher_exact(table, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def wilson_ci(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = s / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bar_plot(ax, names: list[str], rates: list[float], cis: list[tuple[float, float]],
             counts: list[tuple[int, int]], title: str, color_map=None) -> None:
    x = np.arange(len(names))
    err_lo = np.array([r - lo for r, (lo, hi) in zip(rates, cis)])
    err_hi = np.array([hi - r for r, (lo, hi) in zip(rates, cis)])
    yerr = np.vstack([err_lo, err_hi])
    colors = color_map(names) if color_map else None
    bars = ax.bar(x, rates, yerr=yerr, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for i, (r, (s, n)) in enumerate(zip(rates, counts)):
        ax.text(x[i], r + 0.04, f"{int(s)}/{int(n)}\n({r:.0%})",
                ha="center", va="bottom", fontsize=9)


def exp1_color(names):
    cmap = {"vanilla ACT": "#5e5e5e", "P+ACT (none)": "#1f77b4",
            "P+ACT (mean)": "#9467bd", "P+ACT (zero)": "#d62728"}
    return [cmap.get(n, "#888") for n in names]


def exp2_color(names):
    cmap = {"none (ceiling)": "#1f77b4", "approach": "#2ca02c", "pregrasp": "#bcbd22",
            "grasp_lift": "#ff7f0e", "transit": "#8c564b", "place": "#d62728",
            "all (zero)": "#444444"}
    return [cmap.get(n, "#888") for n in names]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", required=True,
                   help="vanilla ACT aggregate n=50")
    p.add_argument("--pact_none_dir", required=True,
                   help="P+ACT aggregate n=50 (no mask)")
    p.add_argument("--exp1_root", default="eval_output",
                   help="contains exp1_mask_zero_n${N}/ and exp1_mask_mean_n${N}/")
    p.add_argument("--exp2_root", default="eval_output",
                   help="contains exp2_mask_<phase>_n${N}/")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- Exp 1 ----
    base = load_summary(Path(args.baseline_dir) / "summary.json")
    pa_none = load_summary(Path(args.pact_none_dir) / "summary.json")
    pa_zero = load_summary(Path(args.exp1_root) / f"exp1_mask_zero_n{args.n}" / "summary.json")
    pa_mean = load_summary(Path(args.exp1_root) / f"exp1_mask_mean_n{args.n}" / "summary.json")

    names = ["vanilla ACT", "P+ACT (none)", "P+ACT (mean)", "P+ACT (zero)"]
    summaries = [base, pa_none, pa_mean, pa_zero]
    counts = [(s["total_successes"], s["total_episodes"]) for s in summaries]
    rates = [s["pooled_success_rate"] for s in summaries]
    cis = [tuple(s["wilson_95_ci"]) for s in summaries]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bar_plot(ax, names, rates, cis, counts,
             title=f"Exp 1: proximity-mask ablation (n={args.n}/condition)",
             color_map=exp1_color)
    # Annotate Fisher p-values vs P+ACT (none).
    if pa_none["total_episodes"] > 0:
        sn, nn = pa_none["total_successes"], pa_none["total_episodes"]
        ps = {
            "P+ACT (zero) vs (none)": fisher_two_sided(
                pa_zero["total_successes"], pa_zero["total_episodes"], sn, nn)
                if not pa_zero.get("missing") else None,
            "P+ACT (mean) vs (none)": fisher_two_sided(
                pa_mean["total_successes"], pa_mean["total_episodes"], sn, nn)
                if not pa_mean.get("missing") else None,
            "P+ACT (zero) vs vanilla": fisher_two_sided(
                pa_zero["total_successes"], pa_zero["total_episodes"],
                base["total_successes"], base["total_episodes"])
                if not pa_zero.get("missing") else None,
        }
        annot = "\n".join(f"{k}: p={v:.3g}" if v is not None and not np.isnan(v)
                          else f"{k}: missing" for k, v in ps.items())
        ax.text(0.02, 0.97, annot, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#aaa"))
        with open(out / "exp1_significance.json", "w") as f:
            json.dump({
                "fisher_p_values": {k: (None if v is None or np.isnan(v) else v)
                                    for k, v in ps.items()},
                "counts": {n: c for n, c in zip(names, counts)},
            }, f, indent=2)
    fig.tight_layout()
    fig.savefig(out / "exp1_bar.png", dpi=140)
    plt.close(fig)
    print(f"[plot] exp1_bar.png saved")

    # ---- Exp 2 ----
    phases = ["approach", "pregrasp", "grasp_lift", "transit", "place"]
    phase_summaries = [
        load_summary(Path(args.exp2_root) / f"exp2_mask_{ph}_n{args.n}" / "summary.json")
        for ph in phases
    ]

    names2 = ["none (ceiling)"] + phases + ["all (zero)"]
    summaries2 = [pa_none] + phase_summaries + [pa_zero]
    counts2 = [(s["total_successes"], s["total_episodes"]) for s in summaries2]
    rates2 = [s["pooled_success_rate"] for s in summaries2]
    cis2 = [tuple(s["wilson_95_ci"]) for s in summaries2]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_plot(ax, names2, rates2, cis2, counts2,
             title=f"Exp 2: phase-localised proximity mask (n={args.n}/condition)",
             color_map=exp2_color)

    if pa_none["total_episodes"] > 0:
        ax.axhline(pa_none["pooled_success_rate"], color="#1f77b4",
                   linestyle="--", linewidth=1, alpha=0.6, label="ceiling (no mask)")
    if pa_zero.get("missing") is False and pa_zero["total_episodes"] > 0:
        ax.axhline(pa_zero["pooled_success_rate"], color="#d62728",
                   linestyle="--", linewidth=1, alpha=0.6, label="all-masked (zero)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "exp2_bar.png", dpi=140)
    plt.close(fig)
    print(f"[plot] exp2_bar.png saved")

    # Save the raw rate table for paper writing.
    with open(out / "all_rates.json", "w") as f:
        json.dump({
            "exp1": [
                {"name": n, "successes": c[0], "total": c[1],
                 "rate": r, "wilson_95_ci": list(ci),
                 "path": s["path"], "missing": s.get("missing", False)}
                for n, c, r, ci, s in zip(names, counts, rates, cis, summaries)
            ],
            "exp2": [
                {"name": n, "successes": c[0], "total": c[1],
                 "rate": r, "wilson_95_ci": list(ci),
                 "path": s["path"], "missing": s.get("missing", False)}
                for n, c, r, ci, s in zip(names2, counts2, rates2, cis2, summaries2)
            ],
        }, f, indent=2)

    print(f"[plot] outputs in {out}")


if __name__ == "__main__":
    main()
