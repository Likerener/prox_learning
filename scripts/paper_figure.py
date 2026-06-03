"""Single paper-ready figure combining Exp 1 (proximity ablation),
Exp 2 (phase-localized mask), and Exp 3 (failure taxonomy).

Layout: 3-panel horizontal — Exp 1 | Exp 2 | Exp 3.

Run AFTER:
  scripts/plot_mask_experiments.py    (writes exp1/exp2 data)
  scripts/failure_taxonomy.py         (writes exp3 data)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plots_dir", required=True,
                   help="from plot_mask_experiments.py (has all_rates.json)")
    p.add_argument("--tax_dir", required=True,
                   help="from failure_taxonomy.py (has chi_square.json)")
    p.add_argument("--out", required=True,
                   help="output png path")
    p.add_argument("--n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plots = Path(args.plots_dir)
    tax = Path(args.tax_dir)
    all_rates = json.loads((plots / "all_rates.json").read_text())
    chi = json.loads((tax / "chi_square.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    # ---- Panel A: Exp 1 ----
    ax = axes[0]
    exp1 = all_rates["exp1"]
    names = [e["name"] for e in exp1]
    rates = [e["rate"] for e in exp1]
    cis = [e["wilson_95_ci"] for e in exp1]
    counts = [(e["successes"], e["total"]) for e in exp1]
    err_lo = np.array([r - lo for r, (lo, _) in zip(rates, cis)])
    err_hi = np.array([hi - r for r, (_, hi) in zip(rates, cis)])
    color = []
    for n in names:
        if "vanilla" in n: color.append("#5e5e5e")
        elif "(none)" in n: color.append("#1f77b4")
        elif "(mean)" in n: color.append("#9467bd")
        else: color.append("#d62728")
    x = np.arange(len(names))
    ax.bar(x, rates, yerr=np.vstack([err_lo, err_hi]), capsize=4,
           color=color, edgecolor="black", linewidth=0.6)
    for i, ((s, n), r) in enumerate(zip(counts, rates)):
        ax.text(x[i], r + 0.03, f"{int(s)}/{int(n)}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"(A) Exp 1: proximity ablation (n={args.n} each)")
    ax.grid(True, axis="y", alpha=0.25)

    # ---- Panel B: Exp 2 ----
    ax = axes[1]
    exp2 = all_rates["exp2"]
    names2 = [e["name"] for e in exp2]
    rates2 = [e["rate"] for e in exp2]
    cis2 = [e["wilson_95_ci"] for e in exp2]
    counts2 = [(e["successes"], e["total"]) for e in exp2]
    err_lo2 = np.array([r - lo for r, (lo, _) in zip(rates2, cis2)])
    err_hi2 = np.array([hi - r for r, (_, hi) in zip(rates2, cis2)])
    color2 = []
    for n in names2:
        if "ceiling" in n: color2.append("#1f77b4")
        elif "all" in n:   color2.append("#d62728")
        else:              color2.append("#ff7f0e")
    x2 = np.arange(len(names2))
    ax.bar(x2, rates2, yerr=np.vstack([err_lo2, err_hi2]), capsize=4,
           color=color2, edgecolor="black", linewidth=0.6)
    for i, ((s, n), r) in enumerate(zip(counts2, rates2)):
        ax.text(x2[i], r + 0.03, f"{int(s)}/{int(n)}", ha="center", fontsize=9)
    if rates2:
        ceiling = next((r for n, r in zip(names2, rates2) if "ceiling" in n), None)
        floor = next((r for n, r in zip(names2, rates2) if "all" in n), None)
        if ceiling is not None:
            ax.axhline(ceiling, color="#1f77b4", linestyle="--", linewidth=1, alpha=0.6)
        if floor is not None:
            ax.axhline(floor, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x2)
    ax.set_xticklabels(names2, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"(B) Exp 2: phase-localised mask (n={args.n} each)")
    ax.grid(True, axis="y", alpha=0.25)

    # ---- Panel C: Exp 3 ----
    ax = axes[2]
    cats = chi["categories"]
    base_table = chi["table"][0]
    pact_table = chi["table"][1]
    width = 0.38
    xc = np.arange(len(cats))
    ax.bar(xc - 0.5 * width, base_table, width, color="#5e5e5e",
           edgecolor="black", linewidth=0.5, label=f"vanilla ACT (failures={sum(base_table)})")
    ax.bar(xc + 0.5 * width, pact_table, width, color="#1f77b4",
           edgecolor="black", linewidth=0.5, label=f"P+ACT (failures={sum(pact_table)})")
    for i, (b, p) in enumerate(zip(base_table, pact_table)):
        ax.text(xc[i] - 0.5 * width, b + 0.2, str(int(b)), ha="center", fontsize=9)
        ax.text(xc[i] + 0.5 * width, p + 0.2, str(int(p)), ha="center", fontsize=9)
    ax.set_xticks(xc)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("# failed rollouts")
    ax.set_title(f"(C) Exp 3: failure taxonomy  "
                 f"(χ²={chi['chi2']:.2f}, p={chi['p']:.3g}, V={chi['cramers_v']:.2f})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"P+ACT vs vanilla ACT — three-panel result (CoRL 2026)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"[paper-fig] saved {args.out}")


if __name__ == "__main__":
    main()
