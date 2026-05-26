"""Statistical significance analysis: Vanilla ACT vs P+ACT.

Runs:
  * Two-proportion z-test (pooled and unpooled SE)
  * Fisher's exact test (one-sided 'greater' and two-sided)
  * Wilson 95 % CIs per arm
  * Newcombe (hybrid Wilson) 95 % CI for the difference in proportions
  * Non-parametric 95 % bootstrap CIs per arm and for the difference (B=20 000)

Reads the same summary.json + results.csv pair used by plot_pact_vs_baseline.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm


def _load(root: Path) -> list[int]:
    runs: list[int] = []
    with open(root / "results.csv") as f:
        for row in csv.DictReader(f):
            runs.append(int(row["success"]))
    return runs


def wilson_ci(succ: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1.0 - (1.0 - conf) / 2.0)
    p = succ / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (centre - half, centre + half)


def newcombe_diff_ci(s1: int, n1: int, s2: int, n2: int, conf: float = 0.95) -> tuple[float, float]:
    """Newcombe (1998) hybrid Wilson CI for p2 - p1."""
    l1, u1 = wilson_ci(s1, n1, conf)
    l2, u2 = wilson_ci(s2, n2, conf)
    p1, p2 = s1 / n1, s2 / n2
    diff = p2 - p1
    lo = diff - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = diff + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def two_prop_z(s1: int, n1: int, s2: int, n2: int) -> dict:
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    se_unp = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_pool = (p2 - p1) / se_pool if se_pool > 0 else float("nan")
    z_unp = (p2 - p1) / se_unp if se_unp > 0 else float("nan")
    return {
        "z_pooled": z_pool,
        "p_pooled_two_sided": 2.0 * (1.0 - norm.cdf(abs(z_pool))),
        "p_pooled_one_sided_greater": 1.0 - norm.cdf(z_pool),
        "z_unpooled": z_unp,
        "p_unpooled_two_sided": 2.0 * (1.0 - norm.cdf(abs(z_unp))),
        "p_unpooled_one_sided_greater": 1.0 - norm.cdf(z_unp),
        "se_pooled": se_pool,
        "se_unpooled": se_unp,
    }


def bootstrap_ci(arr: np.ndarray, B: int = 20_000, conf: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    n = len(arr)
    idx = rng.integers(0, n, size=(B, n))
    samples = arr[idx].mean(axis=1)
    alpha = (1.0 - conf) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi), samples


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, B: int = 20_000, conf: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """CI for mean(b) - mean(a) via independent resampling. Also returns one-sided bootstrap p (P[diff <= 0])."""
    rng = rng or np.random.default_rng(0)
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    alpha = (1.0 - conf) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    p_one_sided = float(np.mean(diffs <= 0.0))
    return float(lo), float(hi), p_one_sided


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_root", default="eval_output/act_house1_mug_random_v1_aggregate")
    p.add_argument("--pact_root", default="eval_output/act_prox_mug_v1_aggregate")
    p.add_argument("--bootstrap", type=int, default=20_000)
    p.add_argument("--out", default="eval_output/act_prox_mug_v1_aggregate/significance.json")
    args = p.parse_args()

    base = np.array(_load(Path(args.baseline_root)), dtype=float)
    pact = np.array(_load(Path(args.pact_root)), dtype=float)
    s1, n1 = int(base.sum()), len(base)
    s2, n2 = int(pact.sum()), len(pact)
    p1, p2 = s1 / n1, s2 / n2
    delta = p2 - p1

    wilson1 = wilson_ci(s1, n1)
    wilson2 = wilson_ci(s2, n2)
    newc = newcombe_diff_ci(s1, n1, s2, n2)
    z = two_prop_z(s1, n1, s2, n2)

    table = [[s2, n2 - s2], [s1, n1 - s1]]  # rows: P+ACT, baseline
    or_fisher, p_fisher_one = fisher_exact(table, alternative="greater")
    _, p_fisher_two = fisher_exact(table, alternative="two-sided")

    rng = np.random.default_rng(2026_05_22)
    boot1_lo, boot1_hi, _ = bootstrap_ci(base, args.bootstrap, rng=rng)
    boot2_lo, boot2_hi, _ = bootstrap_ci(pact, args.bootstrap, rng=rng)
    diff_lo, diff_hi, boot_p_one = bootstrap_diff_ci(base, pact, args.bootstrap, rng=rng)

    out = {
        "baseline": {"successes": s1, "n": n1, "rate": p1, "wilson95": list(wilson1), "bootstrap95": [boot1_lo, boot1_hi]},
        "pact":     {"successes": s2, "n": n2, "rate": p2, "wilson95": list(wilson2), "bootstrap95": [boot2_lo, boot2_hi]},
        "delta_pp": 100.0 * delta,
        "newcombe_95_for_diff": list(newc),
        "bootstrap_95_for_diff": [diff_lo, diff_hi],
        "bootstrap_p_one_sided_greater": boot_p_one,
        "two_prop_z": z,
        "fisher_exact": {
            "odds_ratio": or_fisher,
            "p_one_sided_greater": p_fisher_one,
            "p_two_sided": p_fisher_two,
        },
        "bootstrap_B": args.bootstrap,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    def sig(p_): return "yes" if p_ < 0.05 else "no"

    print("=" * 72)
    print(f"Baseline (Vanilla ACT) : {s1}/{n1} = {p1*100:5.1f} %")
    print(f"P + ACT                : {s2}/{n2} = {p2*100:5.1f} %")
    print(f"Δ                      : {delta*100:+.1f} pp")
    print()
    print("Per-arm 95 % CIs")
    print(f"  Baseline Wilson    : [{wilson1[0]*100:5.1f}, {wilson1[1]*100:5.1f}] %")
    print(f"  Baseline bootstrap : [{boot1_lo*100:5.1f}, {boot1_hi*100:5.1f}] %")
    print(f"  P+ACT    Wilson    : [{wilson2[0]*100:5.1f}, {wilson2[1]*100:5.1f}] %")
    print(f"  P+ACT    bootstrap : [{boot2_lo*100:5.1f}, {boot2_hi*100:5.1f}] %")
    print()
    print("95 % CI for the difference (P+ACT − baseline)")
    print(f"  Newcombe Wilson    : [{newc[0]*100:+.1f}, {newc[1]*100:+.1f}] pp")
    print(f"  Bootstrap          : [{diff_lo*100:+.1f}, {diff_hi*100:+.1f}] pp")
    print()
    print("Two-proportion z-test")
    print(f"  pooled  : z = {z['z_pooled']:+.3f}   one-sided p = {z['p_pooled_one_sided_greater']:.4f} ({sig(z['p_pooled_one_sided_greater'])})   two-sided p = {z['p_pooled_two_sided']:.4f} ({sig(z['p_pooled_two_sided'])})")
    print(f"  unpool. : z = {z['z_unpooled']:+.3f}   one-sided p = {z['p_unpooled_one_sided_greater']:.4f} ({sig(z['p_unpooled_one_sided_greater'])})   two-sided p = {z['p_unpooled_two_sided']:.4f} ({sig(z['p_unpooled_two_sided'])})")
    print()
    print("Fisher's exact test")
    print(f"  odds ratio = {or_fisher:.2f}")
    print(f"  one-sided p (P+ACT > baseline) = {p_fisher_one:.4f}  ({sig(p_fisher_one)})")
    print(f"  two-sided p                    = {p_fisher_two:.4f}  ({sig(p_fisher_two)})")
    print()
    print("Bootstrap test")
    print(f"  one-sided p (Δ ≤ 0)            = {boot_p_one:.4f}  ({sig(boot_p_one)})")
    print("=" * 72)
    print(f"[significance] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
