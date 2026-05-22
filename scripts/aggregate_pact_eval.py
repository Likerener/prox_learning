"""Aggregate per-rollout eval logs into summary.json / results.csv matching
`eval_output/act_house1_mug_random_v1_aggregate/` layout.

Reads:
  <root>/eval_log_run_NN.txt   (one per rollout; each ends with "[act-prox-eval] success M/N")
  <root>/run_NN/running_log.log (molmospaces log, for elapsed time)

Writes:
  <root>/summary.json
  <root>/results.csv

Usage:
  /opt/conda/envs/mlspaces/bin/python scripts/aggregate_pact_eval.py \
      --root eval_output/act_prox_mug_v1_aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


_SUCCESS_LINE = re.compile(r"\[act-prox-eval\]\s+success\s+(\d+)/(\d+)")
_GENERIC_SUCCESS_LINE = re.compile(r"\[act-eval\]\s+success\s+(\d+)/(\d+)")


def _wilson_95_ci(successes: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% normal-approx z
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _parse_eval_log(log_path: Path) -> Dict:
    out: Dict = {"success": None, "total": None, "returncode": 0, "status": "ok"}
    if not log_path.exists():
        out["status"] = "missing"
        return out
    text = log_path.read_text(errors="ignore")
    m = _SUCCESS_LINE.search(text)
    if m is None:
        m = _GENERIC_SUCCESS_LINE.search(text)
    if m is None:
        out["status"] = "no_success_line"
        return out
    out["success"] = int(m.group(1))
    out["total"] = int(m.group(2))
    return out


def _parse_elapsed(running_log: Path) -> float:
    """Read the molmospaces running_log to estimate elapsed seconds.

    Each line is timestamped `MM/DD HH:MM:SS`. We take last - first.
    """
    if not running_log.exists():
        return 0.0
    ts_pat = re.compile(r"^(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})")
    first = last = None
    for line in running_log.read_text(errors="ignore").splitlines():
        m = ts_pat.match(line)
        if m is None:
            continue
        secs = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
        if first is None:
            first = secs
        last = secs
    if first is None or last is None:
        return 0.0
    if last < first:                          # crossed midnight
        last += 86400
    return float(last - first)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Aggregate root (parent of run_NN/ dirs).")
    p.add_argument("--baseline_summary", default=None,
                   help="Optional path to a baseline summary.json for a one-line headline.")
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[agg] {root} not a directory")
        return 1

    rows: List[Dict] = []
    for run_dir in sorted(root.glob("run_*")):
        idx = int(run_dir.name.split("_")[1])
        log_path = root / f"eval_log_run_{idx:02d}.txt"
        result = _parse_eval_log(log_path)
        elapsed = _parse_elapsed(run_dir / "running_log.log")
        success = result["success"] or 0
        total = result["total"] or 0
        rows.append({
            "run_idx": idx,
            "success": success,
            "total": total,
            "success_rate": (success / total) if total else 0.0,
            "returncode": result["returncode"],
            "status": result["status"],
            "elapsed_s": elapsed,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
        })

    if not rows:
        print(f"[agg] no run_*/ subdirs under {root}")
        return 1

    csv_path = root / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[agg] wrote {csv_path}")

    successes = sum(r["success"] for r in rows)
    total = sum(r["total"] for r in rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    per_run = [r["success_rate"] for r in rows if r["status"] == "ok"]
    ci_lo, ci_hi = _wilson_95_ci(successes, total)
    mean_rate = (sum(per_run) / len(per_run)) if per_run else 0.0
    std_rate = math.sqrt(sum((x - mean_rate) ** 2 for x in per_run) / len(per_run)) if per_run else 0.0
    summary = {
        "n_runs": len(rows),
        "n_runs_ok": ok,
        "total_episodes": total,
        "total_successes": successes,
        "pooled_success_rate": (successes / total) if total else 0.0,
        "wilson_95_ci": [ci_lo, ci_hi],
        "per_run_rate_mean": mean_rate,
        "per_run_rate_std": std_rate,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[agg] wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    if args.baseline_summary:
        try:
            with open(args.baseline_summary) as f:
                base = json.load(f)
            lo_b, hi_b = base["wilson_95_ci"]
            print()
            print(f"=== HEADLINE COMPARISON ===")
            print(f"  baseline: {base['total_successes']}/{base['total_episodes']} "
                  f"= {base['pooled_success_rate']*100:.1f}%  "
                  f"CI95=[{lo_b*100:.1f}%, {hi_b*100:.1f}%]")
            print(f"  P+ACT   : {successes}/{total} "
                  f"= {summary['pooled_success_rate']*100:.1f}%  "
                  f"CI95=[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
            delta_pp = (summary['pooled_success_rate'] - base['pooled_success_rate']) * 100
            print(f"  Δ       : {delta_pp:+.1f} pp")
        except (KeyError, FileNotFoundError) as e:
            print(f"[agg] baseline-summary read failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
