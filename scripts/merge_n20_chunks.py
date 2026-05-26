"""Merge two chunk dirs (each with run_NN/ subdirs + per-rollout logs) into a
single aggregate dir with a unified results.csv + summary.json that matches the
layout other scripts expect.

Chunk A contributes its rollouts as-is; chunk B's run_NN entries are renumbered
to start at (max index in A) + 1 to avoid collisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

_SUCCESS_PATTERNS = [
    re.compile(r"\[act-prox-eval\]\s+success\s+(\d+)/(\d+)"),
    re.compile(r"\[act-eval\]\s+success\s+(\d+)/(\d+)"),
]


def _parse_success(log_path: Path) -> tuple[int, int] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    for pat in _SUCCESS_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _wilson_95(s: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = s / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _gather_runs(chunk_dir: Path) -> list[tuple[int, Path, int, int]]:
    """Return [(orig_idx, run_dir, success, total)] for one chunk."""
    out: list[tuple[int, Path, int, int]] = []
    for run_dir in sorted(chunk_dir.glob("run_*")):
        idx = int(run_dir.name.split("_")[1])
        log = chunk_dir / f"eval_log_run_{idx:02d}.txt"
        st = _parse_success(log)
        if st is None:
            print(f"[merge] WARN: no success line in {log}, skipping")
            continue
        out.append((idx, run_dir, st[0], st[1]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_a", required=True)
    p.add_argument("--chunk_b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--copy", action="store_true",
                   help="Copy run subdirs into out; otherwise symlink.")
    args = p.parse_args()

    a_runs = _gather_runs(Path(args.chunk_a))
    b_runs = _gather_runs(Path(args.chunk_b))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    next_idx = 0
    rows: list[dict] = []
    for runs, chunk_dir in [(a_runs, Path(args.chunk_a)), (b_runs, Path(args.chunk_b))]:
        for orig_idx, run_dir, succ, total in runs:
            new_idx = next_idx
            new_name = f"run_{new_idx:02d}"
            dst = out / new_name
            if dst.exists():
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            if args.copy:
                shutil.copytree(run_dir, dst)
            else:
                dst.symlink_to(run_dir.resolve())
            # also copy the eval log
            src_log = chunk_dir / f"eval_log_run_{orig_idx:02d}.txt"
            if src_log.exists():
                shutil.copy(src_log, out / f"eval_log_run_{new_idx:02d}.txt")
            rows.append({
                "run_idx": new_idx,
                "success": succ,
                "total": total,
                "success_rate": (succ / total) if total else 0.0,
                "returncode": 0,
                "status": "ok",
                "elapsed_s": 0.0,
                "run_dir": str(dst),
                "log_path": str(out / f"eval_log_run_{new_idx:02d}.txt"),
            })
            next_idx += 1

    fields = ["run_idx", "success", "total", "success_rate",
              "returncode", "status", "elapsed_s", "run_dir", "log_path"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    total_eps = sum(r["total"] for r in rows)
    total_succ = sum(r["success"] for r in rows)
    pooled = (total_succ / total_eps) if total_eps else 0.0
    wlo, whi = _wilson_95(total_succ, total_eps)
    summary = {
        "n_runs": len(rows),
        "n_runs_ok": len(rows),
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "pooled_success_rate": pooled,
        "wilson_95_ci": [wlo, whi],
        "per_run_rate_mean": pooled,
        "per_run_rate_std": 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[merge] {total_succ}/{total_eps} = {pooled*100:.1f}%  wilson [{wlo*100:.1f}, {whi*100:.1f}]  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
