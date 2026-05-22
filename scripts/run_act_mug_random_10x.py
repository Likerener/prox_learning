"""Run the mug_house1_random ACT eval N times, aggregate success rate, plot.

Each invocation of submodules/act/eval_act_mug_random.py evaluates ONE rollout
(samples_per_house=1, house_inds=[1]). The randomization that varies across runs
comes from import-time np.random.uniform calls inside
FrankaSkinPickAndPlacePilotMediumConfig.task_sampler_config -- so we spawn a
fresh Python subprocess per run.

Outputs:
    eval_output/act_house1_mug_random_v1_aggregate/
        run_00/  ... run_09/       <- per-run rollout MP4s + h5
        results.csv                <- one row per run
        summary.json               <- aggregate stats
        success_plot.png           <- per-run bars + running-mean curve
        eval_log_run_00.txt ...    <- captured stdout per run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "submodules" / "act"
EVAL_SCRIPT = ACT_DIR / "eval_act_mug_random.py"
DEFAULT_CKPT = ACT_DIR / "ckpts" / "act_house1_mug_random_v1"
DEFAULT_OUT = REPO_ROOT / "eval_output" / "act_house1_mug_random_v1_aggregate"
DEFAULT_PY = "/opt/conda/envs/mlspaces/bin/python"

SUCCESS_RE = re.compile(r"\[act-eval\]\s+success\s+(\d+)\s*/\s*(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT),
                   help="Aggregate dir; per-run subdirs are created underneath.")
    p.add_argument("--python", default=DEFAULT_PY,
                   help="Python interpreter (mlspaces env).")
    p.add_argument("--use_wandb", action="store_true",
                   help="Forwarded to the eval script — each run gets its own wandb run "
                        "grouped under act_house1_mug_random_v1.")
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args appended to each eval invocation.")
    return p.parse_args()


def run_one(run_idx: int, args: argparse.Namespace, agg_dir: Path) -> dict:
    run_dir = agg_dir / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = agg_dir / f"eval_log_run_{run_idx:02d}.txt"

    cmd = [
        args.python, str(EVAL_SCRIPT),
        "--ckpt_dir", args.ckpt_dir,
        "--output_dir", str(run_dir),
    ]
    if args.use_wandb:
        cmd += ["--use_wandb",
                "--wandb_run_name", f"eval_mug_random_run_{run_idx:02d}_{int(time.time())}"]
    if args.temp_agg_off:
        cmd.append("--temp_agg_off")
    cmd += list(args.extra)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ACT_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.pop("DISPLAY", None)

    print(f"\n[10x] === run {run_idx + 1}/{args.n_runs} ===", flush=True)
    print(f"[10x] cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, cwd=str(ACT_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
        logf.write(proc.stdout or "")
    dt = time.time() - t0

    success, total = None, None
    for line in (proc.stdout or "").splitlines():
        m = SUCCESS_RE.search(line)
        if m:
            success, total = int(m.group(1)), int(m.group(2))
    status = "ok" if proc.returncode == 0 and success is not None else "fail"
    print(f"[10x] run {run_idx} done in {dt:.1f}s — {status} "
          f"({success}/{total}) — log: {log_path}", flush=True)

    return {
        "run_idx": run_idx,
        "returncode": proc.returncode,
        "success": success if success is not None else 0,
        "total": total if total is not None else 0,
        "success_rate": (success / total) if (success is not None and total) else 0.0,
        "elapsed_s": dt,
        "status": status,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["run_idx", "success", "total", "success_rate",
              "returncode", "status", "elapsed_s", "run_dir", "log_path"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_summary(rows: list[dict], path: Path) -> dict:
    successes = np.array([r["success"] for r in rows], dtype=float)
    totals = np.array([r["total"] for r in rows], dtype=float)
    rates = np.array([r["success_rate"] for r in rows], dtype=float)

    ok_mask = np.array([r["status"] == "ok" for r in rows])
    n_ok = int(ok_mask.sum())
    n_runs = len(rows)
    total_eps = int(totals.sum())
    total_succ = int(successes.sum())
    pooled_rate = (total_succ / total_eps) if total_eps > 0 else 0.0

    # 95% Wilson interval on the pooled rate
    if total_eps > 0:
        z = 1.96
        p = pooled_rate
        n = total_eps
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        wilson_low, wilson_high = float(centre - half), float(centre + half)
    else:
        wilson_low = wilson_high = 0.0

    summary = {
        "n_runs": n_runs,
        "n_runs_ok": n_ok,
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "pooled_success_rate": pooled_rate,
        "wilson_95_ci": [wilson_low, wilson_high],
        "per_run_rate_mean": float(rates.mean()) if len(rates) else 0.0,
        "per_run_rate_std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_results(rows: list[dict], summary: dict, path: Path) -> None:
    n = len(rows)
    idxs = np.arange(n)
    rates = np.array([r["success_rate"] for r in rows], dtype=float)
    successes = np.array([r["success"] for r in rows], dtype=float)
    totals = np.array([r["total"] for r in rows], dtype=float)

    # cumulative pooled success rate as runs accumulate
    cum_succ = np.cumsum(successes)
    cum_tot = np.cumsum(totals)
    cum_rate = np.divide(cum_succ, cum_tot,
                         out=np.zeros_like(cum_succ, dtype=float),
                         where=cum_tot > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    bar_colors = ["#2ca02c" if r >= 0.5 else "#d62728" for r in rates]
    ax.bar(idxs, rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(summary["pooled_success_rate"], color="black",
               linestyle="--", linewidth=1.2,
               label=f"pooled = {summary['pooled_success_rate']:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(idxs)
    ax.set_xlabel("run index")
    ax.set_ylabel("success rate")
    ax.set_title("Per-run success rate")
    ax.legend(loc="upper right", fontsize=9)
    for i, (s, t) in enumerate(zip(successes, totals)):
        ax.text(i, rates[i] + 0.02, f"{int(s)}/{int(t)}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.plot(idxs + 1, cum_rate, marker="o", color="#1f77b4", linewidth=2)
    lo, hi = summary["wilson_95_ci"]
    ax.fill_between(idxs + 1,
                    np.full_like(idxs, lo, dtype=float),
                    np.full_like(idxs, hi, dtype=float),
                    color="#1f77b4", alpha=0.12,
                    label=f"final 95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("runs aggregated")
    ax.set_ylabel("pooled success rate")
    ax.set_title("Running pooled success rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"ACT mug_house1_random eval — {summary['total_successes']}/"
        f"{summary['total_episodes']} = {summary['pooled_success_rate']:.2%} "
        f"(per-run mean {summary['per_run_rate_mean']:.2%} ± "
        f"{summary['per_run_rate_std']:.2%})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    agg_dir = Path(args.output_dir).resolve()
    agg_dir.mkdir(parents=True, exist_ok=True)

    print(f"[10x] aggregate dir: {agg_dir}")
    print(f"[10x] ckpt:          {args.ckpt_dir}")
    print(f"[10x] n_runs:        {args.n_runs}")

    rows: list[dict] = []
    t_all = time.time()
    for i in range(args.n_runs):
        rows.append(run_one(i, args, agg_dir))
        # write CSV after every run so progress is durable
        write_csv(rows, agg_dir / "results.csv")
    print(f"\n[10x] all {args.n_runs} runs done in {(time.time() - t_all) / 60:.1f} min")

    summary = write_summary(rows, agg_dir / "summary.json")
    plot_results(rows, summary, agg_dir / "success_plot.png")

    print("\n[10x] === SUMMARY ===")
    print(f"  runs ok           : {summary['n_runs_ok']}/{summary['n_runs']}")
    print(f"  total episodes    : {summary['total_episodes']}")
    print(f"  total successes   : {summary['total_successes']}")
    print(f"  pooled rate       : {summary['pooled_success_rate']:.2%}")
    lo, hi = summary["wilson_95_ci"]
    print(f"  Wilson 95% CI     : [{lo:.2%}, {hi:.2%}]")
    print(f"  per-run rate mean : {summary['per_run_rate_mean']:.2%}")
    print(f"  per-run rate std  : {summary['per_run_rate_std']:.2%}")
    print(f"\n  CSV  : {agg_dir / 'results.csv'}")
    print(f"  JSON : {agg_dir / 'summary.json'}")
    print(f"  PLOT : {agg_dir / 'success_plot.png'}")


if __name__ == "__main__":
    main()
