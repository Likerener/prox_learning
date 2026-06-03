"""Parallel runner for P+ACT mask / phase-mask experiments.

Launches N evaluation subprocesses in parallel pools of K workers, each
running ONE rollout with the same `--mask_proximity` / `--mask_phase`
settings. Aggregates success rates, writes results.csv + summary.json +
phase log (JSONL across all rollouts) and a per-run-bar plot.

Used for:
  * Exp 1 — `--mask_proximity {zero,mean}` with `--mask_phase none`.
  * Exp 2 — `--mask_proximity zero` with `--mask_phase <phase>` looped over phases.

Example:
    /opt/conda/envs/mlspaces/bin/python scripts/run_pact_mask_experiment.py \\
        --n_runs 80 --parallel 4 \\
        --mask_proximity zero --mask_phase none \\
        --output_dir eval_output/exp1_mask_zero_n80
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "submodules" / "act"
EVAL_SCRIPT = REPO_ROOT / "pact" / "act_prox" / "eval_act_with_prox_encoder.py"
DEFAULT_CKPT = REPO_ROOT / "runs" / "act_prox_mug_v1"
DEFAULT_PROX_ENCODER = REPO_ROOT / "pact" / "outputs_prox" / "runs" / "prox_encoder_v1" / "ckpt_best.pt"
DEFAULT_MAPPING = REPO_ROOT / "act_style_data" / "mug_house1_random_everything" / "prox_mapping.json"
DEFAULT_PROX_MEAN = REPO_ROOT / "pact" / "outputs_prox" / "runs" / "prox_encoder_v1" / "prox_pos_mean.npy"
DEFAULT_PY = "/opt/conda/envs/mlspaces/bin/python"

SUCCESS_RE = re.compile(r"\[act-prox-eval\]\s+success\s+(\d+)\s*/\s*(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=50)
    p.add_argument("--parallel", type=int, default=3,
                   help="Number of subprocesses to run in parallel. "
                        "Each uses ~7GB RAM + ~2GB VRAM. Default 3 fits comfortably.")
    p.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_encoder_ckpt", default=str(DEFAULT_PROX_ENCODER))
    p.add_argument("--prox_mapping_json", default=str(DEFAULT_MAPPING))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--python", default=DEFAULT_PY)

    p.add_argument("--mask_proximity", choices=("none", "zero", "mean", "noise", "shuffle"),
                   default="none")
    p.add_argument("--mask_phase",
                   choices=("none", "approach", "pregrasp", "grasp_lift", "transit", "place"),
                   default="none")
    p.add_argument("--prox_mean_path", default=str(DEFAULT_PROX_MEAN))

    # Defaults match the trained runs/act_prox_mug_v1 checkpoint (see
    # eval_output/act_prox_mug_v1_aggregate_n50/_driver.log).
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--prox_tokens_per_sensor", type=int, default=1)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="pact-eval-masking")
    p.add_argument("--wandb_group", default=None,
                   help="Defaults to mask_<mode>_phase_<phase>")
    p.add_argument("--start_idx", type=int, default=0,
                   help="First run_idx (useful for resuming).")
    p.add_argument("--keep_existing", action="store_true")
    return p.parse_args()


def _build_cmd(run_idx: int, args: argparse.Namespace, run_dir: Path,
               phase_log: Path) -> list[str]:
    run_name = (f"mask_{args.mask_proximity}_phase_{args.mask_phase}_"
                f"run_{run_idx:02d}_{int(time.time())}")
    cmd = [
        args.python, str(EVAL_SCRIPT),
        "--ckpt_dir", args.ckpt_dir,
        "--ckpt_name", args.ckpt_name,
        "--prox_encoder_ckpt", args.prox_encoder_ckpt,
        "--prox_mapping_json", args.prox_mapping_json,
        "--output_dir", str(run_dir),
        "--chunk_size", str(args.chunk_size),
        "--hidden_dim", str(args.hidden_dim),
        "--dim_feedforward", str(args.dim_feedforward),
        "--prox_tokens_per_sensor", str(args.prox_tokens_per_sensor),
        "--temp_agg_m", str(args.temp_agg_m),
        "--mask_proximity", args.mask_proximity,
        "--mask_phase", args.mask_phase,
        "--phase_log_path", str(phase_log),
    ]
    if args.temp_agg_off:
        cmd.append("--temp_agg_off")
    if args.mask_proximity == "mean":
        cmd += ["--prox_mean_path", args.prox_mean_path]
    if args.use_wandb:
        group = args.wandb_group or f"mask_{args.mask_proximity}_phase_{args.mask_phase}"
        cmd += [
            "--use_wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_group", group,
            "--wandb_run_name", run_name,
        ]
    return cmd


def run_one(payload: tuple[int, dict, str]) -> dict:
    """Run a single rollout subprocess. (Picklable: takes a tuple, no closures.)"""
    run_idx, args_dict, agg_dir_s = payload
    # Reconstruct Namespace.
    args = argparse.Namespace(**args_dict)
    agg_dir = Path(agg_dir_s)
    run_dir = agg_dir / f"run_{run_idx:02d}"
    if not args.keep_existing and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = agg_dir / f"eval_log_run_{run_idx:02d}.txt"
    phase_log = agg_dir / "phase_log.jsonl"

    cmd = _build_cmd(run_idx, args, run_dir, phase_log)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{ACT_DIR}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.pop("DISPLAY", None)

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
    status = ("ok" if proc.returncode == 0 and success is not None
              and total is not None and total > 0 else "fail")

    return {
        "run_idx": run_idx,
        "returncode": proc.returncode,
        "success": int(success) if success is not None else 0,
        "total": int(total) if total is not None else 0,
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


def write_summary(rows: list[dict], path: Path, args: argparse.Namespace) -> dict:
    rates = np.array([r["success_rate"] for r in rows], dtype=float)
    successes = np.array([r["success"] for r in rows], dtype=float)
    totals = np.array([r["total"] for r in rows], dtype=float)
    ok_mask = np.array([r["status"] == "ok" for r in rows])

    total_eps = int(totals.sum())
    total_succ = int(successes.sum())
    pooled = (total_succ / total_eps) if total_eps else 0.0

    if total_eps > 0:
        z = 1.96
        p = pooled
        n = total_eps
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        wilson = [float(centre - half), float(centre + half)]
    else:
        wilson = [0.0, 0.0]

    summary = {
        "n_runs": len(rows),
        "n_runs_ok": int(ok_mask.sum()),
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "pooled_success_rate": pooled,
        "wilson_95_ci": wilson,
        "per_run_rate_mean": float(rates.mean()) if len(rates) else 0.0,
        "per_run_rate_std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
        "config": {
            "mask_proximity": args.mask_proximity,
            "mask_phase": args.mask_phase,
            "ckpt_dir": args.ckpt_dir,
            "ckpt_name": args.ckpt_name,
            "prox_mean_path": args.prox_mean_path if args.mask_proximity == "mean" else None,
        },
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_results(rows: list[dict], summary: dict, path: Path, args: argparse.Namespace) -> None:
    n = len(rows)
    idxs = np.arange(n)
    rates = np.array([r["success_rate"] for r in rows], dtype=float)
    successes = np.array([r["success"] for r in rows], dtype=float)
    totals = np.array([r["total"] for r in rows], dtype=float)
    cum_s = np.cumsum(successes)
    cum_t = np.cumsum(totals)
    cum_rate = np.divide(cum_s, cum_t,
                         out=np.zeros_like(cum_s, dtype=float),
                         where=cum_t > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    colors = ["#2ca02c" if r >= 0.5 else "#d62728" for r in rates]
    ax.bar(idxs, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(summary["pooled_success_rate"], color="black",
               linestyle="--", linewidth=1.2,
               label=f"pooled = {summary['pooled_success_rate']:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("run index")
    ax.set_ylabel("success rate")
    ax.set_title(f"mask={args.mask_proximity}  phase={args.mask_phase}")
    ax.legend(loc="upper right", fontsize=9)

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
        f"P+ACT mask={args.mask_proximity} phase={args.mask_phase} — "
        f"{summary['total_successes']}/{summary['total_episodes']} = "
        f"{summary['pooled_success_rate']:.2%}  "
        f"(mean {summary['per_run_rate_mean']:.2%} ± {summary['per_run_rate_std']:.2%})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    agg_dir = Path(args.output_dir).resolve()
    agg_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pact-mask] aggregate dir   : {agg_dir}", flush=True)
    print(f"[pact-mask] ckpt_dir        : {args.ckpt_dir}", flush=True)
    print(f"[pact-mask] mask_proximity  : {args.mask_proximity}", flush=True)
    print(f"[pact-mask] mask_phase      : {args.mask_phase}", flush=True)
    print(f"[pact-mask] n_runs          : {args.n_runs}", flush=True)
    print(f"[pact-mask] parallel        : {args.parallel}", flush=True)

    # Wipe any previous phase log.
    phase_log = agg_dir / "phase_log.jsonl"
    if phase_log.exists() and not args.keep_existing:
        phase_log.unlink()

    # Build payloads.
    args_dict = vars(args)
    payloads = [(args.start_idx + i, args_dict, str(agg_dir))
                for i in range(args.n_runs)]

    rows: list[dict] = []
    t_all = time.time()
    print(f"[pact-mask] launching {args.n_runs} runs with parallelism={args.parallel}", flush=True)

    with ProcessPoolExecutor(max_workers=args.parallel) as exe:
        futures = {exe.submit(run_one, payload): payload for payload in payloads}
        for k, fut in enumerate(as_completed(futures)):
            payload = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                run_idx, _, _ = payload
                res = {
                    "run_idx": run_idx, "returncode": -1, "success": 0,
                    "total": 0, "success_rate": 0.0, "elapsed_s": 0.0,
                    "status": f"exc:{type(e).__name__}",
                    "run_dir": "", "log_path": "",
                }
            rows.append(res)
            rows_sorted = sorted(rows, key=lambda r: r["run_idx"])
            write_csv(rows_sorted, agg_dir / "results.csv")
            print(f"[pact-mask] done {k+1}/{args.n_runs}: "
                  f"run_idx={res['run_idx']} {res['success']}/{res['total']} "
                  f"({res['status']}, {res['elapsed_s']:.0f}s)", flush=True)

    rows = sorted(rows, key=lambda r: r["run_idx"])
    print(f"[pact-mask] all {args.n_runs} runs done in {(time.time() - t_all)/60:.1f} min", flush=True)

    summary = write_summary(rows, agg_dir / "summary.json", args)
    plot_results(rows, summary, agg_dir / "success_plot.png", args)

    print("\n[pact-mask] === SUMMARY ===")
    print(f"  mask              : {args.mask_proximity}  phase: {args.mask_phase}")
    print(f"  runs ok           : {summary['n_runs_ok']}/{summary['n_runs']}")
    print(f"  total episodes    : {summary['total_episodes']}")
    print(f"  total successes   : {summary['total_successes']}")
    print(f"  pooled rate       : {summary['pooled_success_rate']:.2%}")
    lo, hi = summary["wilson_95_ci"]
    print(f"  Wilson 95% CI     : [{lo:.2%}, {hi:.2%}]")


if __name__ == "__main__":
    main()
