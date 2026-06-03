"""Run the prox-encoder ACT eval (`pact/act_prox/eval_act_with_prox_encoder.py`)
N times, aggregate success rate, and plot — mirrors run_act_mug_random_10x.py
for the K6 (or any K) prox-augmented checkpoint.

Each invocation evaluates ONE rollout (samples_per_house=1, house_inds=[1] from
FrankaSkinPickAndPlacePilotMediumConfig). Randomization across runs comes from
import-time np.random.uniform calls in the env config, so we always spawn a
fresh Python subprocess per run.

The wrapper deletes any existing house_*/ directory inside each per-run output
dir BEFORE launching the subprocess so we don't trigger the pipeline's
"skip if trajectories_batch_1_of_1.h5 exists" path.

Outputs under --output_dir:
    run_00/ ... run_09/        per-run rollout MP4s + h5
    eval_log_run_NN.txt        captured stdout per run
    results.csv                one row per run
    summary.json               aggregate stats
    success_plot.png           per-run bars + running pooled-rate curve
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = REPO_ROOT / "submodules" / "act"
EVAL_SCRIPT = REPO_ROOT / "pact" / "act_prox" / "eval_act_with_prox_encoder.py"
DEFAULT_CKPT = REPO_ROOT / "runs" / "act_prox_mug_v1_K6"
DEFAULT_PROX_ENCODER = REPO_ROOT / "pact" / "outputs_prox" / "runs" / "prox_encoder_v1" / "ckpt_best.pt"
DEFAULT_MAPPING = REPO_ROOT / "act_style_data" / "mug_house1_random_everything" / "prox_mapping.json"
DEFAULT_OUT = REPO_ROOT / "eval_output" / "act_prox_mug_v1_K6_aggregate"
DEFAULT_PY = "/opt/conda/envs/mlspaces/bin/python"

SUCCESS_RE = re.compile(r"\[act-prox-eval\]\s+success\s+(\d+)\s*/\s*(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--ckpt_dir", default=str(DEFAULT_CKPT))
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_encoder_ckpt", default=str(DEFAULT_PROX_ENCODER))
    p.add_argument("--prox_mapping_json", default=str(DEFAULT_MAPPING))
    p.add_argument("--output_dir", default=str(DEFAULT_OUT))
    p.add_argument("--python", default=DEFAULT_PY)
    # Match K6 training defaults; override if you trained with different shapes.
    p.add_argument("--chunk_size", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--prox_tokens_per_sensor", type=int, default=6)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="pact-eval")
    p.add_argument("--wandb_group", type=str, default="act_prox_mug_v1_K6")
    p.add_argument("--keep_existing", action="store_true",
                   help="Do NOT wipe per-run dirs before launching (debugging only — "
                        "leaving an old house_*/h5 in place will cause the pipeline to skip).")
    return p.parse_args()


def run_one(run_idx: int, args: argparse.Namespace, agg_dir: Path) -> dict:
    run_dir = agg_dir / f"run_{run_idx:02d}"
    if not args.keep_existing and run_dir.exists():
        # Nuke any leftover house_* output so the eval pipeline doesn't skip.
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = agg_dir / f"eval_log_run_{run_idx:02d}.txt"

    run_name = f"eval_act_prox_K{args.prox_tokens_per_sensor}_run_{run_idx:02d}_{int(time.time())}"
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
    ]
    if args.temp_agg_off:
        cmd.append("--temp_agg_off")
    if args.use_wandb:
        cmd += [
            "--use_wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_group", args.wandb_group,
            "--wandb_run_name", run_name,
        ]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{ACT_DIR}{os.pathsep}{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.pop("DISPLAY", None)

    print(f"\n[prox-10x] === run {run_idx + 1}/{args.n_runs} ===", flush=True)
    print(f"[prox-10x] cmd: {' '.join(cmd)}", flush=True)
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
    status = "ok" if proc.returncode == 0 and success is not None and total is not None and total > 0 else "fail"
    print(f"[prox-10x] run {run_idx} done in {dt:.1f}s — {status} "
          f"({success}/{total}) — log: {log_path}", flush=True)

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


def write_summary(rows: list[dict], path: Path) -> dict:
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
        f"ACT+prox K=? eval — {summary['total_successes']}/{summary['total_episodes']} "
        f"= {summary['pooled_success_rate']:.2%}  "
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

    print(f"[prox-10x] aggregate dir   : {agg_dir}")
    print(f"[prox-10x] ckpt_dir        : {args.ckpt_dir}")
    print(f"[prox-10x] prox_encoder    : {args.prox_encoder_ckpt}")
    print(f"[prox-10x] prox_mapping    : {args.prox_mapping_json}")
    print(f"[prox-10x] K (tokens/sens) : {args.prox_tokens_per_sensor}")
    print(f"[prox-10x] n_runs          : {args.n_runs}")
    print(f"[prox-10x] clean per-run   : {not args.keep_existing}")

    rows: list[dict] = []
    t_all = time.time()
    for i in range(args.n_runs):
        rows.append(run_one(i, args, agg_dir))
        write_csv(rows, agg_dir / "results.csv")
    print(f"\n[prox-10x] all {args.n_runs} runs done in {(time.time() - t_all) / 60:.1f} min")

    summary = write_summary(rows, agg_dir / "summary.json")
    plot_results(rows, summary, agg_dir / "success_plot.png")

    print("\n[prox-10x] === SUMMARY ===")
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
