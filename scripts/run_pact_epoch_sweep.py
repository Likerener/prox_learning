"""Sweep P+ACT eval over multiple checkpoint epochs to find the best behavioral one.

Re-uses the existing parallel runner with different `--ckpt_name` per condition.
Output: `eval_output/epoch_sweep/epoch_<N>/summary.json` + an aggregate plot.

Usage:
    /opt/conda/envs/mlspaces/bin/python scripts/run_pact_epoch_sweep.py \\
        --ckpt_dir runs/act_prox_mug_v1 \\
        --epochs 1500,1700,1900,best,last \\
        --n_runs 12 --parallel 2 \\
        --output_dir eval_output/epoch_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PY = "/opt/conda/envs/mlspaces/bin/python"
RUNNER = REPO_ROOT / "scripts" / "run_pact_mask_experiment.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=str(REPO_ROOT / "runs" / "act_prox_mug_v1"))
    p.add_argument("--epochs", required=True,
                   help="Comma-separated list, e.g. 1500,1700,1900,best,last "
                        "(maps to policy_epoch_1500_seed_0.ckpt, policy_best.ckpt, etc.)")
    p.add_argument("--seed_in_name", type=int, default=0,
                   help="Used to construct policy_epoch_<N>_seed_<S>.ckpt names.")
    p.add_argument("--n_runs", type=int, default=12)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def epoch_to_ckpt_name(epoch: str, seed: int) -> str:
    if epoch == "best":
        return "policy_best.ckpt"
    if epoch == "last":
        return "policy_last.ckpt"
    return f"policy_epoch_{int(epoch)}_seed_{seed}.ckpt"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    epochs = [e.strip() for e in args.epochs.split(",") if e.strip()]
    summaries: dict[str, dict] = {}

    for epoch in epochs:
        ck = epoch_to_ckpt_name(epoch, args.seed_in_name)
        ck_full = Path(args.ckpt_dir) / ck
        if not ck_full.exists():
            print(f"[epoch-sweep] SKIP {epoch}: {ck_full} missing")
            continue
        cond_dir = out / f"epoch_{epoch}"
        cond_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python, str(RUNNER),
            "--n_runs", str(args.n_runs),
            "--parallel", str(args.parallel),
            "--ckpt_dir", args.ckpt_dir,
            "--ckpt_name", ck,
            "--mask_proximity", "none",
            "--mask_phase", "none",
            "--output_dir", str(cond_dir),
        ]
        if args.wandb:
            cmd += ["--use_wandb",
                    "--wandb_project", "pact-epoch-sweep",
                    "--wandb_group", f"epoch_{epoch}"]
        log = cond_dir.parent / f"epoch_{epoch}.log"
        print(f"[epoch-sweep] === {epoch} ===  log={log}", flush=True)
        with open(log, "w") as lf:
            proc = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                env={**os.environ},
                stdout=lf, stderr=subprocess.STDOUT, check=False,
            )
        if proc.returncode != 0:
            print(f"[epoch-sweep] WARN {epoch} returned {proc.returncode}", flush=True)

        sj = cond_dir / "summary.json"
        if sj.exists():
            summaries[epoch] = json.loads(sj.read_text())
            print(f"[epoch-sweep] {epoch}: "
                  f"{summaries[epoch]['total_successes']}/"
                  f"{summaries[epoch]['total_episodes']} = "
                  f"{summaries[epoch]['pooled_success_rate']:.2%}", flush=True)

    # Aggregate plot.
    names = list(summaries.keys())
    rates = [summaries[e]["pooled_success_rate"] for e in names]
    cis   = [tuple(summaries[e]["wilson_95_ci"]) for e in names]
    counts = [(summaries[e]["total_successes"], summaries[e]["total_episodes"])
              for e in names]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(names))
    err_lo = np.array([r - lo for r, (lo, _) in zip(rates, cis)])
    err_hi = np.array([hi - r for r, (_, hi) in zip(rates, cis)])
    yerr = np.vstack([err_lo, err_hi])
    ax.bar(x, rates, yerr=yerr, capsize=4,
           color="#1f77b4", edgecolor="black", linewidth=0.6)
    for i, ((s, n), r) in enumerate(zip(counts, rates)):
        ax.text(x[i], r + 0.03, f"{int(s)}/{int(n)}\n({r:.0%})",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel("success rate")
    ax.set_title(f"P+ACT epoch sweep — n={args.n_runs} per epoch")
    ax.axhline(0.72, color="grey", linestyle="--", linewidth=1,
               label="vanilla ACT n=50 baseline (72%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "epoch_sweep.png", dpi=140)
    plt.close(fig)

    # Pick best.
    if names:
        best_epoch = max(names, key=lambda e: summaries[e]["pooled_success_rate"])
        print(f"\n[epoch-sweep] best epoch: {best_epoch}  "
              f"({summaries[best_epoch]['pooled_success_rate']:.2%})")
        with open(out / "best_epoch.json", "w") as f:
            json.dump({"best_epoch": best_epoch,
                       "best_rate": summaries[best_epoch]["pooled_success_rate"],
                       "all_summaries": summaries}, f, indent=2)

    print(f"[epoch-sweep] outputs in {out}")


if __name__ == "__main__":
    main()
