"""Replay a saved `pla.train` stdout log into a WandB run.

Use when training was launched with `--use_wandb false` and you want the
curves on the cloud dashboard after the fact. The log format expected is
the one produced by pla.train's `--log_every` block:

  [step      50] loss=12.21 l1=0.28 kl=1.19  18.7 samp/s

Usage:
  python scripts/backfill_wandb_from_log.py \
      --log logs/train_pla_v3.log \
      --run_name smoke_pla_v3_full \
      --project pla \
      --tags backfill,smoke,proximity

Each parsed line becomes a single `wandb.log({...}, step=N)` call. Checkpoint
lines (`[ckpt] saved ...`) are emitted as a step-tagged annotation. The
config dict (use_proximity, etc.) is inferred from the `--config_extra` JSON
arg or left blank.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([\d.eE+-]+)\s+l1=([\d.eE+-]+)\s+kl=([\d.eE+-]+)\s+([\d.]+)\s+samp/s"
)
CKPT_RE = re.compile(r"\[ckpt\] saved\s+(\S+)")
MODEL_RE = re.compile(r"\[model\] use_proximity=(\w+),\s+params=([\d.]+)M")
DATASET_RE = re.compile(r"\[dataset\] indexed (\d+) trajectories, (\d+) timesteps")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--project", type=str, default="pla")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--tags", type=str, default="backfill",
                   help="Comma-separated tag list.")
    p.add_argument("--config_extra", type=str, default="{}",
                   help="JSON dict of extra config fields to log.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"log not found: {log_path}", file=sys.stderr)
        return 1

    import wandb  # imported here so the file is importable without wandb

    parsed_steps: list[tuple[int, dict[str, float]]] = []
    config = {"backfilled_from": str(log_path)}

    for line in log_path.read_text().splitlines():
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            parsed_steps.append((step, {
                "train/loss": float(m.group(2)),
                "train/l1": float(m.group(3)),
                "train/kl": float(m.group(4)),
                "train/throughput_samp_per_s": float(m.group(5)),
            }))
            continue
        m = MODEL_RE.search(line)
        if m:
            config["use_proximity"] = m.group(1).lower() == "true"
            config["params_M"] = float(m.group(2))
            continue
        m = DATASET_RE.search(line)
        if m:
            config["n_trajectories"] = int(m.group(1))
            config["n_timesteps"] = int(m.group(2))
            continue

    extra = json.loads(args.config_extra)
    config.update(extra)

    if not parsed_steps:
        print(f"no step rows parsed from {log_path}", file=sys.stderr)
        return 1

    print(f"parsed {len(parsed_steps)} step rows from {log_path}")
    print(f"config: {config}")
    print(f"  steps from {parsed_steps[0][0]} to {parsed_steps[-1][0]}")
    print(f"  final loss = {parsed_steps[-1][1]['train/loss']:.4f}")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        tags=tags,
        notes=f"Backfilled from {log_path.name}",
        reinit=True,
    )
    for step, metrics in parsed_steps:
        wandb.log(metrics, step=step)
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
