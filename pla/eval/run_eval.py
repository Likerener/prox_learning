"""Evaluation runner.

100 episodes per condition on each task (PROJECT.md §4.1). Writes a JSON
report per (model, task) cell with per-episode success, failure category,
and ToF statistics, plus aggregate bootstrap CIs and paired p-values for
PLA vs VLM-only ACT.

Run::

    python -m pla.eval.run_eval \\
        --checkpoint runs/pla/best.pt \\
        --tasks near_contact pnp pnp_color pnp_next_to \\
        --n-episodes 100 \\
        --out reports/eval/pla.json
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on PLA tasks")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Hook to MolmoSpaces FrankaPickandPlace eval. See docs/TIMELINE.md Day 5+."
    )


if __name__ == "__main__":
    main()
