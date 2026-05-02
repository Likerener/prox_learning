"""Train a baseline (VLM-only ACT or prop-only MLP).

This is the most critical experiment: VLM-only ACT is identical to PLA with
proximity tokens removed (one ``--no-proximity`` flag). The PLA − VLM-only
delta on the near-contact task is the paper's headline number.

Run::

    python -m pla.train.train_baseline --variant vlm_only --config configs/train/act_baseline.yaml
    python -m pla.train.train_baseline --variant prop_only --config configs/train/act_baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

VARIANTS = ("vlm_only", "prop_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PLA baseline")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/baseline/"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"baseline training pending for variant={args.variant}. See docs/TIMELINE.md Day 4-5."
    )


if __name__ == "__main__":
    main()
