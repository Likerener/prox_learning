"""Train the full PLA model.

Wires up:
  - frozen Molmo2-4B vision-language backbone
  - ProximityEncoder (shared MLP, trainable)
  - ACT decoder from submodules/act
  - act_loss with β=10, chunk=100, lr=1e-5, batch=8, Adam

Run::

    python -m pla.train.train_pla --config configs/train/pla.yaml

This is a stub. Fill in once data collection (Day 3-5) finishes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PLA")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/pla/"))
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "PLA training loop pending. See docs/TIMELINE.md Day 6-7."
    )


if __name__ == "__main__":
    main()
