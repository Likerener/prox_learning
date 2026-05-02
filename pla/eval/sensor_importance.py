"""Sensor importance analysis.

For a trained PLA checkpoint, mask each sensor individually and measure the
performance drop on the near-contact task. Output: per-sensor success rate
delta, suitable for plotting as a heatmap on the FR3 body
(see ``pla/viz/heatmap.py``).

Run::

    python -m pla.eval.sensor_importance \\
        --checkpoint runs/pla/best.pt \\
        --task near_contact \\
        --n-episodes 50 \\
        --out reports/tables/sensor_importance.json
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sensor importance via masking")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="near_contact")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Loop over sensor index, zero its 8x8 input, run eval, store delta."
    )


if __name__ == "__main__":
    main()
