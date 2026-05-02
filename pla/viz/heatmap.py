"""Sensor placement heatmaps.

Two figures defined in PROJECT.md §6 Day 12:

  * ``tof_sequence_panels`` — 4-panel ToF heatmap sequence: arm far, mid-approach,
    near-contact, pre-grasp. The intuition figure for the paper.
  * ``sensor_importance_heatmap`` — colorize each of the 29–32 sensors on the FR3
    body by its measured contribution to near-contact success.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def tof_sequence_panels(
    tof_frames: list,  # length 4, each [N_sensors, 8, 8] in mm
    *,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    raise NotImplementedError("paper Figure 2; implement after first PLA run.")


def sensor_importance_heatmap(
    per_sensor_delta: dict,  # sensor_name -> success-rate drop when masked
    *,
    fr3_body_image: Path,
    out_path: Path,
) -> None:
    raise NotImplementedError("paper Figure 4; implement after sensor_importance.py runs.")
