"""Shared-MLP encoder for whole-body ToF proximity readings.

Input  : [B, N_sensors, 8, 8] float (millimetres, clipped 20-4000).
Output : [B, N_sensors, d_model] tokens to be concatenated with vision/proprio
         tokens before the ACT decoder.

Why a shared MLP?
  * All sensors are the same hardware (VL53L5CX 8x8 SPAD) — same encoding
    function makes physical sense.
  * The model gets ``N_sensors`` × more gradient signal per training step.
  * Sensor identity is recoverable downstream via positional embeddings if
    needed (see PROJECT.md §3.4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProximityEncoder(nn.Module):
    def __init__(self, d_model: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h, w = x.shape
        assert (h, w) == (8, 8), f"expected 8x8 grid, got {h}x{w}"
        x = x.reshape(b * s, h * w)
        tokens = self.mlp(x)
        return tokens.reshape(b, s, self.d_model)
