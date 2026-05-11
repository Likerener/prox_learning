"""Shared MLP encoder for the 29 proximity sensors.

Maps the (B, 29, 8, 8) depth tensor to (B, 29, hidden_dim) tokens that the
ACT transformer encoder consumes alongside its image and proprio tokens.
Weights are shared across all 29 sensors — each sensor is treated as an
exchangeable instance of the same SPAD model and we let positional/sensor
embeddings (added in `pla/policy.py`) carry the per-sensor identity.

Spec from ../TODO.md §2:
  Linear(64 -> 128) -> ReLU -> Linear(128 -> 512)
"""
from __future__ import annotations

import torch
from torch import nn


class ProximityEncoder(nn.Module):
    """Encode (B, 29, 8, 8) depth → (B, 29, hidden_dim) tokens.

    Args:
        hidden_dim: output token dim per sensor (default 512, matches ACT).
        in_pixels: number of depth pixels per sensor (default 64 = 8 * 8).
        mid_dim: width of the hidden MLP layer (default 128).
        n_sensors: kept for shape sanity-checking (default 29).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        in_pixels: int = 64,
        mid_dim: int = 128,
        n_sensors: int = 29,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_pixels = in_pixels
        self.n_sensors = n_sensors
        # Shared MLP applied identically to every sensor.
        self.mlp = nn.Sequential(
            nn.Linear(in_pixels, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, hidden_dim),
        )

    def forward(self, prox: torch.Tensor) -> torch.Tensor:
        """prox: (B, 29, 8, 8) → tokens: (B, 29, hidden_dim)."""
        if prox.dim() != 4 or prox.shape[1] != self.n_sensors:
            raise ValueError(
                f"expected (B, {self.n_sensors}, 8, 8), got {tuple(prox.shape)}"
            )
        b, n, h, w = prox.shape
        flat = prox.reshape(b * n, h * w)  # (B*29, 64)
        tokens = self.mlp(flat)            # (B*29, hidden_dim)
        return tokens.reshape(b, n, self.hidden_dim)


if __name__ == "__main__":
    enc = ProximityEncoder()
    x = torch.randn(2, 29, 8, 8)
    y = enc(x)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}; params={n_params}")
