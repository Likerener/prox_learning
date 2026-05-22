"""FrozenProxFeatureExtractor — frozen prox-encoder wrapped as an ACT input.

Given a per-sensor temporal proximity window of shape `(B, N_sensors, W*4, 8, 8)`
(already z-scored by the dataset using the checkpoint's `prox_mean / prox_std`),
this module runs the encoder once over `B * N_sensors` items and returns the
predicted 3D object position in each sensor's local frame, shape
`(B, N_sensors, 3)` in **metres**.

The module is parameter-frozen: `requires_grad_(False)` + `eval()` at construction,
and `forward(...)` runs the encoder under `torch.no_grad()`. There is no path
by which an optimizer can update the encoder weights via this module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Make `prox_encoder.*` importable when called as `python -m pact.*` or directly.
_PACT_DIR = Path(__file__).resolve().parents[1]
if str(_PACT_DIR) not in sys.path:
    sys.path.insert(0, str(_PACT_DIR))
from prox_encoder.model import ProxEncoder, ProxEncoderConfig  # noqa: E402


class FrozenProxFeatureExtractor(nn.Module):
    """(B, N_sensors, W*4, 8, 8) z-scored prox  ->  (B, N_sensors, 3) metres."""

    def __init__(self, ckpt_path: str | Path, device: Optional[torch.device] = None):
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ProxEncoderConfig(**ckpt["cfg"])
        self.encoder = ProxEncoder(cfg)
        missing, unexpected = self.encoder.load_state_dict(ckpt["model"], strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state-dict mismatch: missing={missing} unexpected={unexpected}"
            )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # Sanity: belt-and-suspenders.
        for n, p in self.encoder.named_parameters():
            if p.requires_grad:
                raise AssertionError(f"prox encoder param {n} still has requires_grad=True")

        # Persist denormalization constants and the window setting.
        self.register_buffer("label_mean", torch.as_tensor(ckpt["label_mean"], dtype=torch.float32))
        self.register_buffer("label_std",  torch.as_tensor(ckpt["label_std"],  dtype=torch.float32))
        self.register_buffer("prox_mean",  torch.as_tensor(ckpt["prox_mean"],  dtype=torch.float32))
        self.register_buffer("prox_std",   torch.as_tensor(ckpt["prox_std"],   dtype=torch.float32))
        self.window: int = int(ckpt["window"])
        self.cfg_dict: Dict = dict(ckpt["cfg"])

        if device is not None:
            self.to(device)

    @property
    def n_substeps(self) -> int:
        return 4

    @property
    def T(self) -> int:
        return self.window * self.n_substeps

    @torch.no_grad()
    def forward(self, prox_window: torch.Tensor) -> torch.Tensor:
        """Args
            prox_window: (B, N_sensors, W*4, 8, 8) float, z-scored by the dataset.

        Returns
            (B, N_sensors, 3) float, predicted object position per sensor in **metres**.
        """
        if prox_window.dim() != 5:
            raise ValueError(f"expected 5-D input, got shape {tuple(prox_window.shape)}")
        B, S, T, H, W = prox_window.shape
        if T != self.T:
            raise ValueError(f"expected T={self.T} (window={self.window} * 4 substeps), got {T}")
        flat = prox_window.reshape(B * S, T, H, W).contiguous()
        pred_norm = self.encoder(flat)                                  # (B*S, 3)
        # Denormalize to metres.
        pred = pred_norm * self.label_std + self.label_mean
        return pred.view(B, S, 3)
