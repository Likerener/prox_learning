"""PLA — full model: frozen Molmo2 + ProximityEncoder + proprio + ACT decoder.

Token layout fed to the ACT decoder:

    [ visual-language tokens  ]   ~192 per RGB frame from Molmo2 (frozen)
    [ proximity tokens        ]   N_sensors from ProximityEncoder
    [ proprio token           ]   1 from a Linear(7→d_model)

Concatenated, layer-normed, then decoded to a chunk of 100 joint-delta actions.

This module deliberately keeps the heavy backbone wiring abstract so it can be
swapped: ``vision_language_backbone`` is any callable returning
``(B, N_vis, d_model)`` tokens given ``(rgb, language)``.
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn as nn

from pla.models.proximity_encoder import ProximityEncoder


class _VLBackbone(Protocol):
    def __call__(
        self, rgb: torch.Tensor, language_tokens: torch.Tensor
    ) -> torch.Tensor: ...


class PLA(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 512,
        n_sensors: int = 32,
        proprio_dim: int = 7,
        action_dim: int = 7,
        chunk_size: int = 100,
        vision_language_backbone: _VLBackbone | None = None,
        act_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Frozen by convention; freeze externally before passing in.
        self.vl = vision_language_backbone

        self.proximity_encoder = ProximityEncoder(d_model=d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # The ACT transformer decoder is implemented in ``submodules/act``.
        # Bring it in via dependency injection so this file stays import-light.
        self.act_decoder = act_decoder

    def encode_tokens(
        self,
        rgb: torch.Tensor,
        language_tokens: torch.Tensor,
        tof: torch.Tensor,
        qpos: torch.Tensor,
    ) -> torch.Tensor:
        """Returns fused tokens ready for the ACT decoder, shape [B, T, d_model]."""
        if self.vl is None:
            raise RuntimeError("vision_language_backbone not configured")
        vis = self.vl(rgb, language_tokens)  # [B, N_vis, d]
        prox = self.proximity_encoder(tof)  # [B, N_sensors, d]
        prop = self.proprio_proj(qpos).unsqueeze(1)  # [B, 1, d]
        tokens = torch.cat([vis, prox, prop], dim=1)
        return self.fusion_norm(tokens)

    def forward(
        self,
        rgb: torch.Tensor,
        language_tokens: torch.Tensor,
        tof: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict:
        if self.act_decoder is None:
            raise RuntimeError("act_decoder not configured")
        tokens = self.encode_tokens(rgb, language_tokens, tof, qpos)
        return self.act_decoder(tokens=tokens, actions=actions)
