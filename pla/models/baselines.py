"""Baseline models for the ablation ladder.

VLMOnlyACT — primary comparison: identical to PLA but with proximity tokens
removed. The delta on the near-contact task is the paper's headline result.

PropOnlyMLP — sanity floor. Maps qpos directly to actions through a small MLP.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class _VLBackbone(Protocol):
    def __call__(
        self, rgb: torch.Tensor, language_tokens: torch.Tensor
    ) -> torch.Tensor: ...


class VLMOnlyACT(nn.Module):
    """PLA minus the ProximityEncoder. Same backbone, same ACT decoder."""

    def __init__(
        self,
        *,
        d_model: int = 512,
        proprio_dim: int = 7,
        action_dim: int = 7,
        chunk_size: int = 100,
        vision_language_backbone: _VLBackbone | None = None,
        act_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.vl = vision_language_backbone
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.act_decoder = act_decoder
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def forward(
        self,
        rgb: torch.Tensor,
        language_tokens: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict:
        if self.vl is None or self.act_decoder is None:
            raise RuntimeError("backbone or decoder not configured")
        vis = self.vl(rgb, language_tokens)
        prop = self.proprio_proj(qpos).unsqueeze(1)
        tokens = self.fusion_norm(torch.cat([vis, prop], dim=1))
        return self.act_decoder(tokens=tokens, actions=actions)


class PropOnlyMLP(nn.Module):
    """7-dim qpos → action chunk. Floor baseline."""

    def __init__(
        self, proprio_dim: int = 7, action_dim: int = 7, chunk_size: int = 100,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk_size * action_dim),
        )

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        b = qpos.shape[0]
        return self.net(qpos).reshape(b, self.chunk_size, self.action_dim)
