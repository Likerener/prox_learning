"""Encoder-only Transformer that maps a time series of raw 8x8 proximity
frames to a 3D vector (object position in the sensor frame).

Input  : (B, T, 8, 8) normalized depth, where T = window_steps * 4 sub-frames.
Output : (B, 3) normalized 3D position; the dataset stores label_mean/label_std
         so callers can denormalize.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProxEncoderConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    frame_hw: int = 8
    cnn_channels: int = 32
    max_T: int = 128
    use_cls: bool = True


class FrameTokenizer(nn.Module):
    """8x8 depth frame -> d_model token."""

    def __init__(self, cfg: ProxEncoderConfig):
        super().__init__()
        # Small CNN preserves local spatial structure; then flatten + project.
        c = cfg.cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(c * cfg.frame_hw * cfg.frame_hw, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8, 8)
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x.view(B, T, -1)


class SinusoidPE(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProxEncoder(nn.Module):
    def __init__(self, cfg: ProxEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = FrameTokenizer(cfg)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.use_cls else None
        if cfg.use_cls:
            nn.init.normal_(self.cls, std=0.02)
        self.pe = SinusoidPE(cfg.d_model, cfg.max_T + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8, 8) — normalized depth (channel-wise z-scored in dataset).
        h = self.tok(x)                                # (B, T, D)
        if self.cls is not None:
            cls_tok = self.cls.expand(h.size(0), -1, -1)
            h = torch.cat([cls_tok, h], dim=1)         # (B, T+1, D)
        h = self.pe(h)
        h = self.enc(h)
        h = self.norm(h)
        pooled = h[:, 0] if self.cls is not None else h.mean(dim=1)
        return self.head(pooled)                       # (B, 3)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
