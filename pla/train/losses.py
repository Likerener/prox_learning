"""Training losses.

ACT loss (Zhao et al. 2023): chunk-mean L1 + β·KL on the CVAE prior.

    L = (1/k) Σ_j |â_{t+j} − a_{t+j}|_1  +  β · D_KL(N(μ,σ²) ‖ N(0,I))

β = 10, constant (not annealed). Chunk size k = 100. At inference z = 0
and the CVAE encoder is discarded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def act_loss(
    pred_actions: torch.Tensor,  # [B, k, action_dim]
    gt_actions: torch.Tensor,  # [B, k, action_dim]
    mu: torch.Tensor,  # [B, z_dim]
    logvar: torch.Tensor,  # [B, z_dim]
    *,
    beta: float = 10.0,
) -> dict:
    l1 = F.l1_loss(pred_actions, gt_actions, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = l1 + beta * kl
    return {"loss": total, "l1": l1.detach(), "kl": kl.detach()}
