"""Conditional VAE for 29x8x8 Franka-skin proximity readings.

Input x: normalized depth (N, 1856) in [0, 1].
Conditioning y: robot state (N, 14) = arm(7) + tcp_pose(7).

Encoder  q(z|x, y) = N(mu(x,y), sigma(x,y))
Decoder  p(x|z, y) = sigmoid MLP -> reconstructed depth
Prior    p(z|y)    = N(0, I)   (standard normal, independent of y)

The conditioning lets the model focus its latent capacity on environment-
driven variation (what's out there) rather than robot-configuration
variation (which is observable elsewhere).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CondVAE(nn.Module):
    def __init__(self, x_dim: int = 1856, y_dim: int = 14, z_dim: int = 32,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.x_dim, self.y_dim, self.z_dim = x_dim, y_dim, z_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, x_dim),
        )

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparam(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z, y):
        logits = self.dec(torch.cat([z, y], dim=-1))
        return torch.sigmoid(logits)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar


def elbo_loss(xhat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
              logvar: torch.Tensor, beta: float = 1e-3) -> dict:
    """Return dict of losses. Reconstruction is MSE per-pixel, summed per sample."""
    recon = F.mse_loss(xhat, x, reduction='none').sum(dim=-1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    loss = recon + beta * kl
    return dict(loss=loss, recon=recon, kl=kl)


def anomaly_score(model: CondVAE, x: torch.Tensor, y: torch.Tensor,
                  n_samples: int = 8) -> torch.Tensor:
    """Mean squared reconstruction error (a.k.a. VAE anomaly score).

    Uses `n_samples` posterior draws to stabilize the estimate on small
    eval batches. Higher score = more surprising reading.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        mu, logvar = model.encode(x, y)
        for _ in range(n_samples):
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            xhat = model.decode(z, y)
            scores.append(((xhat - x) ** 2).sum(dim=-1))
    return torch.stack(scores, dim=0).mean(dim=0)
