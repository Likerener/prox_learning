"""ACT policy wrapper that optionally fuses 29 proximity tokens.

We do NOT modify the upstream ACT submodule (`submodules/act/`). Instead,
we reuse its primitives (`Transformer`, `Backbone`, the CVAE encoder,
positional encodings) and assemble a `PLA_DETRVAE` model that mirrors
`detr_vae.DETRVAE` but adds an optional proximity-token slot in the
transformer encoder context.

Modes:
- `use_proximity=True`  → encoder context is [latent_z, qpos_token,
                          *proximity_tokens(29), *image_tokens(HW per cam)].
- `use_proximity=False` → identical model + identical pre-trained backbone,
                          but the proximity slot is omitted from the context
                          (baseline: VLM-only ACT).

The CVAE encoder (CLS + qpos + action seq → latent z) is unchanged from
upstream ACT.

Hyperparameters per ../TODO.md §3:
  chunk_size=100  hidden_dim=512  enc/dec layers=7  beta(=kl_weight)=10
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# ---------------------------------------------------------------------
# Path-bootstrap so we can import the upstream ACT package as `detr.*`
# without touching its source.
# ---------------------------------------------------------------------
_ACT_DIR = Path(__file__).resolve().parent.parent / "submodules" / "act"
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
_DETR_DIR = _ACT_DIR / "detr"
if str(_DETR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETR_DIR))

from detr.models.transformer import build_transformer  # noqa: E402
from detr.models.backbone import build_backbone  # noqa: E402
from detr.models.detr_vae import (  # noqa: E402
    build_encoder,
    get_sinusoid_encoding_table,
    reparametrize,
)

from pla.proximity_encoder import ProximityEncoder  # noqa: E402


# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
@dataclass
class PLAConfig:
    use_proximity: bool = True

    # ACT-side hyperparameters (TODO §3 defaults)
    chunk_size: int = 100
    hidden_dim: int = 512
    enc_layers: int = 7
    dec_layers: int = 7
    nheads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    backbone: str = "resnet18"
    position_embedding: str = "sine"
    masks: bool = False
    dilation: bool = False
    pre_norm: bool = False
    lr_backbone: float = 1e-5  # > 0 so backbone params receive gradients

    # I/O
    qpos_dim: int = 7
    action_dim: int = 8  # 7 arm + 1 normalized gripper command (see pla/dataset.py)
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")

    # Proximity branch
    n_proximity_sensors: int = 29

    # Loss
    kl_weight: float = 10.0


def _to_argparse_namespace(cfg: PLAConfig) -> SimpleNamespace:
    """Pack PLAConfig into a SimpleNamespace shaped like argparse args, since
    `build_transformer` / `build_backbone` read attributes directly."""
    return SimpleNamespace(
        backbone=cfg.backbone,
        position_embedding=cfg.position_embedding,
        lr_backbone=cfg.lr_backbone,
        masks=cfg.masks,
        dilation=cfg.dilation,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        nheads=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        pre_norm=cfg.pre_norm,
        num_queries=cfg.chunk_size,
        camera_names=list(cfg.camera_names),
    )


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
class PLA_DETRVAE(nn.Module):
    """ACT-style CVAE policy with an optional proximity-token branch.

    Architecture mirrors `detr_vae.DETRVAE` (CVAE encoder over actions,
    transformer encoder over multi-modal context, transformer decoder
    queries → action sequence). The only difference vs upstream is the
    `proximity_tokens` injected into the transformer encoder context when
    `use_proximity=True`.
    """

    def __init__(self, cfg: PLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        args = _to_argparse_namespace(cfg)

        # --- vision backbone (one shared per cam — ACT default) ---
        self.backbones = nn.ModuleList(
            [build_backbone(args) for _ in cfg.camera_names]
        )
        self.input_proj = nn.Conv2d(
            self.backbones[0].num_channels, cfg.hidden_dim, kernel_size=1
        )

        # --- main transformer (encoder over context, decoder over queries) ---
        self.transformer = build_transformer(args)
        # CVAE prior encoder over (action sequence + qpos + CLS)
        self.encoder = build_encoder(args)

        # --- decoder query embeddings (one per chunk slot) ---
        self.num_queries = cfg.chunk_size
        self.query_embed = nn.Embedding(cfg.chunk_size, cfg.hidden_dim)
        self.action_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.is_pad_head = nn.Linear(cfg.hidden_dim, 1)

        # --- proprioception projection ---
        self.input_proj_robot_state = nn.Linear(cfg.qpos_dim, cfg.hidden_dim)

        # --- CVAE encoder pieces ---
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, cfg.hidden_dim)
        self.encoder_action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.encoder_joint_proj = nn.Linear(cfg.qpos_dim, cfg.hidden_dim)
        self.latent_proj = nn.Linear(cfg.hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + cfg.chunk_size, cfg.hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, cfg.hidden_dim)

        # --- proximity branch ---
        n_extra = cfg.n_proximity_sensors if cfg.use_proximity else 0
        self.proximity_encoder = (
            ProximityEncoder(
                hidden_dim=cfg.hidden_dim, n_sensors=cfg.n_proximity_sensors
            )
            if cfg.use_proximity
            else None
        )
        # Learned positional embeddings for the non-image context tokens:
        # [latent, qpos] when proximity is off (matches upstream ACT),
        # [latent, qpos, *29 proximity sensors] when it's on.
        self.additional_pos_embed = nn.Embedding(2 + n_extra, cfg.hidden_dim)
        self._n_extra_tokens = n_extra

    # ------------------------------------------------------------------
    # CVAE prior encoder: same as DETRVAE
    # ------------------------------------------------------------------
    def _encode_latent(
        self,
        qpos: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = qpos.shape[0]
        action_embed = self.encoder_action_proj(actions)             # (bs, k, h)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)       # (bs, 1, h)
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # (bs, 1, h)
        encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1)
        encoder_input = encoder_input.permute(1, 0, 2)  # (k+2, bs, h)
        cls_qpos_pad = torch.zeros((bs, 2), dtype=torch.bool, device=qpos.device)
        is_pad_full = torch.cat([cls_qpos_pad, is_pad], dim=1)  # (bs, k+2)
        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)  # (k+2, 1, h)
        out = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
        cls_out = out[0]  # (bs, h)
        latent_info = self.latent_proj(cls_out)
        mu, logvar = latent_info[:, : self.latent_dim], latent_info[:, self.latent_dim:]
        latent = reparametrize(mu, logvar)
        return self.latent_out_proj(latent), mu, logvar

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        qpos: torch.Tensor,                       # (B, qpos_dim)
        image: torch.Tensor,                      # (B, num_cam, 3, H, W)
        proximity: torch.Tensor | None = None,    # (B, 29, 8, 8) in [0,1]
        actions: torch.Tensor | None = None,      # (B, k, action_dim) train-only
        is_pad: torch.Tensor | None = None,       # (B, k) train-only
    ):
        bs = qpos.shape[0]
        device = qpos.device

        # --- CVAE latent (training) or prior sample (inference) ---
        if actions is not None:
            latent_input, mu, logvar = self._encode_latent(qpos, actions, is_pad)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(bs, self.latent_dim, device=device)
            latent_input = self.latent_out_proj(latent_sample)
        proprio_input = self.input_proj_robot_state(qpos)  # (bs, h)

        # --- per-camera ResNet features + position encodings ---
        all_cam_features, all_cam_pos = [], []
        for cam_id in range(image.shape[1]):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # fold camera dim into width: (B, h, H', cams*W')
        src_img = torch.cat(all_cam_features, dim=3)
        pos_img = torch.cat(all_cam_pos, dim=3)
        b, c, h, w = src_img.shape
        src_img_flat = src_img.flatten(2).permute(2, 0, 1)            # (HW, B, h)
        pos_img_flat = pos_img.flatten(2).permute(2, 0, 1).repeat(1, b, 1)  # (HW, B, h)

        # --- non-image (additional) tokens ---
        addl_tokens = [latent_input, proprio_input]                  # each (B, h)
        if self.proximity_encoder is not None:
            if proximity is None:
                raise ValueError("use_proximity=True but proximity tensor is None")
            prox_tokens = self.proximity_encoder(proximity)           # (B, 29, h)
            for i in range(prox_tokens.shape[1]):
                addl_tokens.append(prox_tokens[:, i])
        addl_stack = torch.stack(addl_tokens, dim=0)                  # (n_extra+2, B, h)

        # --- positional embedding for additional tokens ---
        addl_pos = (
            self.additional_pos_embed.weight.unsqueeze(1).repeat(1, b, 1)
        )  # (n_extra+2, B, h)
        pos_full = torch.cat([addl_pos, pos_img_flat], dim=0)
        src_full = torch.cat([addl_stack, src_img_flat], dim=0)

        # --- transformer encoder + decoder ---
        memory = self.transformer.encoder(src_full, src_key_padding_mask=None, pos=pos_full)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # (k, B, h)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_full,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)            # (num_layers, B, k, h)
        a_hat = self.action_head(hs)       # (num_layers, B, k, action_dim)
        is_pad_hat = self.is_pad_head(hs)
        # Use only the last decoder layer output (matches upstream ACT)
        return a_hat[-1], is_pad_hat[-1], (mu, logvar)


# ---------------------------------------------------------------------
# top-level policy wrapper (loss + image normalization, like ACTPolicy)
# ---------------------------------------------------------------------
class PLAPolicy(nn.Module):
    """Wraps `PLA_DETRVAE` with image normalization and the L1 + KL loss.

    Train: returns `loss_dict = {l1, kl, loss}`.
    Inference (no actions): returns predicted action chunk `(B, k, action_dim)`.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg: PLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = PLA_DETRVAE(cfg)
        self.kl_weight = cfg.kl_weight
        self._normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)), summed over latent dim, mean over batch."""
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds.sum(1).mean(0, keepdim=True)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,                     # (B, num_cam, 3, H, W) in [0,1]
        proximity: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ):
        # Normalize per-cam.
        b, n_cam = image.shape[0], image.shape[1]
        image_norm = self._normalize(image.reshape(b * n_cam, *image.shape[2:])).reshape(image.shape)

        if actions is None:
            a_hat, _, _ = self.model(qpos, image_norm, proximity=proximity)
            return a_hat

        actions = actions[:, : self.cfg.chunk_size]
        is_pad = is_pad[:, : self.cfg.chunk_size]
        a_hat, _, (mu, logvar) = self.model(
            qpos, image_norm, proximity=proximity, actions=actions, is_pad=is_pad
        )
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * (~is_pad).unsqueeze(-1)).mean()
        total_kld = self.kl_divergence(mu, logvar)
        loss = l1 + self.kl_weight * total_kld[0]
        return {"l1": l1, "kl": total_kld[0], "loss": loss}

    def configure_optimizer(self, lr: float = 1e-5, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
        # Same param-group split as upstream ACT: backbone gets lr_backbone.
        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "backbones" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "backbones" in n and p.requires_grad
                ],
                "lr": self.cfg.lr_backbone,
            },
        ]
        return torch.optim.Adam(param_dicts, lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    cfg = PLAConfig(use_proximity=True)
    p = PLAPolicy(cfg).cuda()
    bs = 2
    qpos = torch.randn(bs, cfg.qpos_dim, device="cuda")
    image = torch.rand(bs, len(cfg.camera_names), 3, 480, 640, device="cuda")
    prox = torch.rand(bs, 29, 8, 8, device="cuda")
    actions = torch.randn(bs, cfg.chunk_size, cfg.action_dim, device="cuda")
    is_pad = torch.zeros(bs, cfg.chunk_size, dtype=torch.bool, device="cuda")
    loss = p(qpos, image, prox, actions, is_pad)
    print("loss_dict:", {k: float(v) for k, v in loss.items()})
    n = sum(pp.numel() for pp in p.parameters() if pp.requires_grad)
    print(f"params: {n/1e6:.2f}M")
