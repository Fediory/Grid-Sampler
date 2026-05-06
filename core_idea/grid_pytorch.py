"""PyTorch active token sampling (grid) — mirrors openpi.models.grid for PI0Pytorch training."""

from __future__ import annotations

import logging

import torch
from torch import nn
_logger = logging.getLogger(__name__)


def grid_sample_bilinear_batch(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling aligned with `openpi.models.grid.grid_sample_jax` (batched).

    Args:
        feat: [B, H, W, C] feature map.
        coords: [B, K, 2] normalized coordinates in [0, 1].

    Returns:
        [B, K, C] sampled features.
    """
    B, H, W, C = feat.shape
    x = coords[..., 0] * (W - 1)
    y = coords[..., 1] * (H - 1)

    x0 = x.floor().to(torch.int64).clamp(0, W - 1)
    y0 = y.floor().to(torch.int64).clamp(0, H - 1)
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)

    K = coords.shape[1]
    b_ix = torch.arange(B, device=feat.device).unsqueeze(1).expand(B, K)
    Ia = feat[b_ix, y0, x0]
    Ib = feat[b_ix, y1, x0]
    Ic = feat[b_ix, y0, x1]
    Id = feat[b_ix, y1, x1]

    x0_f = x0.to(dtype=x.dtype)
    y0_f = y0.to(dtype=y.dtype)
    x1_f = x1.to(dtype=x.dtype)
    y1_f = y1.to(dtype=y.dtype)

    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)
    wc = (x - x0_f) * (y1_f - y)
    wd = (x - x0_f) * (y - y0_f)

    return wa.unsqueeze(-1) * Ia + wb.unsqueeze(-1) * Ib + wc.unsqueeze(-1) * Ic + wd.unsqueeze(-1) * Id


class ActiveTokenSamplerTorch(nn.Module):
    """Predicts normalized [0,1]² coords and bilinear-samples tokens; optional coord embedding."""

    def __init__(
        self,
        vision_dim: int,
        num_tokens: int = 16,
        embed_coords: bool = True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_coords = embed_coords

        self.scout_mlp = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_tokens * 2),
            nn.Sigmoid(),
        )
        if embed_coords:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, vision_dim),
            )
        else:
            self.coord_encoder = None


    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            feature_map: [B, H, W, C]
            noise: if True, add Gaussian noise to pooled features before coord MLP (exploration).

        Returns:
            sampled_feats: [B, num_tokens, C], pred_coords: [B, num_tokens, 2]
        """
        B, _H, _W, C = feature_map.shape
        global_feat = feature_map.mean(dim=(1, 2))

        pred_coords = self.scout_mlp(global_feat).view(B, self.num_tokens, 2)

        sampled_feats = grid_sample_bilinear_batch(feature_map, pred_coords)
        if self.embed_coords and self.coord_encoder is not None:
            coord_emb = self.coord_encoder(pred_coords)
            sampled_feats = sampled_feats + coord_emb

        return sampled_feats, pred_coords
