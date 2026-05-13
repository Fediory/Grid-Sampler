#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F  # noqa: N812
from torch import nn


class ActiveTokenSampler(nn.Module):
    def __init__(self, vision_dim: int, num_tokens: int = 16, embed_coords: bool = True, noise_scale: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_coords = embed_coords
        self.noise_scale = noise_scale
        self.vision_dim = vision_dim

        self.scout_mlp = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_tokens * 2),
            nn.Sigmoid(),
        )

        if self.embed_coords:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, vision_dim),
            )

    def forward(self, feature_map, noise=None):
        B, C, H, W = feature_map.shape
        if H != W:
            s = min(H, W)
            feature_map = F.adaptive_avg_pool2d(feature_map, (s, s))
            B, C, H, W = feature_map.shape

        global_feat = feature_map.mean(dim=(2, 3))
        pred_coords = self.scout_mlp(global_feat).view(B, self.num_tokens, 2)
        grid_coords = pred_coords * 2.0 - 1.0

        raw_sampled_feats = F.grid_sample(
            feature_map, grid_coords.unsqueeze(1), align_corners=False, mode="bilinear"
        ).squeeze(2).transpose(1, 2)
        final_feats = raw_sampled_feats

        if self.embed_coords:
            coord_emb = self.coord_encoder(pred_coords)
            final_feats = raw_sampled_feats + coord_emb

        return final_feats
