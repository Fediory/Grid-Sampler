import logging

import flax.linen as nn
import jax
import jax.numpy as jnp
import einops
import matplotlib
matplotlib.use('Agg')  # Set before importing pyplot
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


def grid_sample_jax(image, coords):
    H, W, C = image.shape
    x = coords[:, 0] * (W - 1)
    y = coords[:, 1] * (H - 1)
    
    # Fast index computation
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    
    # Use min/max instead of clip for better performance
    x0 = jnp.maximum(0, jnp.minimum(x0, W - 1))
    y0 = jnp.maximum(0, jnp.minimum(y0, H - 1))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # memory coalescing — read C dimension in one shot
    Ia = image[y0, x0] # [K, C]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]
    
    # Compute weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # Broadcast and weighted sum
    return (wa[..., None] * Ia + 
            wb[..., None] * Ib + 
            wc[..., None] * Ic + 
            wd[..., None] * Id)



class ActiveTokenSampler(nn.Module):
    vision_dim: int       
    num_tokens: int = 16  
    embed_coords: bool = True 

    def setup(self):
        self.scout_mlp = nn.Sequential([
            nn.Dense(512),
            nn.relu,
            nn.Dense(self.num_tokens * 2),
            nn.sigmoid  
        ])


        if self.embed_coords:
            self.coord_encoder = nn.Sequential([
                nn.Dense(256),
                nn.relu,
                nn.Dense(self.vision_dim)
            ])
        
        self.batch_grid_sample = jax.vmap(grid_sample_jax, in_axes=(0, 0))
        

    def __call__(self, feature_map):
        # use global feature to predict coordinates
        B, H, W, C = feature_map.shape
        
        
        global_feat = jnp.mean(feature_map, axis=(1, 2))  # [B, D]
        pred_coords = self.scout_mlp(global_feat).reshape(B, self.num_tokens, 2)
        sampled_feats = self.batch_grid_sample(feature_map, pred_coords)
    
    
        # Inject Coordinate Embedding
        if self.embed_coords:
            coord_emb = self.coord_encoder(pred_coords) 
            sampled_feats = sampled_feats + coord_emb

        
        return sampled_feats, pred_coords
    