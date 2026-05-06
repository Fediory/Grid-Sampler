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


def grid_sample_nearest(image, coords):
    """Nearest-neighbor interpolation for grid sampling.
    
    Args:
        image: [H, W, C] feature map
        coords: [K, 2] normalized coordinates in [0, 1]
    
    Returns:
        sampled: [K, C] sampled features
    """
    H, W, C = image.shape
    x = coords[:, 0] * (W - 1)
    y = coords[:, 1] * (H - 1)
    
    # Round to nearest integer
    x_idx = jnp.round(x).astype(jnp.int32)
    y_idx = jnp.round(y).astype(jnp.int32)
    
    # Clamp to valid range
    x_idx = jnp.maximum(0, jnp.minimum(x_idx, W - 1))
    y_idx = jnp.maximum(0, jnp.minimum(y_idx, H - 1))
    
    # Direct nearest-neighbor indexing
    sampled = image[y_idx, x_idx]  # [K, C]
    
    return sampled



class ActiveTokenSampler(nn.Module):
    vision_dim: int       
    num_tokens: int = 16  
    embed_coords: bool = True 
    noise_scale: float = 0.1
    # When True, log predicted coords via jax.debug.callback (JIT-safe). Also prints and appends to file if path is set.
    log_pred_coords: bool = False
    # Append mode; empty string skips file output (stdout/logging only).
    pred_coords_log_path: str = "active_token_pred_coords.log"

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
        

    def __call__(self, feature_map, noise=False):
        # use global feature to predict coordinates
        B, H, W, C = feature_map.shape
        
        
        # default
        global_feat = jnp.mean(feature_map, axis=(1, 2))  # [B, D]

        
        # Generate Coordinate
        def single_probe(rng):
            noise = jax.random.normal(rng, global_feat.shape) * self.noise_scale
            return self.scout_mlp(global_feat+noise).reshape(B, self.num_tokens, 2)
            
        if noise:
            noise_rng = jax.random.PRNGKey(42)
            num_probes = 1 # We select the same tokens across the batch to maintain some consistency, but you can modify this if needed.
            rngs = jax.random.split(noise_rng, num_probes)
            candidates = jax.vmap(single_probe)(rngs)
            pred_coords = jnp.mean(candidates, axis=0)  # [b, n, c]
        else:
            pred_coords = self.scout_mlp(global_feat).reshape(B, self.num_tokens, 2)

        
        # Grid Sample
        if self.log_pred_coords:
            log_path = self.pred_coords_log_path

            def _log_pred_coords(coords_host):
                import numpy as np

                arr = np.asarray(coords_host)
                msg = f"[ActiveTokenSampler] pred_coords shape={arr.shape} dtype={arr.dtype}\n{arr}\n"
                print(msg, end="", flush=True)
                _logger.info("%s", msg.rstrip())
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(msg)

            jax.debug.callback(_log_pred_coords, pred_coords)

        sampled_feats = self.batch_grid_sample(feature_map, pred_coords)
        
        # Inject Coordinate Embedding
        if self.embed_coords:
            coord_emb = self.coord_encoder(pred_coords) 
            sampled_feats = sampled_feats + coord_emb
            
        # Visualization
        # self.debug_similarity(feature_map, sampled_feats)
        # self.debug_retention(feature_map, sampled_feats)
        
        return sampled_feats, pred_coords
    
    def debug_similarity(self, feature_map, sampled_feats, filename="token_sim.png"):
        """
        Visualize self-similarity of original vs sampled tokens.

        Uses jax.debug.callback so it works under JIT compilation.
        """
        def _plot_callback(orig_feat, samp_feat):
            # Imports run on host
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 1. Prepare data: orig_feat [H, W, C] -> flatten -> [N, C]
            H, W, C = orig_feat.shape
            orig_flat = np.array(orig_feat).reshape(-1, C)
            samp_flat = np.array(samp_feat)
            
            # 2. Cosine similarity
            def cos_sim(x):
                norm = np.linalg.norm(x, axis=1, keepdims=True)
                x_norm = x / (norm + 1e-6)
                return np.dot(x_norm, x_norm.T)
            
            sim_orig = cos_sim(orig_flat)
            sim_samp = cos_sim(samp_flat)
            
            # 3. Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
            
            # Left: original
            im1 = axes[0].imshow(sim_orig, vmin=0, vmax=1, cmap='RdBu_r')
            axes[0].set_title(f"Original ({H}x{W}={H*W} Tokens)")
            axes[0].axis('off')
            
            # Right: sampled
            im2 = axes[1].imshow(sim_samp, vmin=0, vmax=1, cmap='RdBu_r')
            axes[1].set_title(f"Sampled ({samp_flat.shape[0]} Tokens)")
            axes[1].axis('off')
            
            plt.colorbar(im2, ax=axes, shrink=0.8)
            plt.suptitle("Token Self-Similarity Comparison")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)
            print(f"[Debug] Similarity plot saved to {filename}")

        # Visualize first batch item only [B, H, W, C] -> [H, W, C]
        # Avoids many files and does not affect the compute graph
        jax.debug.callback(_plot_callback, feature_map[0], sampled_feats[0])
    
    def debug_retention(self, feature_map, sampled_feats, filename="token_retention.png"):
        """
        Compute and plot information retention (coverage).

        Metric: max cosine similarity of each map token to the nearest sampled token.
        """
        def _retention_callback(orig_feat, samp_feat):
            import matplotlib.pyplot as plt
            import numpy as np
            
            # orig_feat: [H, W, C]
            # samp_feat: [K, C]
            H, W, C = orig_feat.shape
            orig_flat = np.array(orig_feat).reshape(-1, C)
            samp_flat = np.array(samp_feat)
            
            # 1. Normalize features
            orig_norm = orig_flat / (np.linalg.norm(orig_flat, axis=1, keepdims=True) + 1e-6)
            samp_norm = samp_flat / (np.linalg.norm(samp_flat, axis=1, keepdims=True) + 1e-6)
            
            # 2. Similarity matrix [N_orig, N_sampled]: rows = original tokens, cols = sampled
            sim_matrix = np.dot(orig_norm, samp_norm.T)
            
            # 3. Retention: per original token, max similarity to any sampled token
            # (how well the sample set covers that location)
            retention_map = np.max(sim_matrix, axis=1).reshape(H, W)
            mean_retention = np.mean(retention_map)
            
            # 4. Plot
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
            im = ax.imshow(retention_map, cmap='RdYlGn', vmin=0, vmax=1)
            
            ax.set_title(f"Information Retention Map\nMean Coverage Score: {mean_retention:.4f}")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Max Cos Similarity (Retention)')
            
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)
            
            print(f"[Debug] Retention Map saved to {filename}. Mean Score: {mean_retention:.4f}")

        # First batch item only
        jax.debug.callback(_retention_callback, feature_map[0], sampled_feats[0])