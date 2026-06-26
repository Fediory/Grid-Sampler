import flax.linen as nn
import jax
import jax.numpy as jnp


def grid_sample_jax(image, coords):
    """Bilinear sampling on normalized [0, 1] coordinates."""
    height, width, _ = image.shape
    x = coords[:, 0] * (width - 1)
    y = coords[:, 1] * (height - 1)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.maximum(0, jnp.minimum(x0, width - 1))
    y0 = jnp.maximum(0, jnp.minimum(y0, height - 1))
    x1 = jnp.maximum(0, jnp.minimum(x0 + 1, width - 1))
    y1 = jnp.maximum(0, jnp.minimum(y0 + 1, height - 1))

    ia = image[y0, x0]
    ib = image[y1, x0]
    ic = image[y0, x1]
    id_ = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., None] * ia + wb[..., None] * ib + wc[..., None] * ic + wd[..., None] * id_


class ActiveTokenSampler(nn.Module):
    """Learned sampler that keeps a compact subset of image tokens."""

    vision_dim: int
    num_tokens: int = 16
    embed_coords: bool = True
    noise_scale: float = 0.1
    coordinate_margin: float = 0.0  # [0, 0.5). If >0, restricts coords from [0,1] to [m, 1-m].

    def setup(self):
        self.scout_mlp = nn.Sequential(
            [
                nn.Dense(512),
                nn.relu,
                nn.Dense(self.num_tokens * 2),
                nn.sigmoid,
            ]
        )
        # LayerNorm before scout_mlp prevents sigmoid saturation: global-pooled
        # features can have large variance across images, driving sigmoid outputs
        # to 0 or 1 and collapsing all token predictions to the same corner.
        self.input_norm = nn.LayerNorm()
        if self.embed_coords:
            self.coord_encoder = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(self.vision_dim)])
        self.batch_grid_sample = jax.vmap(grid_sample_jax, in_axes=(0, 0))

    def __call__(self, feature_map, *, noise=False, num_probes=1):
        batch_size, _, _, _ = feature_map.shape
        global_feat = jnp.mean(feature_map, axis=(1, 2))
        # Normalize before scout_mlp to keep sigmoid inputs in a reasonable range.
        global_feat = self.input_norm(global_feat)

        def _single_probe(rng):
            perturb = jax.random.normal(rng, global_feat.shape) * self.noise_scale
            return self.scout_mlp(global_feat + perturb).reshape(batch_size, self.num_tokens, 2)

        if noise:
            rngs = jax.random.split(jax.random.PRNGKey(42), num_probes)
            pred_coords = jnp.mean(jax.vmap(_single_probe)(rngs), axis=0)
        else:
            pred_coords = self.scout_mlp(global_feat).reshape(batch_size, self.num_tokens, 2)

        # Apply coordinate margin: squash [0,1] → [margin, 1-margin]
        if self.coordinate_margin > 0:
            m = self.coordinate_margin
            pred_coords = m + (1.0 - 2.0 * m) * pred_coords

        sampled_feats = self.batch_grid_sample(feature_map, pred_coords)
        if self.embed_coords:
            sampled_feats = sampled_feats + self.coord_encoder(pred_coords)
        return sampled_feats, pred_coords
