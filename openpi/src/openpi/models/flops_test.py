import jax
import jax.numpy as jnp
from flax import struct
import flax.nnx as nnx
from typing import Dict
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0


@struct.dataclass
class MockObservation:
    images: Dict[str, jax.Array]
    image_masks: Dict[str, jax.Array]
    state: jax.Array
    tokenized_prompt: jax.Array
    tokenized_prompt_mask: jax.Array
    token_ar_mask: jax.Array
    token_loss_mask: jax.Array

def generate_dummy_inputs(batch_size, config):
    img_size = getattr(config, "image_size", 224)
    prompt_len = getattr(config, "max_token_len", 32)
    state_dim = getattr(config, "action_dim", 6)
    action_horizon = getattr(config, "action_horizon", 10)

    images = {
        "base_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
        "left_wrist_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
        "right_wrist_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
    }
    image_masks = {k: jnp.ones((batch_size,), dtype=bool) for k in images}
    
    obs = MockObservation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((batch_size, state_dim)),
        tokenized_prompt=jnp.zeros((batch_size, prompt_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, prompt_len), dtype=bool),
        token_ar_mask=jnp.zeros((batch_size, prompt_len), dtype=jnp.int32),
        token_loss_mask=jnp.zeros((batch_size, prompt_len), dtype=bool)
    )

    actions = jnp.zeros((batch_size, action_horizon, state_dim))
    return obs, actions

def print_flops(name, compiled_fn):
    flops = "N/A (CPU not supported)"
    try:
        # Try to get Cost Analysis
        analysis = compiled_fn.cost_analysis()
        if analysis and isinstance(analysis, list):
            flops_val = analysis[0].get('flops', 0)
        elif analysis and isinstance(analysis, dict):
            flops_val = analysis.get('flops', 0)
        else:
            flops_val = 0
            
        if flops_val > 0:
            flops = f"{flops_val / 1e9:.2f} GFLOPs"
        else:
            flops = "Unable to count (possibly JAX version or Control Flow issue)"
    except Exception as e:
        flops = f"Unable to obtain ({e})"
    
    print(f"    FLOPs ({name}): {flops}")


def profile():
    print(">>> Initialization...")
    config = pi0_config.Pi0Config(
            pi05=True, 
            action_horizon=10, 
            discrete_state_input=False, 
            grid=True, 
            use_fps=False, 
            num_token_samples=16, 
            num_probes=1,
            target_k=16
        )
    rngs = nnx.Rngs(0)
    model = Pi0(config, rngs)
    
    state = nnx.state(model)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\n>>> Params: {total_params / 1e6:.2f} Million (M)")

    # Prepare data
    batch_size = 2
    obs, actions = generate_dummy_inputs(batch_size, config)
    rng = jax.random.PRNGKey(0)

    graph_def, state = nnx.split(model)

    # -----------------------------------------------------
    # Test 1: Training forward pass (Compute Loss)
    # -----------------------------------------------------
    print(f"\n>>> Analysis 1: Training forward pass (compute_loss)")

    # Define a pure function: input state -> output loss
    def pure_train_step(graph_def, state, rng, obs, actions):
        # Re-merge the model
        model = nnx.merge(graph_def, state)
        # Execute computation
        loss = model.compute_loss(rng, obs, actions, train=True)
        return loss

    print("  -> Compiling (Lowering)...")
    lowered_train = jax.jit(pure_train_step).lower(graph_def, state, rng, obs, actions)
    compiled_train = lowered_train.compile()
    
    print_flops("Training", compiled_train)

    # -----------------------------------------------------
    # Test 2: Inference generation (Sample Actions)
    # -----------------------------------------------------
    print(f"\n>>> Analysis 2: Inference generation (sample_actions)")

    def pure_infer_step(graph_def, state, rng, obs):
        model = nnx.merge(graph_def, state)
        return model.sample_actions(rng, obs, num_steps=10)

    print("  -> Compiling (Lowering)...")
    lowered_infer = jax.jit(pure_infer_step).lower(graph_def, state, rng, obs)
    compiled_infer = lowered_infer.compile()
    
    print_flops("Inference", compiled_infer)

if __name__ == "__main__":
    profile()