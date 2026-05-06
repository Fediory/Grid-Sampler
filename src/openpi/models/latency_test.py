import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.models.pi0 import Pi0
import logging
from flax import struct
from typing import Dict


@struct.dataclass
class MockObservation:
    images: Dict[str, jax.Array]
    image_masks: Dict[str, jax.Array]
    state: jax.Array
    tokenized_prompt: jax.Array
    tokenized_prompt_mask: jax.Array
    token_ar_mask: jax.Array    
    token_loss_mask: jax.Array  

def generate_dummy_observation(batch_size, config):
    img_size = getattr(config, "image_size", 224) 
    prompt_len = getattr(config, "max_token_len", 32)
    state_dim = getattr(config, "action_dim", 6)


    images = {
        "base_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
        "left_wrist_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
        "right_wrist_0_rgb": jnp.zeros((batch_size, img_size, img_size, 3)),
    }
    
    image_masks = {
        "base_0_rgb": jnp.ones((batch_size,), dtype=bool),
        "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=bool),
        "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=bool),
    }


    return MockObservation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((batch_size, state_dim)),
        

        tokenized_prompt=jnp.zeros((batch_size, prompt_len), dtype=jnp.int32), 
        tokenized_prompt_mask=jnp.ones((batch_size, prompt_len), dtype=bool),
        token_ar_mask=jnp.zeros((batch_size, prompt_len), dtype=bool),
        token_loss_mask=jnp.zeros((batch_size, prompt_len), dtype=bool) 
    )

def benchmark():
    print(">>> Initialization...")
    config = model=pi0_config.Pi0Config(
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
    
    batch_size = 2
    obs = generate_dummy_observation(batch_size, config)
    rng = jax.random.PRNGKey(0)

    @nnx.jit
    def inference_step(model, rng, obs):
        return model.sample_actions(rng, obs, num_steps=1)

    print(">>> Warm-up...")
    start_compile = time.time()
    _ = inference_step(model, rng, obs).block_until_ready()
    print(f"Warm-up time: {time.time() - start_compile:.4f} seconds")

    num_runs = 50
    print(f">>> Start testing in ({num_runs} loops)...")
    
    latencies = []
    for i in range(num_runs):
        rng, step_rng = jax.random.split(rng)
        
        t0 = time.time()
        actions = inference_step(model, step_rng, obs)
        actions.block_until_ready()
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: {latencies[-1]:.2f} ms")

    avg_latency = sum(latencies) / len(latencies)
    print("-" * 30)
    print(f"Batch Size: {batch_size}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Minimum Latency: {min(latencies):.2f} ms")
    print(f"Maximum Latency: {max(latencies):.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    benchmark()