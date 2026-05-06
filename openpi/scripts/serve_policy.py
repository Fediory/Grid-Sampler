import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO_PI0 = "libero_pi0"
    LIBERO_PI0_GRID = "libero_pi0_grid"
    LIBERO_PI0_GRID8 = "libero_pi0_grid8"
    LIBERO_PI0_GRID4 = "libero_pi0_grid4"
    LIBERO_PI05 = "libero_pi05"
    LIBERO_PI05_BASE = "libero_base"
    # LIBERO_PI05_LORA = "libero_lora"
    LIBERO_PI05_GRID32 = "libero_grid32"
    LIBERO_PI05_GRID = "libero_grid"
    LIBERO_PI05_GRID12 = "libero_grid12"
    LIBERO_PI05_GRID10 = "libero_grid10"
    LIBERO_PI05_GRID8 = "libero_grid8"
    LIBERO_PI05_GRID6 = "libero_grid6"
    LIBERO_PI05_GRID4 = "libero_grid4"
    LIBERO_PI05_GRID1 = "libero_grid1"
    # LIBERO_PI05_GRID_LORA = "libero_grid_lora"
    
    ALOHA_PI0_TRANSFER_CUBE_SCRIPT = "aloha_pi0_transfer_cube_script"
    ALOHA_PI0_TRANSFER_CUBE_HUMAN = "aloha_pi0_transfer_cube_human"
    ALOHA_PI0_INSERTION_SCRIPT = "aloha_pi0_insertion_script"
    ALOHA_PI0_INSERTION_HUMAN = "aloha_pi0_insertion_human"
    ALOHA_PI0_GRID_TRANSFER_CUBE_HUMAN  = "aloha_pi0_grid_transfer_cube_human"
    ALOHA_PI0_GRID_TRANSFER_CUBE_SCRIPT = "aloha_pi0_grid_transfer_cube_script"
    ALOHA_PI0_GRID_INSERTION_HUMAN = "aloha_pi0_grid_insertion_human"
    ALOHA_PI0_GRID_INSERTION_SCRIPT = "aloha_pi0_grid_insertion_script"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8888
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.ALOHA_PI0_TRANSFER_CUBE_SCRIPT: Checkpoint(
        config="pi0_aloha_sim",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim/pi0_aloha_transfer_cube_scripted/14999",
    ),
    EnvMode.ALOHA_PI0_TRANSFER_CUBE_HUMAN: Checkpoint(
        config="pi0_aloha_sim",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim/pi0_aloha_transfer_cube_human/14999",
    ),
    EnvMode.ALOHA_PI0_INSERTION_SCRIPT: Checkpoint(
        config="pi0_aloha_sim",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim/pi0_aloha_insertion_scripted/14999",
    ),
    EnvMode.ALOHA_PI0_INSERTION_HUMAN: Checkpoint(
        config="pi0_aloha_sim",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim/pi0_aloha_insertion_human/14999",
    ),
    EnvMode.ALOHA_PI0_GRID_TRANSFER_CUBE_SCRIPT: Checkpoint(
        config="pi0_aloha_sim_grid",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim_grid/pi0_aloha_grid_transfer_cube_scripted/14999",
    ),
    EnvMode.ALOHA_PI0_GRID_TRANSFER_CUBE_HUMAN: Checkpoint(
        config="pi0_aloha_sim_grid",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim_grid/pi0_aloha_grid_transfer_cube_human/14999",
    ),
    EnvMode.ALOHA_PI0_GRID_INSERTION_SCRIPT: Checkpoint(
        config="pi0_aloha_sim_grid",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim_grid/pi0_aloha_grid_insertion_scripted/14999",
    ),
    EnvMode.ALOHA_PI0_GRID_INSERTION_HUMAN: Checkpoint(
        config="pi0_aloha_sim_grid",
        dir="/hdd/fediory/checkpoints/pi0_aloha_sim_grid/pi0_aloha_grid_insertion_human/14999",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO_PI0: Checkpoint(
        config="pi0_libero",
        dir="gs://openpi-assets/checkpoints/pi0_libero",
    ),
    EnvMode.LIBERO_PI0_GRID: Checkpoint(
        config="pi0_libero_grid",
        dir="/hdd/fediory/checkpoints/pi0_libero_grid/pi0_libero_grid_max/29999",
    ),
    EnvMode.LIBERO_PI0_GRID8: Checkpoint(
        config="pi0_libero_grid8",
        dir="/hdd/fediory/checkpoints/pi0_libero_grid8/pi0_libero_grid8/29999",
    ),
    EnvMode.LIBERO_PI0_GRID4: Checkpoint(
        config="pi0_libero_grid4",
        dir="/hdd/fediory/checkpoints/pi0_libero_grid4/pi0_libero_grid4/29999",
    ),
    EnvMode.LIBERO_PI05: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.LIBERO_PI05_BASE: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    # EnvMode.LIBERO_PI05_LORA: Checkpoint(
    #     config="pi05_libero_lora",
    #     dir="/hdd/fediory/checkpoints/pi05_libero_lora/pi05_libero_lora/5000",
    # ),
    EnvMode.LIBERO_PI05_GRID32: Checkpoint(
        config="pi05_libero_grid32",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid32/pi05_libero_grid32/29999",
    ),
    EnvMode.LIBERO_PI05_GRID: Checkpoint(
        config="pi05_libero_grid",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid/pi05_libero_grid/29999",
    ),
    EnvMode.LIBERO_PI05_GRID12: Checkpoint(
        config="pi05_libero_grid12",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid12/pi05_libero_grid12/29999",
    ),
    EnvMode.LIBERO_PI05_GRID10: Checkpoint(
        config="pi05_libero_grid10",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid10/pi05_libero_grid10/29999",
    ),
    EnvMode.LIBERO_PI05_GRID8: Checkpoint(
        config="pi05_libero_grid8",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid8/pi05_libero_grid8/29999",
    ),
    EnvMode.LIBERO_PI05_GRID6: Checkpoint(
        config="pi05_libero_grid6",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid6/pi05_libero_grid6/29999",
    ),
    EnvMode.LIBERO_PI05_GRID4: Checkpoint(
        config="pi05_libero_grid4",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid4/pi05_libero_grid4/29999",
    ),
    EnvMode.LIBERO_PI05_GRID1: Checkpoint(
        config="pi05_libero_grid1",
        dir="/hdd/fediory/checkpoints/pi05_libero_grid1/pi05_libero_grid1/29999",
    ),
    # EnvMode.LIBERO_PI05_GRID_LORA: Checkpoint(
    #     config="pi05_libero_grid_lora",
    #     dir="/hdd/fediory/checkpoints/pi05_libero_grid_lora/pi05_libero_grid_lora/29999",
    # ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
