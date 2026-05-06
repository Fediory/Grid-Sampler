import dataclasses
import logging
import pathlib
import time

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import stats_tracker as _stats_tracker
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0" # AlohaInsertion, AlohaTransferCube
    seed: int = 1000

    action_horizon: int = 25

    host: str = "0.0.0.0"
    port: int = 8888

    display: bool = False


def main(args: Args) -> None:
    start_time = time.time()
    
    environment = _env.AlohaSimEnvironment(
        task=args.task,
        seed=args.seed,
    )
    
    stats_tracker = _stats_tracker.StatsTracker()
    
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
            stats_tracker,
        ],
        max_hz=50,
        num_episodes=97,
        max_episode_steps=300
    )

    # Monkey-patch the runtime to track episode success
    original_run_episode = runtime._run_episode
    def tracked_run_episode():
        original_run_episode()
        stats_tracker.set_episode_reward(environment._episode_reward)
    
    runtime._run_episode = tracked_run_episode
    runtime.run()
    
    # Log summary statistics
    total_time = time.time() - start_time
    logging.info(f"Task: {args.task}")
    logging.info(f"Total episodes: {runtime._num_episodes}")
    logging.info(f"Successful episodes: {stats_tracker.num_successes}")
    logging.info(f"Success rate: {stats_tracker.get_success_rate() * 100:.1f}%")
    logging.info(f"Average reward: {stats_tracker.get_average_reward():.4f}")
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    logging.info(f"Videos saved to: {args.out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
