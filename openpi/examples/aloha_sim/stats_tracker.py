import logging
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class StatsTracker(_subscriber.Subscriber):
    """Tracks episode statistics like success rate and average reward."""

    def __init__(self, success_threshold: float = 0.5) -> None:
        """
        Args:
            success_threshold: Reward threshold to consider an episode successful.
        """
        self._success_threshold = success_threshold
        self._episode_rewards: list[float] = []
        self._current_reward: float = 0.0
        self._num_successes: int = 0

    @override
    def on_episode_start(self) -> None:
        self._current_reward = 0.0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        # Reward tracking happens at episode end via environment state
        pass

    @override
    def on_episode_end(self) -> None:
        # Get reward from observation (latest reward)
        # For now, we'll use a simple approach - this will be updated
        # by the environment during step execution
        pass

    def set_episode_reward(self, reward: float) -> None:
        """Called by the environment to set the episode reward."""
        self._current_reward = reward
        self._episode_rewards.append(reward)
        
        if reward >= self._success_threshold:
            self._num_successes += 1
            logging.info(f"Episode success! Reward: {reward:.4f}")
        else:
            logging.info(f"Episode failed. Reward: {reward:.4f}")

    @property
    def num_successes(self) -> int:
        """Number of successful episodes."""
        return self._num_successes

    @property
    def total_episodes(self) -> int:
        """Total number of episodes completed."""
        return len(self._episode_rewards)

    def get_success_rate(self) -> float:
        """Get the success rate (0.0 to 1.0)."""
        if self.total_episodes == 0:
            return 0.0
        return self._num_successes / self.total_episodes

    def get_average_reward(self) -> float:
        """Get the average reward across all episodes."""
        if self.total_episodes == 0:
            return 0.0
        return sum(self._episode_rewards) / self.total_episodes
