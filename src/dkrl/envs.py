"""EnvPool environment construction and episode statistics tracking."""

from typing import Protocol

import envpool
import gymnasium
import numpy as np

from dkrl.config import EnvConfig, EvalConfig


class VectorEnv(Protocol):
    """Minimal protocol for EnvPool's vectorized gymnasium-style API."""

    action_space: gymnasium.spaces.Discrete

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset all envs, returning batched observations and info."""
        ...

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all envs, returning (obs, reward, terminated, truncated, info)."""
        ...

    def close(self) -> None:
        """Release env resources."""
        ...


def make_train_envs(cfg: EnvConfig, seed: int) -> VectorEnv:
    """Build the vectorized EnvPool training environment.

    Args:
        cfg: Environment configuration.
        seed: Base seed for the pool.

    Returns:
        An EnvPool gymnasium-API vectorized environment.
    """
    return envpool.make(
        cfg.env_id,
        env_type="gymnasium",
        num_envs=cfg.num_envs,
        num_threads=cfg.num_threads,
        seed=seed,
        max_episode_steps=cfg.max_episode_steps,
        img_height=cfg.img_size,
        img_width=cfg.img_size,
        stack_num=cfg.frame_stack,
        frame_skip=cfg.frame_skip,
        noop_max=cfg.noop_max,
        episodic_life=cfg.episodic_life,
        reward_clip=cfg.reward_transform == "clip",
        repeat_action_probability=cfg.sticky_action_prob,
        full_action_space=cfg.full_action_space,
        use_fire_reset=cfg.fire_reset,
    )


def make_eval_envs(
    cfg: EvalConfig,
    env_id: str,
    img_size: int,
    frame_stack: int,
    seed: int,
    full_action_space: bool | None = None,
) -> VectorEnv:
    """Build the evaluation environment under the Machado et al. protocol.

    No episodic-life trick, raw rewards, sticky actions, full action set,
    108k-frame cap.

    Args:
        cfg: Evaluation protocol configuration.
        env_id: EnvPool task id, for example ``DonkeyKong-v5``.
        img_size: Observation height and width, matching training.
        frame_stack: Number of stacked frames, matching training.
        seed: Base seed for the pool.
        full_action_space: Override for the action set. A policy trained on
            the minimal set must be evaluated on it (action indices map to
            different buttons in each set); such a run is a recorded protocol
            deviation. ``None`` uses the protocol default from ``cfg``.

    Returns:
        An EnvPool gymnasium-API vectorized environment.
    """
    if full_action_space is None:
        full_action_space = cfg.full_action_space
    return envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=cfg.num_envs,
        seed=seed,
        max_episode_steps=cfg.max_episode_steps,
        img_height=img_size,
        img_width=img_size,
        stack_num=frame_stack,
        noop_max=cfg.noop_max,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=cfg.sticky_action_prob,
        full_action_space=full_action_space,
    )


class EpisodeStats:
    """Track true episode returns and lengths from EnvPool info dicts.

    EnvPool reports the raw (unclipped) per-step reward in ``info["reward"]``
    and marks real game-over in ``info["terminated"]``, which differs from the
    step's ``terminated`` array when ``episodic_life`` is on. This tracker
    accumulates raw returns and reports an episode only on real termination
    or truncation.
    """

    def __init__(self, num_envs: int) -> None:
        """Initialize accumulators for ``num_envs`` parallel environments."""
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.max_step_rewards = np.zeros(num_envs, dtype=np.float64)

    def update(
        self,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict,
    ) -> list[tuple[float, int, float]]:
        """Accumulate one vectorized step and return completed episodes.

        Args:
            terminated: Per-env terminated flags from the step.
            truncated: Per-env truncated flags from the step.
            info: EnvPool info dict containing ``reward`` and ``terminated``.

        Returns:
            List of ``(episode_return, episode_length, max_step_reward)`` for
            episodes that really ended this step. The max single-step raw
            reward drives the screen-clear detector (ADR 0002).
        """
        self.returns += info["reward"]
        self.lengths += 1
        self.max_step_rewards = np.maximum(self.max_step_rewards, info["reward"])
        real_done = np.logical_or(info.get("terminated", terminated), truncated)
        completed = [
            (float(self.returns[i]), int(self.lengths[i]), float(self.max_step_rewards[i]))
            for i in np.nonzero(real_done)[0]
        ]
        self.returns[real_done] = 0.0
        self.lengths[real_done] = 0
        self.max_step_rewards[real_done] = 0.0
        return completed
