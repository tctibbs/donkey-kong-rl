"""Checkpoint evaluation and random baseline under the Machado et al. protocol."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from dkrl.agents.ppo import PPOAgent
from dkrl.config import EvalConfig, ExperimentConfig
from dkrl.envs import EpisodeStats, VectorEnv, make_eval_envs


def summarize(returns: list[float], lengths: list[int]) -> dict:
    """Compute summary statistics over per-episode returns.

    Args:
        returns: Raw episode returns.
        lengths: Episode lengths in agent steps.

    Returns:
        Dict with episode count, mean, std, median, min, max, and mean length.
    """
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": len(returns),
        "return_mean": float(arr.mean()),
        "return_std": float(arr.std()),
        "return_median": float(np.median(arr)),
        "return_min": float(arr.min()),
        "return_max": float(arr.max()),
        "length_mean": float(np.mean(lengths)),
        "returns": [float(r) for r in returns],
    }


def _rollout_episodes(
    envs: VectorEnv,
    num_envs: int,
    episodes: int,
    act: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[float], list[int]]:
    """Step vectorized envs until `episodes` episodes complete.

    Args:
        envs: EnvPool gymnasium-API environment.
        num_envs: Number of parallel envs in the pool.
        episodes: Number of completed episodes to collect.
        act: Callable mapping a batch of observations to a batch of actions.

    Returns:
        Tuple of (returns, lengths) lists, truncated to `episodes` entries.
    """
    stats = EpisodeStats(num_envs)
    returns: list[float] = []
    lengths: list[int] = []
    obs, _ = envs.reset()
    while len(returns) < episodes:
        actions = act(obs)
        obs, _, terminated, truncated, info = envs.step(actions)
        for ep_return, ep_length in stats.update(terminated, truncated, info):
            returns.append(ep_return)
            lengths.append(ep_length)
    return returns[:episodes], lengths[:episodes]


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int | None = None,
    seed: int | None = None,
    device_str: str | None = None,
) -> dict:
    """Evaluate a trained checkpoint under the locked evaluation protocol.

    The protocol comes from the EvalConfig stored in the checkpoint's config:
    sticky actions, full action set, no episodic-life, raw rewards, 108k-frame
    cap. Actions are sampled from the policy.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint written by training.
        episodes: Override for the number of evaluation episodes.
        seed: Override for the evaluation seed.
        device_str: Override device, for example ``cpu``.

    Returns:
        Summary statistics dict, including checkpoint provenance.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    eval_cfg = cfg.eval
    n_episodes = episodes if episodes is not None else eval_cfg.episodes
    eval_seed = seed if seed is not None else eval_cfg.seed

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent = PPOAgent(ckpt["n_actions"], cfg.env.frame_stack).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    envs = make_eval_envs(
        eval_cfg, cfg.env.env_id, cfg.env.img_size, cfg.env.frame_stack, eval_seed
    )

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).to(device))
        return action.cpu().numpy()

    logger.info(f"evaluating {checkpoint_path} for {n_episodes} episodes (seed={eval_seed})")
    returns, lengths = _rollout_episodes(envs, eval_cfg.num_envs, n_episodes, act)
    envs.close()

    result = summarize(returns, lengths)
    result.update(
        {
            "kind": "eval",
            "checkpoint": str(checkpoint_path),
            "run_id": ckpt.get("run_id"),
            "config_hash": ckpt.get("config_hash"),
            "git_sha_train": ckpt.get("git_sha"),
            "eval_seed": eval_seed,
            "protocol": eval_cfg.model_dump(),
        }
    )
    logger.info(
        f"eval done: mean={result['return_mean']:.1f} std={result['return_std']:.1f} "
        f"median={result['return_median']:.1f} max={result['return_max']:.0f}"
    )
    return result


def random_baseline(
    eval_cfg: EvalConfig,
    env_id: str = "DonkeyKong-v5",
    episodes: int | None = None,
    seed: int | None = None,
) -> dict:
    """Measure the random-policy baseline under the evaluation protocol.

    Args:
        eval_cfg: Evaluation protocol configuration.
        env_id: EnvPool task id.
        episodes: Override for the number of episodes.
        seed: Override for the seed.

    Returns:
        Summary statistics dict.
    """
    n_episodes = episodes if episodes is not None else eval_cfg.episodes
    eval_seed = seed if seed is not None else eval_cfg.seed
    envs = make_eval_envs(eval_cfg, env_id, 84, 4, eval_seed)
    n_actions = int(envs.action_space.n)
    rng = np.random.default_rng(eval_seed)

    def act(obs: np.ndarray) -> np.ndarray:
        return rng.integers(0, n_actions, size=eval_cfg.num_envs)

    logger.info(f"random baseline on {env_id}: {n_episodes} episodes (seed={eval_seed})")
    returns, lengths = _rollout_episodes(envs, eval_cfg.num_envs, n_episodes, act)
    envs.close()

    result = summarize(returns, lengths)
    result.update(
        {
            "kind": "random_baseline",
            "env_id": env_id,
            "eval_seed": eval_seed,
            "protocol": eval_cfg.model_dump(),
        }
    )
    logger.info(
        f"random baseline: mean={result['return_mean']:.1f} std={result['return_std']:.1f} "
        f"max={result['return_max']:.0f}"
    )
    return result
