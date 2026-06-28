"""Search-based DAgger: relabel the policy's own visited states with a
Go-Explore search expert.

One round: roll the current policy in the stochastic env, snapshot emulator
states along the way, then from each snapshot run a short archive-based search
to find the best continuation and record its first action as the expert label.
The aggregated (observation, expert-action) pairs train the next policy by
behavior cloning, so the policy learns to recover from the states it actually
reaches, not just replay one open-loop trajectory.
"""

from pathlib import Path
from typing import Any, cast

import ale_py
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from loguru import logger

from dkrl.explore import _sticky_effective

gym.register_envs(ale_py)

# Depth-biased action pool for search expansion (climb, move, jump).
_ACTION_POOL = [2] * 6 + [3] * 3 + [4] * 3 + [1, 11, 12] * 2 + [0]

RAM_MARIO_Y = 27
RAM_BOARD = 16
_BOTTOM_Y = 154


def _progress_potential(ram: np.ndarray) -> float:
    """Dense progress signal so short rollouts can score sparse-reward states.

    Combines height climbed (lower Mario y is higher on the screen) with the
    board index (each completed two-screen loop advances the board). This
    gives the search expert a gradient toward climbing and screen progress
    even when no game points are scored within the short horizon.
    """
    height = float(_BOTTOM_Y - int(ram[RAM_MARIO_Y]))
    board = float(int(ram[RAM_BOARD]))
    return 2.0 * height + 4000.0 * board


def _make_env() -> tuple[gym.Env, Any]:
    """Build the training-identical preprocessing stack (frameskip 4)."""
    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
    )
    env = AtariPreprocessing(base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    return env, cast(Any, base.unwrapped).ale


def _search_first_action(
    env: gym.Env,
    ale: Any,  # noqa: ANN401
    root_state: Any,  # noqa: ANN401
    n_actions: int,
    rng: np.random.Generator,
    rollouts: int = 24,
    horizon: int = 60,
) -> int:
    """Return the first action of the best short continuation from a state.

    Random-shooting MPC: for each candidate first action, roll several
    depth-biased random continuations from the restored state and keep the
    action whose continuations score highest. The deterministic sticky rule is
    applied so the labels reflect the perturbed dynamics the policy faces.

    Args:
        env: Wrapped env (its ALE is restored each rollout).
        ale: The underlying ALE.
        root_state: System state to restore before each rollout.
        n_actions: Action-space size.
        rng: RNG for continuations.
        rollouts: Continuations per candidate first action.
        horizon: Steps per continuation.

    Returns:
        The best first action (ALE full-action-set index).
    """
    candidates = sorted(set(_ACTION_POOL))
    best_action = 0
    best_score = -1e9
    for first in candidates:
        total = 0.0
        for _ in range(rollouts):
            env.reset()
            ale.restoreSystemState(root_state)
            score = 0.0
            p_eff = first
            hold = int(rng.integers(2, 6))
            cur = first
            terminated = truncated = False
            for step in range(horizon):
                if step >= hold:
                    cur = int(_ACTION_POOL[rng.integers(0, len(_ACTION_POOL))])
                    hold = step + int(rng.integers(2, 6))
                eff = _sticky_effective(cur, p_eff, step, 0.25)
                _, reward, terminated, truncated, _ = env.step(eff)
                p_eff = eff
                score += float(reward)
                if terminated or truncated:
                    break
            # Dense terminal progress potential, minus a death penalty.
            score += _progress_potential(ale.getRAM())
            if terminated:
                score -= 4000.0
            total += score
        mean = total / rollouts
        if mean > best_score:
            best_score = mean
            best_action = first
    return best_action


def dagger_collect(
    checkpoint_path: Path,
    out_path: Path,
    states: int = 120,
    seed: int = 20000,
    snapshot_every: int = 30,
) -> dict:
    """Roll the policy, snapshot visited states, and relabel them via search.

    Args:
        checkpoint_path: Policy checkpoint to roll out (feedforward).
        out_path: Output ``.npz`` of (observations, expert-actions).
        states: Number of visited states to relabel.
        seed: Base seed.
        snapshot_every: Snapshot one state every N agent steps along rollouts.

    Returns:
        Dict with counts and label distribution.
    """
    from dkrl.config import ExperimentConfig
    from dkrl.evaluate import load_policy

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, cfg, device)
    n_actions = ckpt["n_actions"]

    env, ale = _make_env()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    collected_obs: list[np.ndarray] = []
    collected_states: list[Any] = []
    obs, _ = env.reset(seed=seed)
    step = 0
    # Phase 1: roll the policy in the (deterministic-sticky) env, snapshotting.
    while len(collected_states) < states:
        if step % snapshot_every == 0:
            collected_obs.append(np.asarray(obs, dtype=np.uint8))
            collected_states.append(ale.cloneSystemState())
        with torch.no_grad():
            action = int(policy(torch.from_numpy(np.asarray(obs)[None]).to(device)).item())
        eff = _sticky_effective(action, action, step, 0.25)
        obs, _, terminated, truncated, _ = env.step(eff)
        step += 1
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step)
            step = 0

    # Phase 2: relabel each snapshot with the search expert.
    actions = []
    for i, st in enumerate(collected_states):
        a = _search_first_action(env, ale, st, n_actions, rng)
        actions.append(a)
        if (i + 1) % 30 == 0:
            logger.info(f"relabeled {i + 1}/{len(collected_states)}")
    env.close()

    observations = np.stack(collected_obs[: len(actions)])
    actions_arr = np.asarray(actions, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, observations=observations, actions=actions_arr)
    uniq, counts = np.unique(actions_arr, return_counts=True)
    result = {
        "states": len(actions),
        "samples": int(len(actions_arr)),
        "label_counts": {int(k): int(v) for k, v in zip(uniq, counts, strict=True)},
        "path": str(out_path),
    }
    logger.info(f"dagger collect: {result}")
    return result


def merge_demo_sets(paths: list[Path], out_path: Path) -> dict:
    """Concatenate several (observations, actions) npz datasets into one."""
    obs_list = []
    act_list = []
    for p in paths:
        d = np.load(p)
        obs_list.append(d["observations"])
        act_list.append(d["actions"])
    observations = np.concatenate(obs_list)
    actions = np.concatenate(act_list)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, observations=observations, actions=actions)
    return {"samples": int(len(actions)), "sources": len(paths), "path": str(out_path)}
