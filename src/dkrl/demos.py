"""Scripted RAM-based demonstrator and demonstration collection.

A rule-based climbing policy reads the Atari 2600 Donkey Kong RAM (byte map
in docs/research/2026-06-11-donkeykong-ram-map.md) and climbs the barrels
screen while jumping approaching barrels. Successful episodes (screen
cleared) are saved as (observation, action) pairs for behavior-cloning
warm-starts.

All coordinates are raw RAM units. Mario y grows downward.
"""

import pickle
from pathlib import Path
from typing import Any, cast

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from loguru import logger

gym.register_envs(ale_py)

NOOP, FIRE, UP, RIGHT, LEFT = 0, 1, 2, 3, 4
RIGHTFIRE, LEFTFIRE = 11, 12

RAM_MARIO_X = 19
RAM_MARIO_Y = 27
RAM_DIRECTION = 20
RAM_JUMP = 5
RAM_BOARD = 16
RAM_BARREL_Y = (0, 1, 2, 3)
RAM_BARREL_X = (65, 66, 67, 68)
RAM_GAMESTATE = 12
GAME_IN_PROGRESS_BIT = 0x80

LADDER_CHAIN = [
    (109, 154, 132),
    (81, 130, 101),
    (89, 101, 74),
    (69, 73, 46),
    (109, 43, 21),
    (77, 21, 5),
]
BARREL_SAME_FLOOR_DY = 10
BARREL_THREAT_DX = 12

RIVETS_FLOOR_BOTTOMS = (133, 105, 77, 49)
RIVETS_LADDER_RIGHT = 117
RIVETS_LADDER_LEFT = 41


class ScriptedClimber:
    """Rule-based climbing policy for the barrels screen, driven by RAM."""

    def __init__(self) -> None:
        """Initialize the start-press edge toggle."""
        self._fire_toggle = False

    def act(self, ram: np.ndarray) -> int:
        """Choose the next full-action-set action from the current RAM state.

        Args:
            ram: The 128-byte ALE RAM array.

        Returns:
            ALE full-action-set action index.
        """
        if not int(ram[RAM_GAMESTATE]) & GAME_IN_PROGRESS_BIT:
            self._fire_toggle = not self._fire_toggle
            return FIRE if self._fire_toggle else NOOP
        x = int(ram[RAM_MARIO_X])
        y = int(ram[RAM_MARIO_Y])
        jumping = 0 < int(ram[RAM_JUMP]) < 255

        if int(ram[RAM_BOARD]) != 0:
            return self._rivets_routine(ram, x, y)

        target = None
        for lx, bottom, top in LADDER_CHAIN:
            if y > top:
                target = (lx, bottom, top)
                break
        if target is None:
            return NOOP

        lx, _bottom, _top = target
        if 0 <= x - lx <= 2:
            return UP
        toward = RIGHT if x < lx else LEFT
        if jumping:
            return toward
        if self._barrel_threat(ram, x, y):
            return RIGHTFIRE if toward == RIGHT else LEFTFIRE
        return toward

    def _barrel_threat(self, ram: np.ndarray, x: int, y: int) -> bool:
        """Return True when any same-floor barrel is within jumping distance."""
        for yi, xi in zip(RAM_BARREL_Y, RAM_BARREL_X, strict=True):
            by = int(ram[yi])
            if by == 0 or abs(by - y) > BARREL_SAME_FLOOR_DY:
                continue
            if abs(int(ram[xi]) - x) <= BARREL_THREAT_DX:
                return True
        return False

    def _rivets_routine(self, ram: np.ndarray, x: int, y: int) -> int:
        """Zigzag the rivets screen: sweep each floor, climb at alternate ends.

        Walking a floor end to end removes its rivets (100 points each);
        removing all rivets completes the screen for the bonus. Floors sit at
        y 133, 105, 77, 49 with ladders at x 41, 61, 97, 117 on every floor.
        """
        floor_idx = None
        for i, fy in enumerate(RIVETS_FLOOR_BOTTOMS):
            if 0 <= fy - y <= 2 or (i == 0 and y >= fy):
                floor_idx = i
                break
        if floor_idx is None:
            return UP if y > 23 else NOOP

        ladder_x = RIVETS_LADDER_RIGHT if floor_idx % 2 == 0 else RIVETS_LADDER_LEFT
        if 0 <= x - ladder_x <= 2:
            return UP
        toward = RIGHT if x < ladder_x else LEFT
        if 0 < int(ram[RAM_JUMP]) < 255:
            return toward
        if self._barrel_threat(ram, x, y):
            return RIGHTFIRE if toward == RIGHT else LEFTFIRE
        return toward


def collect_demos(
    out_path: Path,
    episodes: int = 25,
    seed: int = 7000,
    max_steps: int = 4000,
    clear_threshold: float = 1500.0,
    sticky_action_prob: float = 0.0,
) -> dict:
    """Collect successful scripted episodes as a BC dataset.

    The env matches training observations (grayscale 84x84, frame skip 4,
    stack 4, full action set). Only episodes that clear the barrels screen
    (single step raw reward at or above ``clear_threshold``) are kept. The
    closed-loop script tolerates sticky actions (about 12/15 clears at 0.25),
    so demos can be collected distribution-matched to training.

    Args:
        out_path: Output ``.npz`` path for observations and actions.
        episodes: Number of SUCCESSFUL episodes to collect.
        seed: Base env seed; incremented per attempt.
        max_steps: Per-episode step cap.
        clear_threshold: Single-step reward marking a screen clear (ADR 0002).
        sticky_action_prob: Sticky-action probability during collection.

    Returns:
        Dict with attempts, successes, returns, and dataset sizes.
    """
    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=sticky_action_prob,
        full_action_space=True,
    )
    env = AtariPreprocessing(base, noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    ale = cast(Any, base.unwrapped).ale

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    kept_returns: list[float] = []
    attempts = 0
    successes = 0

    while successes < episodes and attempts < episodes * 8:
        attempts += 1
        policy = ScriptedClimber()
        obs, _ = env.reset(seed=seed + attempts)
        ep_obs: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_return = 0.0
        cleared = False
        for _ in range(max_steps):
            action = policy.act(ale.getRAM())
            ep_obs.append(np.asarray(obs, dtype=np.uint8))
            ep_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            if float(reward) >= clear_threshold:
                cleared = True
            if terminated or truncated:
                break
        if cleared:
            successes += 1
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            kept_returns.append(ep_return)
        logger.info(
            f"attempt {attempts}: return={ep_return:.0f} cleared={cleared} "
            f"kept={successes}/{episodes}"
        )
    env.close()

    if successes == 0:
        raise RuntimeError("scripted demonstrator never cleared the screen; tune the policy")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=np.stack(all_obs),
        actions=np.asarray(all_actions, dtype=np.int64),
    )
    result = {
        "attempts": attempts,
        "successes": successes,
        "samples": len(all_actions),
        "returns": kept_returns,
        "path": str(out_path),
    }
    logger.info(
        f"demos saved: {successes} episodes, {len(all_actions)} samples, "
        f"mean return {np.mean(kept_returns):.0f}, {out_path}"
    )
    return result


def collect_self_demos(
    checkpoint_path: Path,
    out_path: Path,
    episodes: int = 300,
    keep: int = 25,
    seed: int = 11000,
    merge_with: Path | None = None,
) -> dict:
    """Harvest a trained agent's best episodes as a BC dataset (expert iteration).

    Rolls the checkpoint's stochastic policy through the protocol eval env,
    keeps the ``keep`` highest-return episodes out of ``episodes``, and
    optionally concatenates an existing demo set so earlier behavior support
    is preserved.

    Args:
        checkpoint_path: Trained checkpoint to roll out.
        out_path: Output ``.npz`` path.
        episodes: Total episodes to sample.
        keep: Number of top episodes to keep.
        seed: Eval pool seed (also seeds policy sampling).
        merge_with: Optional existing demo ``.npz`` to concatenate.

    Returns:
        Dict with kept-episode returns and dataset sizes.
    """
    import torch

    from dkrl.config import ExperimentConfig
    from dkrl.envs import make_eval_envs
    from dkrl.evaluate import load_policy

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, cfg, device)
    torch.manual_seed(seed)
    envs = make_eval_envs(
        cfg.eval,
        cfg.env.env_id,
        cfg.env.img_size,
        cfg.env.frame_stack,
        seed,
        full_action_space=cfg.env.full_action_space,
    )
    n = cfg.eval.num_envs

    buf_obs: list[list[np.ndarray]] = [[] for _ in range(n)]
    buf_act: list[list[int]] = [[] for _ in range(n)]
    returns_acc = np.zeros(n, dtype=np.float64)
    kept: list[tuple[float, np.ndarray, np.ndarray]] = []
    done_eps = 0

    obs, _ = envs.reset()
    while done_eps < episodes:
        actions = policy(torch.from_numpy(obs).to(device)).cpu().numpy()
        for i in range(n):
            buf_obs[i].append(obs[i].astype(np.uint8))
            buf_act[i].append(int(actions[i]))
        obs, _, terminated, truncated, info = envs.step(actions)
        returns_acc += info["reward"]
        real_done = np.logical_or(info.get("terminated", terminated), truncated)
        for i in np.nonzero(real_done)[0]:
            done_eps += 1
            ep_return = float(returns_acc[i])
            worst_kept = kept[0][0] if len(kept) >= keep else None
            if worst_kept is None or ep_return > worst_kept:
                kept.append((ep_return, np.stack(buf_obs[i]), np.asarray(buf_act[i], np.int64)))
                kept.sort(key=lambda e: e[0])
                if len(kept) > keep:
                    kept.pop(0)
            returns_acc[i] = 0.0
            buf_obs[i] = []
            buf_act[i] = []
            if done_eps % 50 == 0:
                floor = kept[0][0] if kept else float("nan")
                logger.info(f"self-demos: {done_eps}/{episodes} episodes, kept floor {floor:.0f}")
    envs.close()

    obs_arrays = [e[1] for e in kept]
    act_arrays = [e[2] for e in kept]
    kept_returns = [e[0] for e in kept]
    if merge_with is not None:
        prior = np.load(merge_with)
        obs_arrays.append(prior["observations"])
        act_arrays.append(prior["actions"])
    observations = np.concatenate(obs_arrays)
    actions_all = np.concatenate(act_arrays)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, observations=observations, actions=actions_all)
    result = {
        "episodes_sampled": episodes,
        "episodes_kept": len(kept),
        "kept_return_min": float(min(kept_returns)),
        "kept_return_mean": float(np.mean(kept_returns)),
        "kept_return_max": float(max(kept_returns)),
        "samples": int(len(actions_all)),
        "merged_with": str(merge_with) if merge_with else None,
        "path": str(out_path),
    }
    logger.info(
        f"self-demos saved: kept {len(kept)} of {episodes}, returns "
        f"{result['kept_return_min']:.0f}..{result['kept_return_max']:.0f} "
        f"(mean {result['kept_return_mean']:.0f}), {result['samples']} samples, {out_path}"
    )
    return result


def harvest_self_trajectories(
    checkpoint_path: Path,
    out_path: Path,
    episodes: int = 300,
    keep: int = 25,
    seed: int = 12000,
) -> dict:
    """Harvest a trained agent's best episodes as full SIL demonstrations.

    Like :func:`collect_self_demos`, but records per-step rewards and keeps the
    episode structure, so each kept episode can be replayed into a
    self-imitation buffer with a proper discounted return-to-go. Output is a
    pickle of episodes (observation, action, reward arrays), not a flat BC npz.

    Args:
        checkpoint_path: Trained checkpoint to roll out.
        out_path: Output ``.pkl`` path.
        episodes: Total episodes to sample.
        keep: Number of top-return episodes to keep.
        seed: Eval pool seed (also seeds policy sampling).

    Returns:
        Dict with kept-episode returns and dataset sizes.
    """
    import torch

    from dkrl.config import ExperimentConfig
    from dkrl.envs import make_eval_envs
    from dkrl.evaluate import load_policy

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, cfg, device)
    torch.manual_seed(seed)
    envs = make_eval_envs(
        cfg.eval,
        cfg.env.env_id,
        cfg.env.img_size,
        cfg.env.frame_stack,
        seed,
        full_action_space=cfg.env.full_action_space,
    )
    n = cfg.eval.num_envs

    buf_obs: list[list[np.ndarray]] = [[] for _ in range(n)]
    buf_act: list[list[int]] = [[] for _ in range(n)]
    buf_rew: list[list[float]] = [[] for _ in range(n)]
    returns_acc = np.zeros(n, dtype=np.float64)
    kept: list[dict[str, Any]] = []
    done_eps = 0

    obs, _ = envs.reset()
    while done_eps < episodes:
        actions = policy(torch.from_numpy(obs).to(device)).cpu().numpy()
        for i in range(n):
            buf_obs[i].append(obs[i].astype(np.uint8))
            buf_act[i].append(int(actions[i]))
        obs, _, terminated, truncated, info = envs.step(actions)
        step_reward = info["reward"]
        returns_acc += step_reward
        for i in range(n):
            buf_rew[i].append(float(step_reward[i]))
        real_done = np.logical_or(info.get("terminated", terminated), truncated)
        for i in np.nonzero(real_done)[0]:
            done_eps += 1
            ep_return = float(returns_acc[i])
            worst = kept[0]["return"] if len(kept) >= keep else None
            if worst is None or ep_return > worst:
                kept.append(
                    {
                        "return": ep_return,
                        "obs": np.stack(buf_obs[i]),
                        "actions": np.asarray(buf_act[i], np.int64),
                        "rewards": np.asarray(buf_rew[i], np.float32),
                    }
                )
                kept.sort(key=lambda e: e["return"])
                if len(kept) > keep:
                    kept.pop(0)
            returns_acc[i] = 0.0
            buf_obs[i] = []
            buf_act[i] = []
            buf_rew[i] = []
            if done_eps % 50 == 0:
                floor = kept[0]["return"] if kept else float("nan")
                logger.info(f"harvest: {done_eps}/{episodes} episodes, kept floor {floor:.0f}")
    envs.close()

    returns = [e["return"] for e in kept]
    samples = int(sum(len(e["actions"]) for e in kept))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"episodes": kept}, f)
    result = {
        "episodes_sampled": episodes,
        "episodes_kept": len(kept),
        "kept_return_min": float(min(returns)),
        "kept_return_mean": float(np.mean(returns)),
        "kept_return_max": float(max(returns)),
        "samples": samples,
        "path": str(out_path),
    }
    logger.info(
        f"harvest saved: kept {len(kept)} of {episodes}, returns "
        f"{result['kept_return_min']:.0f}..{result['kept_return_max']:.0f} "
        f"(mean {result['kept_return_mean']:.0f}), {samples} transitions, {out_path}"
    )
    return result
