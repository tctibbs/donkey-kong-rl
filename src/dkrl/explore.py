"""Go-Explore-style trajectory search for demonstration generation.

Explores Donkey Kong with emulator state save/restore (ale-py cloneState),
archiving the best game state per progress cell and expanding from promising
cells with short random action bursts. The output is a deterministic action
sequence reaching deep game states (rivets-screen completions, later rounds),
which `replay_to_demos` converts into (observation, action) BC pairs through
the standard training preprocessing.

Exploration runs without sticky actions and without no-op starts so action
sequences replay exactly.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from loguru import logger

gym.register_envs(ale_py)

RAM_BOARD = 16
RAM_LIVES = 35
RAM_MARIO_Y = 27
RAM_GAMESTATE = 12


@dataclass
class Cell:
    """Archive entry: best state seen for one progress bucket."""

    score: float
    actions: list[int]
    ale_state: Any
    steps: int


def _make_env() -> tuple[gym.Env, Any]:
    """Build the deterministic exploration env (frameskip 4, no sticky)."""
    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=True,
    )
    return base, cast(Any, base.unwrapped).ale


def _cell_key(ale: Any, score: float, boards_cleared: int) -> tuple:
    """Progress bucket: board progression, lives, position bands, score band."""
    ram = ale.getRAM()
    return (
        boards_cleared,
        int(ram[RAM_BOARD]),
        int(ram[RAM_LIVES]),
        int(ram[RAM_MARIO_Y]) // 12,
        int(ram[19]) // 14,
        int(score // 300),
    )


def explore(
    out_path: Path,
    iterations: int = 4000,
    burst_min: int = 20,
    burst_max: int = 80,
    seed: int = 0,
    target_score: float = 35000.0,
    resume_from: list[Path] | None = None,
    resume_top: int = 8,
) -> dict:
    """Run the exploration loop and save the best trajectories found.

    Args:
        out_path: Pickle path for the result (best action sequences).
        iterations: Number of cell expansions.
        burst_min: Minimum random actions per expansion burst.
        burst_max: Maximum random actions per expansion burst.
        seed: RNG seed (env seed is fixed to 0 for replay determinism).
        target_score: Stop early when any cell reaches this score.
        resume_from: Prior result pickles; their top trajectories are
            replayed to seed the archive (iterative deepening).
        resume_top: Trajectories to replay per resume file.

    Returns:
        Dict with archive size, best score, and output path.
    """
    rng = np.random.default_rng(seed)
    base, ale = _make_env()
    base.reset(seed=0)

    start_actions: list[int] = []
    for _ in range(40):
        a = 1 if len(start_actions) % 2 == 0 else 0
        base.step(a)
        start_actions.append(a)
    if not int(ale.getRAM()[RAM_GAMESTATE]) & 0x80:
        raise RuntimeError("game did not start after FIRE presses")

    archive: dict[tuple, Cell] = {}
    root = Cell(0.0, start_actions, ale.cloneState(), len(start_actions))
    archive[_cell_key(ale, 0.0, 0)] = root
    best = root
    boards_cleared_by_cell: dict[tuple, int] = {next(iter(archive)): 0}

    if resume_from:
        for path in resume_from:
            with path.open("rb") as f:
                prior = pickle.load(f)
            for score_prior, actions_prior in prior["trajectories"][:resume_top]:
                base.reset(seed=0)
                score = 0.0
                cleared = 0
                trace: list[int] = []
                for a in actions_prior:
                    _, reward, terminated, truncated, _ = base.step(int(a))
                    trace.append(int(a))
                    score += float(reward)
                    if float(reward) >= 1500:
                        cleared += 1
                    if terminated or truncated:
                        break
                    k = _cell_key(ale, score, cleared)
                    existing = archive.get(k)
                    if existing is None or score > existing.score:
                        archive[k] = Cell(score, list(trace), ale.cloneState(), len(trace))
                        boards_cleared_by_cell[k] = cleared
                        if score > best.score:
                            best = archive[k]
        base.reset(seed=0)
        for a in start_actions:
            base.step(a)
        logger.info(f"resumed: archive {len(archive)} cells, best {best.score:.0f}")

    action_pool = [2] * 6 + [3] * 3 + [4] * 3 + [1, 11, 12] * 2 + [0, 5, 6, 7]
    needs_reset = False
    for it in range(iterations):
        keys = list(archive.keys())
        weights = np.array(
            [
                1.0
                + 4.0 * k[0]
                + archive[k].score / 1500.0
                + (2.0 if k[1] == 1 else 0.0)
                + (154 - min(154, k[3] * 12)) / 30.0
                for k in keys
            ]
        )
        weights /= weights.sum()
        key = keys[rng.choice(len(keys), p=weights)]
        cell = archive[key]
        boards_cleared = boards_cleared_by_cell.get(key, 0)

        if needs_reset:
            base.reset(seed=0)
            needs_reset = False
        ale.restoreState(cell.ale_state)
        score = cell.score
        actions = list(cell.actions)
        cleared_now = boards_cleared
        steps_left = int(rng.integers(burst_min, burst_max + 1))
        while steps_left > 0:
            current = int(action_pool[rng.integers(0, len(action_pool))])
            hold = int(rng.integers(3, 41)) if current in (2, 3, 4, 5) else int(rng.integers(1, 4))
            ended = False
            for _ in range(min(hold, steps_left)):
                steps_left -= 1
                _, reward, terminated, truncated, _ = base.step(current)
                actions.append(current)
                score += float(reward)
                if float(reward) >= 1500:
                    cleared_now += 1
                if terminated or truncated:
                    ended = True
                    needs_reset = True
                    break
                new_key = _cell_key(ale, score, cleared_now)
                existing = archive.get(new_key)
                if (
                    existing is None
                    or score > existing.score
                    or (score == existing.score and len(actions) < len(existing.actions))
                ):
                    archive[new_key] = Cell(score, list(actions), ale.cloneState(), len(actions))
                    boards_cleared_by_cell[new_key] = cleared_now
                    if score > best.score:
                        best = archive[new_key]
            if ended:
                break
        if it % 200 == 0:
            logger.info(f"iter {it}: best {best.score:.0f}, archive {len(archive)} cells")
        if best.score >= target_score:
            logger.info(f"target reached at iter {it}")
            break

    base.close()
    ranked = sorted(archive.values(), key=lambda c: c.score, reverse=True)
    payload = {
        "start_seed": 0,
        "trajectories": [(c.score, c.actions) for c in ranked[:50]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    result = {
        "iterations": it + 1,
        "archive_cells": len(archive),
        "best_score": best.score,
        "path": str(out_path),
    }
    logger.info(f"explore done: best {best.score:.0f}, {len(archive)} cells, {out_path}")
    return result


def replay_to_demos(
    trajectory_paths: list[Path],
    out_path: Path,
    top_n: int = 25,
    min_score: float = 6000.0,
    merge_with: Path | None = None,
) -> dict:
    """Replay explored action sequences into (observation, action) BC pairs.

    Replays through the standard training preprocessing (grayscale 84x84,
    frame skip 4, stack 4) with the exploration env settings (no sticky, no
    no-op starts, fixed seed) so observations match what the sequences did.

    Args:
        trajectory_paths: Pickles written by `explore` (workers are pooled).
        out_path: Output ``.npz``.
        top_n: Maximum trajectories to convert.
        min_score: Skip trajectories below this score.
        merge_with: Optional existing demo set to concatenate.

    Returns:
        Dict with converted counts and dataset sizes.
    """
    pooled: list[tuple[float, list[int]]] = []
    for path in trajectory_paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        pooled.extend(payload["trajectories"])
    pooled.sort(key=lambda t: t[0], reverse=True)
    payload = {"start_seed": 0, "trajectories": pooled}

    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
    )
    env = AtariPreprocessing(base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    converted = 0
    replay_scores: list[float] = []
    for score, actions in payload["trajectories"]:
        if converted >= top_n or score < min_score:
            continue
        obs, _ = env.reset(seed=payload["start_seed"])
        total = 0.0
        for a in actions:
            all_obs.append(np.asarray(obs, dtype=np.uint8))
            all_actions.append(int(a))
            obs, reward, terminated, truncated, _ = env.step(int(a))
            total += float(reward)
            if terminated or truncated:
                break
        converted += 1
        replay_scores.append(total)
        logger.info(f"replayed trajectory: explored {score:.0f}, replayed {total:.0f}")

    env.close()
    if converted == 0:
        raise ValueError("no trajectories above min_score")

    obs_arrays = [np.stack(all_obs)]
    act_arrays = [np.asarray(all_actions, np.int64)]
    if merge_with is not None:
        prior = np.load(merge_with)
        obs_arrays.append(prior["observations"])
        act_arrays.append(prior["actions"])
    observations = np.concatenate(obs_arrays)
    actions_all = np.concatenate(act_arrays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, observations=observations, actions=actions_all)
    result = {
        "converted": converted,
        "replay_score_max": float(max(replay_scores)),
        "replay_score_mean": float(np.mean(replay_scores)),
        "samples": int(len(actions_all)),
        "merged_with": str(merge_with) if merge_with else None,
        "path": str(out_path),
    }
    logger.info(
        f"replay demos saved: {converted} trajectories, replay max "
        f"{result['replay_score_max']:.0f}, {result['samples']} samples, {out_path}"
    )
    return result
