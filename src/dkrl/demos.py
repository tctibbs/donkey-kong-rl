"""Scripted RAM-based demonstrator and demonstration collection.

A rule-based climbing policy reads the Atari 2600 Donkey Kong RAM (byte map
in docs/research/2026-06-11-donkeykong-ram-map.md) and climbs the barrels
screen while jumping approaching barrels. Successful episodes (screen
cleared) are saved as (observation, action) pairs for behavior-cloning
warm-starts.

All coordinates are raw RAM units. Mario y grows downward.
"""

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
            return self._rivets_fallback(ram, x)

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

    def _rivets_fallback(self, ram: np.ndarray, x: int) -> int:
        """Simple survival behavior on the rivets screen (out of demo scope)."""
        toward = RIGHT if x < 80 else LEFT
        if self._barrel_threat(ram, x, int(ram[RAM_MARIO_Y])):
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
                break
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
