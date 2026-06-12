"""Backward-curriculum robustification (Go-Explore phase 2) with stage experts.

Builds a bank of emulator states along explored trajectories, then trains
PPO from restored states sampled by a receding-window curriculum: episodes
start near the end of the trajectory (reward seconds away) and the window
recedes as the policy's success rate rises. Training runs under real sticky
actions, so the policy learns robust execution by trial rather than
imitation. States are tagged by stage (barrels or rivets) so per-stage
experts train only on their own states.
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


@dataclass
class BankEntry:
    """One restorable state along a trajectory."""

    state_blob: bytes
    step_index: int
    score_at: float
    score_to_go: float
    board: int


def build_state_bank(
    trajectory_paths: list[Path],
    out_path: Path,
    top_n: int = 5,
    snapshot_every: int = 25,
) -> dict:
    """Replay top trajectories and snapshot restorable states along them.

    Args:
        trajectory_paths: Explore result pickles (non-sticky trajectories).
        out_path: Output pickle for the bank.
        top_n: Trajectories to bank per file pool.
        snapshot_every: Steps between snapshots.

    Returns:
        Dict with bank size and composition.
    """
    pooled: list[tuple[float, list[int]]] = []
    for path in trajectory_paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if float(payload.get("sticky", 0.0)) != 0.0:
            raise ValueError(f"{path} was explored with sticky; bank needs exact replay")
        pooled.extend(payload["trajectories"])
    pooled.sort(key=lambda t: t[0], reverse=True)

    base = gym.make(
        "ALE/DonkeyKong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=True,
    )
    env = AtariPreprocessing(base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, 4)
    ale = cast(Any, base.unwrapped).ale

    bank: list[BankEntry] = []
    seen: set[tuple[float, int]] = set()
    banked = 0
    for score, actions in pooled[: top_n * 3]:
        if banked >= top_n or (score, len(actions)) in seen:
            continue
        seen.add((score, len(actions)))
        env.reset(seed=0)
        total = 0.0
        snapshots: list[tuple[bytes, int, float, int]] = []
        for i, a in enumerate(actions):
            if i % snapshot_every == 0:
                snapshots.append(
                    (
                        pickle.dumps(ale.cloneSystemState()),
                        i,
                        total,
                        int(ale.getRAM()[RAM_BOARD]),
                    )
                )
            _, reward, terminated, truncated, _ = env.step(int(a))
            total += float(reward)
            if terminated or truncated:
                break
        if total < 0.9 * score:
            logger.warning(f"bank replay mismatch ({total:.0f} vs {score:.0f}), skipping")
            continue
        banked += 1
        for blob, idx, score_at, board in snapshots:
            bank.append(BankEntry(blob, idx, score_at, total - score_at, board))
        logger.info(f"banked trajectory score {total:.0f}: {len(snapshots)} states")
    env.close()

    if not bank:
        raise RuntimeError("no trajectories banked")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bank, f)
    boards = [e.board for e in bank]
    result = {
        "states": len(bank),
        "trajectories": banked,
        "barrels_states": int(sum(1 for b in boards if b == 0)),
        "rivets_states": int(sum(1 for b in boards if b == 1)),
        "max_score_to_go": float(max(e.score_to_go for e in bank)),
        "path": str(out_path),
    }
    logger.info(f"state bank: {result}")
    return result


class RestoreResetEnv(gym.Env):
    """Single ALE env whose reset restores a sampled bank state.

    The curriculum window is controlled externally via `set_window`: reset
    samples states whose score_to_go lies inside [window_lo, window_hi], so
    training starts close to upcoming reward and recedes backward over time.
    Runs under true sticky actions for robustness.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bank_path: str,
        board_filter: int | None = None,
        sticky: float = 0.25,
        max_steps: int = 2000,
        seed: int = 0,
    ) -> None:
        """Load the bank and build the wrapped ALE stack."""
        super().__init__()
        with Path(bank_path).open("rb") as f:
            bank: list[BankEntry] = pickle.load(f)
        if board_filter is not None:
            bank = [e for e in bank if e.board == board_filter]
        if not bank:
            raise ValueError("empty state bank after filtering")
        self.bank = sorted(bank, key=lambda e: e.score_to_go)
        base = gym.make(
            "ALE/DonkeyKong-v5",
            frameskip=1,
            repeat_action_probability=sticky,
            full_action_space=True,
        )
        env = AtariPreprocessing(
            base, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True
        )
        self._env = FrameStackObservation(env, 4)
        self._ale = cast(Any, base.unwrapped).ale
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._max_steps = max_steps
        self._lo = 0.0
        self._hi = 1.0
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._env.reset(seed=seed)

    def set_window(self, lo_frac: float, hi_frac: float) -> None:
        """Set the curriculum window as fractions of the bank's depth order."""
        self._lo = lo_frac
        self._hi = hi_frac

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Restore a bank state sampled from the current curriculum window."""
        n = len(self.bank)
        lo = int(self._lo * (n - 1))
        hi = max(lo + 1, int(self._hi * (n - 1)))
        entry = self.bank[int(self._rng.integers(lo, hi + 1)) % n]
        self._env.reset(seed=0)
        self._ale.restoreSystemState(pickle.loads(entry.state_blob))
        self._steps = 0
        obs, reward, terminated, truncated, info = self._env.step(0)
        return obs, info

    def step(self, action: int) -> tuple:
        """Step the underlying stack with an episode-step cap."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        if self._steps >= self._max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying env."""
        self._env.close()
