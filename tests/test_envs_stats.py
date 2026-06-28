"""Tests for the episode-stats tracker that produces every reported return and clear rate.

EpisodeStats.update accumulates raw (info['reward']) returns, takes the running max single-step
reward (the >=1500 clear detector), and ends an episode only on real termination, distinct from
the episodic-life step-terminated flag.
"""

import numpy as np

from dkrl.envs import EpisodeStats


def _info(reward: list[float], terminated: list[bool]) -> dict:
    return {"reward": np.array(reward), "terminated": np.array(terminated)}


def test_accumulates_raw_reward_and_resets_only_on_real_done() -> None:
    stats = EpisodeStats(num_envs=2)
    nope = np.array([False, False])

    # Step 1: env0 +100, env1 +1500, neither ends.
    assert stats.update(nope, nope, _info([100.0, 1500.0], [False, False])) == []

    # Step 2: env1 really ends (info terminated), env0 continues.
    out = stats.update(np.array([False, True]), nope, _info([1500.0, 0.0], [False, True]))
    assert out == [(1500.0, 2, 1500.0)]  # env1: return 1500, length 2, max 1500

    # Step 3: env0 really ends; its accumulators ran across all three steps.
    out = stats.update(nope, nope, _info([0.0, 0.0], [True, False]))
    assert out == [(1600.0, 3, 1500.0)]  # env0: 100+1500+0, length 3, max 1500


def test_episodic_life_loss_is_not_a_real_done() -> None:
    stats = EpisodeStats(num_envs=1)
    # Step terminated True (a life was lost) but info terminated False (game not over).
    out = stats.update(np.array([True]), np.array([False]), _info([100.0], [False]))
    assert out == []


def test_truncation_ends_the_episode() -> None:
    stats = EpisodeStats(num_envs=1)
    out = stats.update(np.array([False]), np.array([True]), _info([200.0], [False]))
    assert out == [(200.0, 1, 200.0)]
