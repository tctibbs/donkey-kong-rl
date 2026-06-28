"""Tests for the SIL buffer and Monte-Carlo return labelling that underpin the SIL results."""

import numpy as np

from dkrl.train.ppo_sil import SILBuffer, _discounted_returns


def _obs(n: int, val: int) -> list[np.ndarray]:
    return [np.full((1, 2, 2), val, dtype=np.uint8) for _ in range(n)]


def test_discounted_returns_limits() -> None:
    assert _discounted_returns([1.0, 1.0, 1.0], 0.5) == [1.75, 1.5, 1.0]
    assert _discounted_returns([1.0, 2.0, 3.0], 0.0) == [1.0, 2.0, 3.0]  # gamma=0: unchanged
    assert _discounted_returns([1.0, 2.0, 3.0], 1.0) == [6.0, 5.0, 3.0]  # gamma=1: reverse cumsum


def test_add_episode_wraps_circularly() -> None:
    buf = SILBuffer(capacity=3, obs_shape=(1, 2, 2), demo_capacity=2)
    buf.add_episode(_obs(2, 1), [0, 1], [1.0, 2.0])
    assert buf.size == 2 and buf.pos == 2
    buf.add_episode(_obs(2, 2), [2, 3], [3.0, 4.0])  # overflows capacity 3
    assert buf.size == 3  # caps at capacity
    assert buf.pos == 1  # (2 + 2) % 3


def test_add_demos_is_non_evicting() -> None:
    buf = SILBuffer(capacity=10, obs_shape=(1, 2, 2), demo_capacity=2)
    buf.add_demos(_obs(5, 9), [0, 1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert buf.d_size == 2  # stops at demo_capacity, never overwrites
    assert list(buf.d_actions[:2]) == [0, 1]  # keeps the first, does not evict


def test_prioritized_uniform_when_returns_equal() -> None:
    rng = np.random.default_rng(0)
    draws = SILBuffer._prioritized(np.array([5.0, 5.0, 5.0, 5.0]), 4000, rng)
    counts = np.bincount(draws, minlength=4)
    assert counts.min() > 800 and counts.max() < 1200  # ~uniform over 4 indices


def test_prioritized_favours_higher_return() -> None:
    rng = np.random.default_rng(0)
    draws = SILBuffer._prioritized(np.array([0.0, 100.0]), 10000, rng)  # priorities 1 vs 101
    assert (draws == 1).mean() > 0.9


def test_sample_splits_batch_by_demo_fraction() -> None:
    rng = np.random.default_rng(0)
    buf = SILBuffer(capacity=10, obs_shape=(1, 2, 2), demo_capacity=4)
    buf.add_episode(_obs(6, 1), list(range(6)), [1.0] * 6)
    buf.add_demos(_obs(4, 9), list(range(4)), [1.0] * 4)
    _, act, _ = buf.sample(batch=10, demo_fraction=0.5, rng=rng)
    assert len(act) == 10  # 5 self + 5 demo


def test_sample_falls_back_to_self_only_without_demos() -> None:
    rng = np.random.default_rng(0)
    buf = SILBuffer(capacity=10, obs_shape=(1, 2, 2), demo_capacity=4)
    buf.add_episode(_obs(8, 1), list(range(8)), [1.0] * 8)
    _, act, _ = buf.sample(batch=8, demo_fraction=0.5, rng=rng)
    assert len(act) == 8  # d_size == 0 -> n_demo == 0, all from self
