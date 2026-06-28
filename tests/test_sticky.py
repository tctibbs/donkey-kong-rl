"""Tests for the deterministic sticky-action primitive shared by Go-Explore and SIL seeding.

Exploration and replay must realize identical perturbations, so this is a pure, deterministic
function of the step index; if it drifted, seeded demo trajectories would diverge from the
scores they were harvested at.
"""

from dkrl.explore import _sticky_effective


def test_passthrough_when_sticky_zero() -> None:
    assert _sticky_effective(5, 3, 0, 0.0) == 5
    assert _sticky_effective(5, 3, 100, 0.0) == 5


def test_sticky_one_always_repeats_previous() -> None:
    assert all(_sticky_effective(5, 3, step, 1.0) == 3 for step in range(50))


def test_is_deterministic_across_calls() -> None:
    a = [_sticky_effective(7, 2, s, 0.25) for s in range(200)]
    b = [_sticky_effective(7, 2, s, 0.25) for s in range(200)]
    assert a == b


def test_realized_repeat_frequency_matches_sticky() -> None:
    n = 20000
    repeats = sum(1 for s in range(n) if _sticky_effective(7, 2, s, 0.25) == 2)
    assert abs(repeats / n - 0.25) < 0.03
