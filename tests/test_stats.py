"""Tests for the reported aggregate statistics: the per-checkpoint summary and the
cross-seed pooling (IQM, bootstrap CIs, clear rate) that produce the headline numbers.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dkrl.evaluate import summarize
from dkrl.stats import aggregate_runs


def test_summarize_basic() -> None:
    out = summarize([100.0, 200.0], [10, 20], [1500.0, 0.0], 1500.0)
    assert out["episodes"] == 2
    assert out["return_mean"] == 150.0
    assert out["return_max"] == 200.0
    assert out["clear_rate"] == 0.5  # one of two episodes hit the >=1500 threshold
    assert out["clear_reward_threshold"] == 1500.0


def _write_ledger(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_aggregate_runs_pools_seeds(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(
        ledger,
        [
            {
                "kind": "eval",
                "run_id": "exp_s1",
                "return_mean": 100.0,
                "clear_rate": 0.5,
                "episodes": 4,
                "returns": [0.0, 100.0, 100.0, 200.0],
                "max_step_rewards": [0.0, 1500.0, 0.0, 1500.0],
                "clear_reward_threshold": 1500.0,
            },
            {
                "kind": "eval",
                "run_id": "exp_s2",
                "return_mean": 150.0,
                "clear_rate": 1.0,
                "episodes": 4,
                "returns": [100.0, 100.0, 200.0, 200.0],
                "max_step_rewards": [1500.0, 1500.0, 1500.0, 1500.0],
                "clear_reward_threshold": 1500.0,
            },
        ],
    )
    out = aggregate_runs("exp", ledger_path=ledger, bootstrap_samples=500, seed=0)
    pooled = [0, 100, 100, 200, 100, 100, 200, 200]
    assert out["seeds"] == 2
    assert out["episodes_total"] == 8
    assert out["return_mean"] == pytest.approx(float(np.mean(pooled)))
    assert out["return_median"] == pytest.approx(float(np.median(pooled)))
    assert out["clear_rate"] == pytest.approx(6 / 8)  # 6 of 8 episodes cleared
    lo, hi = out["return_mean_ci95"]
    assert lo <= out["return_mean"] <= hi  # the CI brackets the point estimate


def test_aggregate_runs_raises_without_matching_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    with pytest.raises(ValueError):
        aggregate_runs("nope", ledger_path=ledger)
