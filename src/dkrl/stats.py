"""Aggregate statistics over ledger evaluations (IQM, bootstrap CIs).

Uses rliable's interquartile mean (Agarwal et al. 2021). Confidence intervals
come from a plain percentile bootstrap over episodes; rliable's own
stratified-bootstrap module is unused because its `arch` dependency is
incompatible with pandas 3.
"""

from pathlib import Path

import numpy as np
from rliable import metrics

from dkrl.ledger import LEDGER_PATH, read_rows


def aggregate_runs(
    name_prefix: str,
    ledger_path: Path = LEDGER_PATH,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Aggregate all eval rows whose run_id starts with a name prefix.

    Pools per-episode returns across seeds (one eval row per seed) and
    reports mean, IQM, clear rate, and 95 percent bootstrap CIs.

    Args:
        name_prefix: Run-id prefix, for example ``ppo_final_25m``.
        ledger_path: Ledger file.
        bootstrap_samples: Bootstrap resamples for the CIs.
        seed: Bootstrap RNG seed.

    Returns:
        Dict with per-seed summaries and pooled statistics.
    """
    rows = [
        r
        for r in read_rows(ledger_path)
        if r.get("kind") == "eval" and str(r.get("run_id", "")).startswith(name_prefix)
    ]
    if not rows:
        raise ValueError(f"no eval rows match prefix {name_prefix!r}")

    per_seed = [
        {
            "run_id": r["run_id"],
            "return_mean": r["return_mean"],
            "clear_rate": r.get("clear_rate"),
            "episodes": r["episodes"],
        }
        for r in rows
    ]
    pooled = np.concatenate([np.asarray(r["returns"], dtype=np.float64) for r in rows])
    clears = np.concatenate(
        [
            (
                np.asarray(r["max_step_rewards"], dtype=np.float64)
                >= float(r.get("clear_reward_threshold", 1500.0))
            )
            if "max_step_rewards" in r
            else np.full(len(r["returns"]), np.nan)
            for r in rows
        ]
    )

    rng = np.random.default_rng(seed)
    n = len(pooled)
    idx = rng.integers(0, n, size=(bootstrap_samples, n))
    resamples = pooled[idx]
    boot_means = resamples.mean(axis=1)
    sorted_rows = np.sort(resamples, axis=1)
    k = n // 4
    boot_iqms = sorted_rows[:, k : n - k].mean(axis=1)

    return {
        "name_prefix": name_prefix,
        "seeds": len(rows),
        "episodes_total": n,
        "per_seed": per_seed,
        "return_mean": float(pooled.mean()),
        "return_mean_ci95": [
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        ],
        "return_iqm": float(metrics.aggregate_iqm(pooled.reshape(1, -1))),
        "return_iqm_ci95": [
            float(np.percentile(boot_iqms, 2.5)),
            float(np.percentile(boot_iqms, 97.5)),
        ],
        "return_median": float(np.median(pooled)),
        "return_std": float(pooled.std()),
        "clear_rate": float(np.nanmean(clears)),
    }
