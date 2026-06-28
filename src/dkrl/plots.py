"""Plotting helpers: learning curves and evaluation score distributions."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(path: Path) -> tuple[list[int], list[float]]:
    """Read (global_step, mean_return_100) series from a metrics CSV.

    Args:
        path: CSV written by the training loop.

    Returns:
        Tuple of step and return lists, NaN rows skipped.
    """
    steps: list[int] = []
    returns: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            value = float(row["mean_return_100"])
            if value == value:
                steps.append(int(row["global_step"]))
                returns.append(value)
    return steps, returns


def plot_learning_curves(
    metric_files: list[Path], out_path: Path, title: str = "DonkeyKong-v5 learning curves"
) -> Path:
    """Plot one learning curve per metrics CSV onto a single figure.

    Args:
        metric_files: Metrics CSVs written by training runs.
        out_path: Output PNG path.
        title: Figure title.

    Returns:
        The output path.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for path in metric_files:
        steps, returns = read_metrics(path)
        ax.plot(steps, returns, label=path.stem, linewidth=1.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (mean of last 100)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def read_eval_returns(ledger_path: Path, run_id: str) -> list[float]:
    """Read the per-episode returns of a run's latest evaluation row.

    Args:
        ledger_path: Path to the append-only ledger (results/ledger.jsonl).
        run_id: Run id whose evaluation row to read.

    Returns:
        Per-episode raw returns from the last eval row matching the run id,
        or an empty list if no such row exists.
    """
    returns: list[float] = []
    with ledger_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "eval" and row.get("run_id") == run_id:
                returns = [float(x) for x in row["returns"]]
    return returns


def plot_score_distribution(
    returns: list[float],
    out_path: Path,
    title: str = "DonkeyKong-v5 evaluation score distribution",
    goal: float | None = None,
) -> Path:
    """Plot a histogram of per-episode evaluation returns with mean and median.

    Args:
        returns: Per-episode raw returns from one evaluation.
        out_path: Output PNG path.
        title: Figure title.
        goal: Optional reference score drawn as a vertical line (e.g. a target).

    Returns:
        The output path.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(returns, bins=20, color="#4c72b0", edgecolor="white")
    mean = statistics.fmean(returns)
    median = statistics.median(returns)
    ax.axvline(mean, color="#c44e52", linewidth=1.5, label=f"mean {mean:,.0f}")
    ax.axvline(
        median, color="#55a868", linewidth=1.5, linestyle="--", label=f"median {median:,.0f}"
    )
    if goal is not None:
        ax.axvline(goal, color="#8172b3", linewidth=1.5, linestyle=":", label=f"goal {goal:,.0f}")
    ax.set_xlabel("episode return")
    ax.set_ylabel("episodes")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
