"""Learning-curve plotting from per-run metrics CSVs."""

import csv
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
