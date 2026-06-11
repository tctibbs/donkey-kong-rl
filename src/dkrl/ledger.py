"""Append-only JSONL results ledger.

Every run writes one row keyed by config hash, with git SHA, seed, scores,
frame count, and wall-clock, so results survive independently of any external
tracking service.
"""

import json
import subprocess
from pathlib import Path

LEDGER_PATH = Path("results/ledger.jsonl")


def git_sha() -> str:
    """Return the current git SHA, with a ``-dirty`` suffix if the tree is dirty."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def append_row(row: dict, path: Path = LEDGER_PATH) -> None:
    """Append one result row to the JSONL ledger.

    Args:
        row: JSON-serializable result record.
        path: Ledger file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(path: Path = LEDGER_PATH) -> list[dict]:
    """Read all rows from the JSONL ledger.

    Args:
        path: Ledger file path.

    Returns:
        List of result records, oldest first.
    """
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
