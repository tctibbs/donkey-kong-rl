"""Tests for the JSONL results ledger."""

from pathlib import Path

from dkrl.ledger import append_row, read_rows


def test_append_and_read(tmp_path: Path) -> None:
    """Rows append in order and read back identically."""
    path = tmp_path / "ledger.jsonl"
    append_row({"run_id": "r1", "score": 10.0}, path)
    append_row({"run_id": "r2", "score": 20.0}, path)
    rows = read_rows(path)
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert rows[1]["score"] == 20.0


def test_read_missing_file(tmp_path: Path) -> None:
    """Reading a nonexistent ledger returns an empty list."""
    assert read_rows(tmp_path / "nope.jsonl") == []
