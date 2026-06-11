"""Loguru logging setup with a run id bound to every line."""

import sys
from pathlib import Path

from loguru import logger

FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{extra[run_id]}</cyan> | {message}"
)


def setup_logging(run_id: str, log_dir: Path | None = None) -> None:
    """Configure loguru sinks for a run.

    Args:
        run_id: Identifier bound to every log line.
        log_dir: If given, also log to ``{log_dir}/run.log``.
    """
    logger.remove()
    logger.configure(extra={"run_id": run_id})
    logger.add(sys.stderr, format=FORMAT)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "run.log", format=FORMAT)
