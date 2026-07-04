"""Structured logging setup shared by the CLI, trainer, and serving app."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger with console (and optional file) handlers.

    Idempotent: existing handlers installed by a previous call are replaced, so
    repeated CLI invocations in one process never duplicate output.

    Args:
        level: logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_file: when given, also write logs to this file (parents created).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party chatter that would drown our own logs at INFO.
    for noisy in ("matplotlib", "PIL", "fsspec", "git", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (thin wrapper for consistency)."""
    return logging.getLogger(name)
