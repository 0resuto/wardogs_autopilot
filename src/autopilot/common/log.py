"""Centralized logging configuration for WARDOGS Autopilot.

Writes formatted log records to both the console (when available) and to
output/autopilot.log. Complements crashlog.py (which intercepts segfaults and
unhandled exceptions).
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_INITIALIZED = False


def setup_logging(
    log_dir: str = "output",
    log_filename: str = "autopilot.log",
    *,
    debug: bool = False,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure application logging with file rotation and console output.

    Safe to call multiple times.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    log_level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 1. Rotating file handler (always writes to disk, even under pythonw.exe)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # 2. Console handler (if stdout/stderr are interactive or not /dev/null)
    if sys.stderr and hasattr(sys.stderr, "write"):
        try:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)
        except Exception:
            pass

    _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger instance with the given module name."""
    return logging.getLogger(name or "autopilot")
