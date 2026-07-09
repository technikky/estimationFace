"""Logging configuration.

Logs go to both the console and a dated file under ``logs/YYYY-MM-DD.log``.
"""
from __future__ import annotations

import logging
import sys
from datetime import date

from .paths import LOGS_DIR, ensure_dir

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the ``estimationface`` logger once; safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger("estimationface")
    if _CONFIGURED:
        return logger

    ensure_dir(LOGS_DIR)
    log_file = LOGS_DIR / f"{date.today().isoformat()}.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    _CONFIGURED = True
    logger.info("Logging initialised -> %s", log_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"estimationface.{name}")
