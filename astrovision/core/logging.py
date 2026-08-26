"""Lightweight structured logging shared by every subsystem."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_ROOT = "astrovision"
_CONFIGURED = False

LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


class _Formatter(logging.Formatter):
    """Compact single-line formatter: ``12:00:03 INFO detect.sources | msg``."""

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        name = record.name
        if name.startswith(_ROOT + "."):
            name = name[len(_ROOT) + 1:]
        elif name == _ROOT:
            name = "core"
        return f"{stamp} {record.levelname:<7} {name:<22} | {record.getMessage()}"


def configure(level: str = "info", stream=None) -> None:
    """Attach a single stderr handler to the ``astrovision`` logger tree."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_Formatter())
    logger.addHandler(handler)
    logger.setLevel(LEVELS.get(str(level).lower(), logging.INFO))
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str = _ROOT) -> logging.Logger:
    """Return a namespaced logger, configuring the tree on first use."""
    if not _CONFIGURED:
        configure(os.environ.get("ASTROVISION_LOG_LEVEL", "info"))
    if name != _ROOT and not name.startswith(_ROOT + "."):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change the verbosity of the whole ``astrovision`` logger tree."""
    logging.getLogger(_ROOT).setLevel(LEVELS.get(str(level).lower(), logging.INFO))


@contextmanager
def timed(message: str, logger: Optional[logging.Logger] = None,
          level: int = logging.INFO) -> Iterator[None]:
    """Log ``message`` with the wall-clock duration of the wrapped block."""
    log = logger or get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.log(level, "%s (%.3fs)", message, elapsed)
