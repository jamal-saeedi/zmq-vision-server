"""Structured logging configuration for zmq_inference."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    logger_name: str = "zmq_inference",
) -> logging.Logger:
    """Configure and return a named logger with a clean, structured format.

    The format includes the timestamp, logger name, level, and message so
    that log lines are easy to parse by log-aggregation tools.

    Args:
        level:       Logging level string (``"DEBUG"``, ``"INFO"``, etc.).
        fmt:         Override the default format string.
        logger_name: Name of the logger to configure.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers when called multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``zmq_inference`` namespace.

    Example::

        logger = get_logger(__name__)
        logger.info("Server started on port %d", config.port)
    """
    return logging.getLogger(f"zmq_inference.{name}")
