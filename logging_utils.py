"""Shared logging configuration helpers."""
from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(*, level: int = logging.INFO, format: Optional[str] = None) -> None:
    """Configure root logging with a consistent format."""

    logging.basicConfig(level=level, format=format or _DEFAULT_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the shared configuration."""

    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
