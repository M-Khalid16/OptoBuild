"""Logging helpers.

The library only *emits* log records under the ``optobuild`` logger namespace
and never configures handlers on import (standard library practice).
Applications (CLI, GUI, examples) call :func:`configure_logging` once.
"""

from __future__ import annotations

import logging

ROOT_LOGGER_NAME = "optobuild"
RUN_LOGGER_PREFIX = f"{ROOT_LOGGER_NAME}.run"


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``optobuild`` namespace (``name`` without the prefix is allowed)."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def component_logger(component_name: str) -> logging.Logger:
    """Logger for one component instance: ``optobuild.run.<component_name>``."""
    return logging.getLogger(f"{RUN_LOGGER_PREFIX}.{component_name}")


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Attach a single stream handler to the ``optobuild`` logger (idempotent)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_optobuild", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        handler._optobuild = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


logging.getLogger(ROOT_LOGGER_NAME).addHandler(logging.NullHandler())

__all__ = ["component_logger", "configure_logging", "get_logger"]
