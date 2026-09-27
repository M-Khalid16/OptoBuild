from __future__ import annotations

import logging

from optobuild.core.log import component_logger, configure_logging, get_logger


def test_logger_names() -> None:
    assert get_logger("engine").name == "optobuild.engine"
    assert get_logger("optobuild.engine").name == "optobuild.engine"
    assert component_logger("fiber").name == "optobuild.run.fiber"


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger("optobuild")
    before = len(root.handlers)
    configure_logging("DEBUG")
    configure_logging("INFO")
    assert len(root.handlers) <= before + 1
    assert root.level == logging.INFO
