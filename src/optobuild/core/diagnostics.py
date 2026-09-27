"""Structured diagnostics.

Validation code (parameter checks, sampling checks, graph checks) reports
problems as :class:`Diagnostic` records instead of printing or silently
clamping values. Callers decide whether a diagnostic is fatal:
``Severity.ERROR`` diagnostics must stop a simulation; ``WARNING`` diagnostics
(e.g. aliasing risk) must be surfaced to the user with the results.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Severity(enum.Enum):
    """Severity of a diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Diagnostic:
    """One finding produced by a validation step.

    Attributes
    ----------
    severity:
        How serious the finding is.
    code:
        Stable machine-readable identifier, e.g. ``"sampling.aliasing_risk"``.
    message:
        What is wrong, including the offending values and their units.
    hint:
        How the user can fix it.
    source:
        Name of the component/signal/graph element concerned, if any.
    """

    severity: Severity
    code: str
    message: str
    hint: str = ""
    source: str = ""

    def __str__(self) -> str:
        where = f"[{self.source}] " if self.source else ""
        hint = f" Hint: {self.hint}" if self.hint else ""
        return f"{self.severity.value.upper()} {self.code}: {where}{self.message}{hint}"


def has_errors(diagnostics: list[Diagnostic] | tuple[Diagnostic, ...]) -> bool:
    """Return ``True`` if any diagnostic has :attr:`Severity.ERROR`."""
    return any(d.severity is Severity.ERROR for d in diagnostics)


__all__ = ["Diagnostic", "Severity", "has_errors"]
