"""Domain-specific exception hierarchy.

Every error raised by OptoBuild derives from :class:`OptoBuildError`. Messages
must state (1) what is wrong and (2) how the user can correct it; the optional
``hint`` argument carries the corrective action so that user interfaces can
display it separately.
"""

from __future__ import annotations


class OptoBuildError(Exception):
    """Base class for all OptoBuild errors.

    Parameters
    ----------
    message:
        Description of the problem, including the offending value(s).
    hint:
        Corrective action the user can take. Appended to ``str(error)``.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message if hint is None else f"{message} Hint: {hint}")


# --- configuration / modelling errors -------------------------------------------------


class InvalidParameterError(OptoBuildError, ValueError):
    """A component or model parameter is missing, of the wrong type, or out of range."""


class UnitError(OptoBuildError, ValueError):
    """A physical unit is unknown or dimensionally incompatible with the requested one."""


class SignalTypeError(OptoBuildError, TypeError):
    """A signal of the wrong kind was supplied (e.g. electrical where optical is required)."""


# --- graph errors ------------------------------------------------------------------------


class InvalidGraphError(OptoBuildError):
    """The simulation graph is structurally invalid (unknown node, dangling port, ...)."""


class PortTypeMismatchError(InvalidGraphError, TypeError):
    """Two ports with incompatible signal kinds were connected."""


class SimulationCycleError(InvalidGraphError):
    """A feed-forward graph contains a cycle; feedback needs the iterative executor."""


# --- numerical errors --------------------------------------------------------------------


class SamplingError(OptoBuildError, ValueError):
    """Sampling is invalid: inconsistent rates, too few samples, aliasing, short window."""


class NumericalStabilityError(OptoBuildError, ArithmeticError):
    """A numerical method left its stable/accurate regime (step size, NaN/Inf, ...)."""


# --- execution errors --------------------------------------------------------------------


class SimulationCancelledError(OptoBuildError):
    """A running simulation was cancelled by the user."""


class ComponentExecutionError(OptoBuildError):
    """A component failed while running; wraps the original exception as ``__cause__``."""


# --- persistence errors ------------------------------------------------------------------


class ProjectFormatError(OptoBuildError, ValueError):
    """A project file is malformed, has an unsupported schema version, or unknown content."""


__all__ = [
    "ComponentExecutionError",
    "InvalidGraphError",
    "InvalidParameterError",
    "NumericalStabilityError",
    "OptoBuildError",
    "PortTypeMismatchError",
    "ProjectFormatError",
    "SamplingError",
    "SignalTypeError",
    "SimulationCancelledError",
    "SimulationCycleError",
    "UnitError",
]
