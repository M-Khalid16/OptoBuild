"""The component interface (ADR-0004).

A component is a thin, stateless wrapper that:

1. declares metadata (type id, version, name, category, ports, parameters);
2. validates its parameters (Phase 1);
3. in :meth:`Component.run`, maps input signals to output signals by calling
   physics / solver / analysis functions. It must not re-implement equations.

``run`` must be a deterministic function of (parameters, inputs, context.rng):
no hidden global state, no reading of wall-clock time, no un-seeded randomness.
This is what makes caching and reproducibility possible (ADR-0005, ADR-0008).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from optobuild.components.spec import ComponentCategory, ParameterSpec, PortSpec
from optobuild.core.diagnostics import Diagnostic
from optobuild.core.errors import InvalidGraphError

TYPE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
"""Type ids are dotted lowercase namespaces, e.g. ``optobuild.source.cw_laser``."""


@runtime_checkable
class RunContext(Protocol):
    """Services the execution engine provides to a running component.

    Defined here (layer 5) rather than in the engine (layer 7) so that
    components never import the engine; the engine implements this protocol.
    """

    @property
    def rng(self) -> np.random.Generator:
        """Component-private generator derived from the project seed (ADR-0008)."""
        ...

    @property
    def logger(self) -> logging.Logger:
        """Logger scoped to the component instance."""
        ...

    def check_cancelled(self) -> None:
        """Raise ``SimulationCancelledError`` if the user cancelled the run."""
        ...

    def report_progress(self, fraction: float, message: str = "") -> None:
        """Report progress in [0, 1] for long-running components."""
        ...


class Component(ABC):
    """Abstract base class of all simulation components.

    Concrete subclasses must define the class attributes below and implement
    :meth:`run`. Ports may be made parameter-dependent (e.g. an N-way combiner)
    by overriding :meth:`inputs` / :meth:`outputs`.
    """

    type_id: ClassVar[str]
    """Globally unique, stable identifier used in project files."""
    version: ClassVar[str]
    """Model version (semantic versioning); bump when numerical output changes."""
    display_name: ClassVar[str]
    """Human-readable name."""
    category: ClassVar[ComponentCategory]
    input_ports: ClassVar[tuple[PortSpec, ...]] = ()
    output_ports: ClassVar[tuple[PortSpec, ...]] = ()
    parameter_specs: ClassVar[tuple[ParameterSpec, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # intermediate abstract classes need not be complete
        for attr in ("type_id", "version", "display_name", "category"):
            if not hasattr(cls, attr):
                raise TypeError(f"Component subclass {cls.__name__} must define '{attr}'.")
        if not TYPE_ID_PATTERN.match(cls.type_id):
            raise TypeError(
                f"{cls.__name__}.type_id={cls.type_id!r} is invalid; use a dotted lowercase "
                "namespace such as 'optobuild.source.cw_laser'."
            )
        for group, specs in (
            ("input_ports", cls.input_ports),
            ("output_ports", cls.output_ports),
            ("parameter_specs", cls.parameter_specs),
        ):
            names = [s.name for s in specs]
            if len(names) != len(set(names)):
                raise TypeError(f"{cls.__name__}.{group} contains duplicate names: {names}.")

    def __init__(self, name: str, parameters: Mapping[str, Any] | None = None) -> None:
        if not name or not isinstance(name, str):
            raise InvalidGraphError(
                "A component instance needs a non-empty string name.",
                hint="Give each component a unique name within its graph.",
            )
        self._name = name
        self._parameters: dict[str, Any] = dict(parameters or {})

    @property
    def name(self) -> str:
        """Instance name, unique within a graph."""
        return self._name

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Read-only view of the parameter values (SI units)."""
        return MappingProxyType(self._parameters)

    def inputs(self) -> tuple[PortSpec, ...]:
        """Input ports of this instance (default: the class declaration)."""
        return type(self).input_ports

    def outputs(self) -> tuple[PortSpec, ...]:
        """Output ports of this instance (default: the class declaration)."""
        return type(self).output_ports

    def validate(self) -> list[Diagnostic]:
        """Return diagnostics for the current parameters.

        Phase 0 defines the hook only. Phase 1 adds schema-driven validation
        (types, ranges, required values) here; subclasses extend it with
        cross-parameter physical checks and must call ``super().validate()``.
        """
        return []

    @abstractmethod
    def run(self, inputs: Mapping[str, Any], context: RunContext) -> Mapping[str, Any]:
        """Compute output signals (keyed by output port name) from input signals."""

    @classmethod
    def documentation(cls) -> str:
        """Documentation hook: the class docstring (model equations live in physics docs)."""
        return (cls.__doc__ or "").strip()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


__all__ = ["TYPE_ID_PATTERN", "Component", "RunContext"]
