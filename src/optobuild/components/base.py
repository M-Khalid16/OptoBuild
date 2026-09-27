"""The component interface (ADR-0004).

A component is a thin, stateless wrapper that:

1. declares metadata (type id, version, name, category, ports, parameters);
2. validates its parameters against the schema at construction time, so a
   component instance always holds valid, canonical (SI) parameter values;
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

from optobuild.components.spec import ComponentCategory, ParameterSpec, ParameterType, PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import InvalidGraphError, InvalidParameterError, SamplingError
from optobuild.numerics.layout import SimulationLayout

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

    @property
    def layout(self) -> SimulationLayout | None:
        """Global simulation layout of the run, or ``None`` (ADR-0011)."""
        ...

    def check_cancelled(self) -> None:
        """Raise ``SimulationCancelledError`` if the user cancelled the run."""
        ...

    def report_progress(self, fraction: float, message: str = "") -> None:
        """Report progress in [0, 1] for long-running components."""
        ...

    def record(self, key: str, value: Any) -> None:
        """Store a named analysis result (numbers, arrays, dataclasses) (ADR-0009).

        Used by analyzers/sinks, which produce results rather than signals.
        """
        ...

    def warn(self, diagnostic: Diagnostic) -> None:
        """Attach a run-time diagnostic (e.g. aliasing risk) to the results."""
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
        given = dict(parameters or {})
        known = {spec.name: spec for spec in type(self).parameter_specs}
        unknown = sorted(set(given) - set(known))
        if unknown:
            raise InvalidParameterError(
                f"Component '{name}' ({type(self).type_id}) has no parameter(s) {unknown}.",
                hint=f"Valid parameters: {sorted(known)}.",
            )
        values: dict[str, Any] = {}
        problems: list[str] = []
        for spec in known.values():
            if spec.name in given:
                raw = given[spec.name]
            elif spec.required:
                problems.append(f"'{spec.name}' is required [{spec.unit}]")
                continue
            else:
                raw = spec.default
            try:
                values[spec.name] = spec.coerce(raw)
            except ValueError as exc:
                problems.append(str(exc))
        if problems:
            raise InvalidParameterError(
                f"Invalid parameters for component '{name}' ({type(self).type_id}): "
                + "; ".join(problems),
                hint="Correct the listed values; numeric values are in SI units.",
            )
        self._parameters = values
        diagnostics = self.validate()
        errors = [d for d in diagnostics if d.severity is Severity.ERROR]
        if errors:
            raise InvalidParameterError(
                f"Invalid parameters for component '{name}' ({type(self).type_id}): "
                + "; ".join(d.message for d in errors),
                hint=" ".join(d.hint for d in errors if d.hint) or None,
            )
        self._diagnostics = tuple(diagnostics)

    @property
    def name(self) -> str:
        """Instance name, unique within a graph."""
        return self._name

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Read-only view of the parameter values (SI units)."""
        return MappingProxyType(self._parameters)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Non-fatal diagnostics (warnings/info) found when the component was created."""
        return self._diagnostics

    def with_parameters(self, **changes: Any) -> Component:
        """New instance with the same name and some parameters changed (validated)."""
        merged = dict(self._parameters)
        merged.update(changes)
        return type(self)(self._name, merged)

    @classmethod
    def parameter_spec(cls, name: str) -> ParameterSpec:
        """Schema of parameter ``name``."""
        for spec in cls.parameter_specs:
            if spec.name == name:
                return spec
        raise InvalidParameterError(f"{cls.type_id} has no parameter '{name}'.")

    def inputs(self) -> tuple[PortSpec, ...]:
        """Input ports of this instance (default: the class declaration)."""
        return type(self).input_ports

    def outputs(self) -> tuple[PortSpec, ...]:
        """Output ports of this instance (default: the class declaration)."""
        return type(self).output_ports

    def validate(self) -> list[Diagnostic]:
        """Cross-parameter physical checks, run after schema validation.

        Schema checks (types, ranges, required values) already happened in
        ``__init__``. Subclasses override this to add checks that involve
        several parameters, returning ``ERROR`` diagnostics for invalid
        combinations (which make construction fail) and ``WARNING``s for
        questionable ones (kept in :attr:`diagnostics`).
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


TIMING_SOURCE_SPEC = ParameterSpec(
    "timing_source",
    ParameterType.CHOICE,
    default="parameters",
    choices=("parameters", "layout"),
    description="Take bit rate / sequence length / sampling from this component's own "
    "parameters or from the global simulation layout (ADR-0011)",
)
"""Shared parameter of source components that can follow the global layout."""


def require_layout(context: RunContext, component_name: str) -> SimulationLayout:
    """The run's layout, or a ``SamplingError`` explaining how to provide one."""
    layout = getattr(context, "layout", None)
    if layout is None:
        raise SamplingError(
            f"'{component_name}' has timing_source='layout' but the simulation has no layout.",
            hint="Set a layout (bit_rate, n_bits, samples_per_bit) on the project or pass "
            "layout= to the executor, or use timing_source='parameters'.",
        )
    return layout


__all__ = ["TIMING_SOURCE_SPEC", "TYPE_ID_PATTERN", "Component", "RunContext", "require_layout"]
