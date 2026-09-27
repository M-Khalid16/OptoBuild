"""Declarative metadata for components: ports, parameters, categories (ADR-0004).

These records are pure data. They are consumed by the graph (port typing),
by persistence (parameter serialization), by the Phase 1 parameter validator,
and later by the GUI to generate forms automatically, so components never
hand-write GUI code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from optobuild.signals.kinds import SignalKind


class ComponentCategory(enum.Enum):
    """Library category of a component (used for palettes and documentation)."""

    SOURCE = "source"
    MODULATOR = "modulator"
    CHANNEL = "channel"
    AMPLIFIER = "amplifier"
    PASSIVE = "passive"
    DETECTOR = "detector"
    ELECTRICAL = "electrical"
    DSP = "dsp"
    ANALYZER = "analyzer"
    PHOTONIC_CIRCUIT = "photonic_circuit"
    LASER = "laser"


@dataclass(frozen=True)
class PortSpec:
    """A typed input or output port.

    Attributes
    ----------
    name:
        Identifier, unique among the component's inputs (resp. outputs).
    kind:
        Signal kind carried by the port.
    description:
        One-line human-readable description.
    tap:
        Inputs only. A *tap* input observes a signal without consuming it
        (power meters, spectrum and eye analyzers). An OPTICAL output may feed
        any number of tap inputs but at most one non-tap input; physical
        power splitting requires an explicit splitter component (ADR-0004).
    optional:
        Inputs only. The component can run with this input unconnected.
    """

    name: str
    kind: SignalKind
    description: str = ""
    tap: bool = False
    optional: bool = False


class ParameterType(enum.Enum):
    """Value type of a parameter."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"
    STRING = "string"


@dataclass(frozen=True)
class ParameterSpec:
    """Schema of one component parameter.

    Numeric values are always stored in the SI ``unit`` (ADR-0003).
    ``display_unit`` only tells a user interface how to present the value;
    conversion goes through ``optobuild.core.units`` (Phase 1).

    Attributes
    ----------
    name:
        Identifier used in code and project files.
    type:
        Value type.
    default:
        Default value in SI units; ``None`` means the parameter is required.
    unit:
        SI unit symbol of the stored value (``"1"`` for dimensionless).
    display_unit:
        Preferred presentation unit, e.g. ``"dBm"``, ``"nm"``, ``"ps/(nm km)"``.
    minimum, maximum:
        Allowed range in SI units (``None`` = unbounded).
    minimum_inclusive, maximum_inclusive:
        Whether the bounds are allowed values.
    choices:
        Allowed values for ``ParameterType.CHOICE``.
    symbol:
        Mathematical symbol used in docs/physics_models.md, e.g. ``"V_pi"``.
    description:
        Human-readable description including physical meaning.
    """

    name: str
    type: ParameterType
    default: Any = None
    unit: str = "1"
    display_unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    choices: tuple[Any, ...] = ()
    symbol: str = ""
    description: str = ""

    @property
    def required(self) -> bool:
        """True if the parameter has no default and must be supplied."""
        return self.default is None


__all__ = ["ComponentCategory", "ParameterSpec", "ParameterType", "PortSpec"]
