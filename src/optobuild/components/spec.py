"""Declarative metadata for components: ports, parameters, categories (ADR-0004).

These records are pure data. They are consumed by the graph (port typing),
by persistence (parameter serialization), by the Phase 1 parameter validator,
and later by the GUI to generate forms automatically, so components never
hand-write GUI code.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

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

    def coerce(self, value: Any) -> Any:
        """Validate ``value`` against this schema and return it in canonical type.

        Raises ``ValueError`` with a message naming the parameter, the value,
        the expected type/range and the SI unit. The caller (component base
        class) turns it into a Diagnostic / ``InvalidParameterError``.
        """
        t = self.type
        if t is ParameterType.BOOL:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"'{self.name}' must be a boolean, got {value!r}.")
            return bool(value)
        if t is ParameterType.STRING:
            if not isinstance(value, str):
                raise ValueError(f"'{self.name}' must be a string, got {value!r}.")
            return self._check_choice(value)
        if t is ParameterType.CHOICE:
            return self._check_choice(value)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"'{self.name}' must be a number [{self.unit}], got {value!r}.")
        if t is ParameterType.INT:
            if isinstance(value, (float, np.floating)) and not float(value).is_integer():
                raise ValueError(f"'{self.name}' must be an integer, got {value!r}.")
            number: float | int = int(value)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"'{self.name}' must be finite, got {value!r}.")
        self._check_range(number)
        return number

    def _check_choice(self, value: Any) -> Any:
        if self.choices and value not in self.choices:
            raise ValueError(f"'{self.name}' must be one of {list(self.choices)}, got {value!r}.")
        return value

    def _check_range(self, number: float) -> None:
        lo, hi = self.minimum, self.maximum
        if lo is not None and (number < lo or (number == lo and not self.minimum_inclusive)):
            op = ">=" if self.minimum_inclusive else ">"
            raise ValueError(
                f"'{self.name}' must be {op} {lo:g} {self.unit}, got {number:g} {self.unit}."
            )
        if hi is not None and (number > hi or (number == hi and not self.maximum_inclusive)):
            op = "<=" if self.maximum_inclusive else "<"
            raise ValueError(
                f"'{self.name}' must be {op} {hi:g} {self.unit}, got {number:g} {self.unit}."
            )


__all__ = ["ComponentCategory", "ParameterSpec", "ParameterType", "PortSpec"]
