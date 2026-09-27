"""Parameter-form models generated from ``ParameterSpec`` (Qt-free).

The GUI never hand-codes forms: each component parameter becomes a
:class:`FieldModel` that knows its widget kind, label, tooltip and how to
convert between the stored SI value and the text shown in the display unit.
All unit conversion goes through ``optobuild.core.units`` (ADR-0003); nothing
here performs physics.

Text entered by the user may be a bare number (interpreted in the display
unit) or a number with any compatible unit, e.g. ``"1.55 um"`` for a
wavelength shown in nm. Expressions are never evaluated.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any

from optobuild.components.spec import ParameterSpec, ParameterType
from optobuild.core.errors import InvalidParameterError, UnitError
from optobuild.core.units import from_si, get_unit, parse_quantity, to_si

_COMPATIBLE = {("dimensionless", "ratio"), ("ratio", "dimensionless")}


class WidgetKind(enum.Enum):
    """Which editor a field needs."""

    TEXT_NUMBER = "number"
    INTEGER = "integer"
    CHECKBOX = "checkbox"
    CHOICE = "choice"
    TEXT = "text"


def _dimension(unit: str) -> str | None:
    try:
        return get_unit(unit).dimension
    except UnitError:
        return None  # e.g. "input": no conversion possible


def _format(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.12g}"


@dataclass(frozen=True)
class FieldModel:
    """Presentation model of one parameter."""

    spec: ParameterSpec

    @property
    def name(self) -> str:
        """Parameter name."""
        return self.spec.name

    @property
    def widget(self) -> WidgetKind:
        """Editor kind."""
        return {
            ParameterType.FLOAT: WidgetKind.TEXT_NUMBER,
            ParameterType.INT: WidgetKind.INTEGER,
            ParameterType.BOOL: WidgetKind.CHECKBOX,
            ParameterType.CHOICE: WidgetKind.CHOICE,
            ParameterType.STRING: WidgetKind.TEXT,
        }[self.spec.type]

    @property
    def unit(self) -> str:
        """Unit shown next to the field (display unit if convertible, else SI)."""
        d = self.spec.display_unit
        return d if d and self._convertible(d) else self.spec.unit

    def _convertible(self, unit: str) -> bool:
        a, b = _dimension(self.spec.unit), _dimension(unit)
        return a is not None and b is not None and (a == b or (a, b) in _COMPATIBLE)

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``"v_pi (V_pi) [V]"``."""
        sym = f" ({self.spec.symbol})" if self.spec.symbol else ""
        unit = self.unit
        suffix = f" [{unit}]" if unit not in ("1", "", "input") else ""
        if unit == "input":
            suffix = " [A or V]"
        return f"{self.spec.name}{sym}{suffix}"

    @property
    def tooltip(self) -> str:
        """Description, stored SI unit and allowed range."""
        s = self.spec
        parts = [s.description] if s.description else []
        if s.type in (ParameterType.FLOAT, ParameterType.INT):
            lo = "-inf" if s.minimum is None else f"{s.minimum:g}"
            hi = "inf" if s.maximum is None else f"{s.maximum:g}"
            lb = "[" if s.minimum_inclusive else "("
            rb = "]" if s.maximum_inclusive else ")"
            parts.append(f"Stored in SI unit '{s.unit}', allowed {lb}{lo}, {hi}{rb}.")
        if s.choices:
            parts.append(f"Choices: {', '.join(map(str, s.choices))}.")
        return " ".join(parts)

    @property
    def choices(self) -> tuple[Any, ...]:
        """Choices for CHOICE fields."""
        return self.spec.choices

    # --- conversion ----------------------------------------------------------------------
    def to_display(self, si_value: Any) -> Any:
        """Stored SI value -> value/text for the editor."""
        if self.spec.type is not ParameterType.FLOAT:
            return si_value
        if self.unit == self.spec.unit:
            return _format(float(si_value))
        if si_value == 0 and self.unit in ("dBm", "dB"):
            return "-inf"
        return _format(float(from_si(si_value, self.unit)))

    def from_display(self, value: Any) -> Any:
        """Editor value/text -> SI value (not yet range-checked; the component does that).

        Raises ``InvalidParameterError`` with a user-readable message.
        """
        if self.spec.type is ParameterType.FLOAT:
            return self._parse_float(str(value))
        if self.spec.type is ParameterType.INT:
            try:
                return int(str(value).strip())
            except ValueError:
                raise InvalidParameterError(
                    f"'{self.name}' must be an integer, got {value!r}."
                ) from None
        return value

    def _parse_float(self, text: str) -> float:
        t = text.strip()
        if t in ("-inf", "−inf") and self.unit in ("dBm", "dB"):
            return 0.0
        try:
            number, unit = float(t), self.unit  # bare number: display unit
        except ValueError:
            try:
                number, unit = parse_quantity(t)
            except UnitError as exc:
                raise InvalidParameterError(
                    f"'{self.name}': {exc.message}",
                    hint=f"Enter a number in {self.unit} or a number with a unit.",
                ) from None
        if not math.isfinite(number):
            raise InvalidParameterError(f"'{self.name}' must be finite, got {text!r}.")
        if unit in (self.spec.unit, "input"):
            return float(number)
        if not self._convertible(unit):
            raise InvalidParameterError(
                f"'{self.name}' cannot be given in '{unit}'.",
                hint=f"Use {self.unit} or another unit of the same dimension.",
            )
        return float(to_si(number, unit))


def fields_for(specs: tuple[ParameterSpec, ...]) -> list[FieldModel]:
    """Field models for a component's parameter specs, in declaration order."""
    return [FieldModel(s) for s in specs]


__all__ = ["FieldModel", "WidgetKind", "fields_for"]
