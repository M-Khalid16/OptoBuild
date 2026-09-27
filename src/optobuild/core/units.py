"""Centralized unit conversion (ADR-0003).

All simulation code works in SI. This module is the *only* place where
presentation units (nm, THz, dBm, ps/(nm km), ...) are converted.

Two interfaces are provided:

* explicit functions (``dbm_to_watt``, ``dispersion_to_beta2``, ...) for code;
* a whitelisted unit table (:func:`to_si`, :func:`from_si`,
  :func:`parse_quantity`) for user input and project files. Strings are
  tokenized by a strict regular expression and looked up in the table; they
  are never evaluated.

Conventions
-----------
* ``dB`` for power ratios: ``r = 10**(x/10)``; ``dB loss`` for transmissions
  expressed as a positive loss: ``r = 10**(-x/10)`` (5 dB loss -> r = 0.316).
* ``dBm``: ``P = 1 mW * 10**(x/10)``.
* ``dB/km`` is converted to the *power* attenuation coefficient
  ``alpha [1/m]`` with ``P(L) = P(0) exp(-alpha L)``:
  ``alpha = x * ln(10) / 10 / 1000``.
* Chromatic dispersion ``D`` [s/m^2]: 1 ps/(nm km) = 1e-6 s/m^2.
* ``beta2 = -D lambda^2 / (2 pi c)``;
  ``beta3 = (lambda / (2 pi c))^2 (lambda^2 S + 2 lambda D)`` with slope
  ``S`` [s/m^3] (1 ps/(nm^2 km) = 1e3 s/m^3). Reference: G. P. Agrawal,
  *Fiber-Optic Communication Systems*, 5th ed., sec. 2.3.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import UnitError

_LN10_OVER_10 = math.log(10.0) / 10.0


# --- explicit conversion functions ---------------------------------------------------------


def db_to_linear(db: ArrayLike) -> np.ndarray | float:
    """Power ratio in dB -> linear ratio."""
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)[()]


def linear_to_db(ratio: ArrayLike) -> np.ndarray | float:
    """Linear power ratio (> 0) -> dB."""
    r = np.asarray(ratio, dtype=float)
    if np.any(r <= 0):
        raise UnitError(f"Cannot express non-positive ratio {ratio!r} in dB.")
    return (10.0 * np.log10(r))[()]


def dbm_to_watt(dbm: ArrayLike) -> np.ndarray | float:
    """Power in dBm -> W."""
    return (1e-3 * np.power(10.0, np.asarray(dbm, dtype=float) / 10.0))[()]


def watt_to_dbm(watt: ArrayLike) -> np.ndarray | float:
    """Power in W (> 0) -> dBm."""
    p = np.asarray(watt, dtype=float)
    if np.any(p <= 0):
        raise UnitError(
            f"Cannot express non-positive power {watt!r} W in dBm.",
            hint="dBm is only defined for P > 0.",
        )
    return (10.0 * np.log10(p / 1e-3))[()]


def db_per_km_to_per_m(db_per_km: ArrayLike) -> np.ndarray | float:
    """Attenuation in dB/km -> power attenuation coefficient alpha [1/m]."""
    return (np.asarray(db_per_km, dtype=float) * _LN10_OVER_10 / 1000.0)[()]


def per_m_to_db_per_km(alpha: ArrayLike) -> np.ndarray | float:
    """Power attenuation coefficient alpha [1/m] -> dB/km."""
    return (np.asarray(alpha, dtype=float) * 1000.0 / _LN10_OVER_10)[()]


def wavelength_to_frequency(wavelength: ArrayLike) -> np.ndarray | float:
    """Vacuum wavelength [m] -> optical frequency [Hz]."""
    lam = np.asarray(wavelength, dtype=float)
    if np.any(lam <= 0):
        raise UnitError(f"Wavelength must be > 0 m, got {wavelength!r}.")
    return (SPEED_OF_LIGHT / lam)[()]


def frequency_to_wavelength(frequency: ArrayLike) -> np.ndarray | float:
    """Optical frequency [Hz] -> vacuum wavelength [m]."""
    nu = np.asarray(frequency, dtype=float)
    if np.any(nu <= 0):
        raise UnitError(f"Frequency must be > 0 Hz, got {frequency!r}.")
    return (SPEED_OF_LIGHT / nu)[()]


def dispersion_to_beta2(dispersion: float, wavelength: float) -> float:
    """D [s/m^2] at vacuum wavelength [m] -> beta2 [s^2/m]: beta2 = -D lambda^2/(2 pi c)."""
    return -dispersion * wavelength**2 / (2.0 * math.pi * SPEED_OF_LIGHT)


def beta2_to_dispersion(beta2: float, wavelength: float) -> float:
    """beta2 [s^2/m] -> D [s/m^2] (inverse of :func:`dispersion_to_beta2`)."""
    return -beta2 * 2.0 * math.pi * SPEED_OF_LIGHT / wavelength**2


def dispersion_slope_to_beta3(dispersion: float, slope: float, wavelength: float) -> float:
    """(D [s/m^2], S [s/m^3], lambda [m]) -> beta3 [s^3/m]."""
    a = wavelength / (2.0 * math.pi * SPEED_OF_LIGHT)
    return a**2 * (wavelength**2 * slope + 2.0 * wavelength * dispersion)


# --- whitelisted unit table ----------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """A unit known to the table.

    ``to_si``/``from_si`` convert a float or array between this unit and SI.
    """

    symbol: str
    dimension: str
    to_si: Callable[[np.ndarray], np.ndarray]
    from_si: Callable[[np.ndarray], np.ndarray]


def _scale(symbol: str, dimension: str, factor: float) -> Unit:
    return Unit(symbol, dimension, lambda v, f=factor: v * f, lambda v, f=factor: v / f)


_UNITS: dict[str, Unit] = {}


def _add(unit: Unit, *aliases: str) -> None:
    for s in (unit.symbol, *aliases):
        _UNITS[s] = unit


_PREFIXED = {
    "time": ("s", {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}),
    "length": (
        "m",
        {"": 1.0, "k": 1e3, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12},
    ),
    "frequency": ("Hz", {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}),
    "power": ("W", {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9}),
    "voltage": ("V", {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6}),
    "current": ("A", {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12}),
    "bit_rate": ("bit/s", {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}),
    "symbol_rate": ("Bd", {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9}),
}
for _dim, (_base, _prefixes) in _PREFIXED.items():
    for _p, _f in _prefixes.items():
        _add(_scale(_p + _base, _dim, _f))
for _p in ("", "k", "M", "G", "T"):
    _add(_scale(_p + "b/s", "bit_rate", _PREFIXED["bit_rate"][1][_p]), _p + "bps")

_add(_scale("1", "dimensionless", 1.0), "")
_add(_scale("rad", "angle", 1.0))
_add(_scale("deg", "angle", math.pi / 180.0))
_add(_scale("K", "temperature", 1.0))
_add(_scale("ohm", "resistance", 1.0), "Ω", "Ohm")
_add(_scale("kohm", "resistance", 1e3), "kΩ")
_add(_scale("A/W", "responsivity", 1.0))
_add(_scale("s/m^2", "dispersion", 1.0))
_add(_scale("ps/(nm km)", "dispersion", 1e-6), "ps/nm/km", "ps/(nm·km)", "ps/nm-km")
_add(_scale("s/m^3", "dispersion_slope", 1.0))
_add(_scale("ps/(nm^2 km)", "dispersion_slope", 1e3), "ps/nm^2/km")
_add(_scale("s^2/m", "beta2", 1.0))
_add(_scale("ps^2/km", "beta2", 1e-27))
_add(_scale("s^3/m", "beta3", 1.0))
_add(_scale("ps^3/km", "beta3", 1e-39))
_add(_scale("1/m", "attenuation", 1.0))
_add(_scale("1/(W m)", "nonlinear_coefficient", 1.0))
_add(_scale("1/(W km)", "nonlinear_coefficient", 1e-3), "1/W/km")
_add(_scale("m^2/W", "nonlinear_index", 1.0))
_add(_scale("m^2", "area", 1.0))
_add(_scale("um^2", "area", 1e-12), "µm^2")
_add(Unit("dB/km", "attenuation", lambda v: db_per_km_to_per_m(v), lambda v: per_m_to_db_per_km(v)))
_add(Unit("dBm", "power", lambda v: dbm_to_watt(v), lambda v: watt_to_dbm(v)))
_add(_scale("ratio", "ratio", 1.0))
_add(Unit("dB", "ratio", lambda v: db_to_linear(v), lambda v: linear_to_db(v)))
_add(
    Unit(
        "dB loss",
        "ratio",
        lambda v: db_to_linear(-np.asarray(v, dtype=float)),
        lambda v: -np.asarray(linear_to_db(v)),
    )
)


def get_unit(symbol: str) -> Unit:
    """Look up a unit by symbol; raises :class:`UnitError` for unknown symbols."""
    try:
        return _UNITS[symbol.strip()]
    except KeyError:
        raise UnitError(
            f"Unknown unit {symbol!r}.",
            hint="Use one of: " + ", ".join(sorted(k for k in _UNITS if k)),
        ) from None


def known_units(dimension: str | None = None) -> list[str]:
    """Sorted list of known unit symbols, optionally restricted to one dimension."""
    return sorted(
        k for k, u in _UNITS.items() if k and (dimension is None or u.dimension == dimension)
    )


def _check_dimension(unit: Unit, expect: str | None) -> None:
    if expect is not None and unit.dimension != expect:
        raise UnitError(
            f"Unit {unit.symbol!r} has dimension '{unit.dimension}', expected '{expect}'.",
            hint=f"Use one of: {', '.join(known_units(expect))}",
        )


def to_si(value: ArrayLike, unit: str, *, expect: str | None = None) -> np.ndarray | float:
    """Convert ``value`` given in ``unit`` to SI, checking the dimension if ``expect`` is set."""
    u = get_unit(unit)
    _check_dimension(u, expect)
    return np.asarray(u.to_si(np.asarray(value, dtype=float)))[()]


def from_si(value: ArrayLike, unit: str, *, expect: str | None = None) -> np.ndarray | float:
    """Convert an SI ``value`` to ``unit``."""
    u = get_unit(unit)
    _check_dimension(u, expect)
    return np.asarray(u.from_si(np.asarray(value, dtype=float)))[()]


_QUANTITY_RE = re.compile(
    r"^\s*(?P<num>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>\S.*?)?\s*$"
)


def parse_quantity(text: str) -> tuple[float, str]:
    """Split user text like ``"1550 nm"`` or ``"-3dBm"`` into ``(1550.0, "nm")``.

    Only a single decimal number followed by a unit symbol from the table is
    accepted; anything else raises :class:`UnitError`. No evaluation occurs.
    """
    m = _QUANTITY_RE.match(text)
    if m is None:
        raise UnitError(
            f"Cannot parse quantity {text!r}.",
            hint="Write a number followed by a unit, e.g. '1550 nm' or '-3 dBm'.",
        )
    unit = (m.group("unit") or "1").strip()
    get_unit(unit)
    return float(m.group("num")), unit


def parse_to_si(text: str, *, expect: str | None = None) -> float:
    """Parse ``"<number> <unit>"`` and return the SI value (dimension-checked)."""
    value, unit = parse_quantity(text)
    return float(to_si(value, unit, expect=expect))


__all__ = [
    "Unit",
    "beta2_to_dispersion",
    "db_per_km_to_per_m",
    "db_to_linear",
    "dbm_to_watt",
    "dispersion_slope_to_beta3",
    "dispersion_to_beta2",
    "frequency_to_wavelength",
    "from_si",
    "get_unit",
    "known_units",
    "linear_to_db",
    "parse_quantity",
    "parse_to_si",
    "per_m_to_db_per_km",
    "to_si",
    "watt_to_dbm",
    "wavelength_to_frequency",
]
