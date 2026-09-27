from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.core import units as u
from optobuild.core.constants import SPEED_OF_LIGHT
from optobuild.core.errors import UnitError


def test_dbm_watt_round_trip() -> None:
    assert u.dbm_to_watt(0.0) == pytest.approx(1e-3)
    assert u.dbm_to_watt(10.0) == pytest.approx(1e-2)
    assert u.dbm_to_watt(-30.0) == pytest.approx(1e-6)
    assert u.watt_to_dbm(1e-3) == pytest.approx(0.0, abs=1e-12)
    x = np.array([-20.0, 0.0, 3.0])
    np.testing.assert_allclose(u.watt_to_dbm(u.dbm_to_watt(x)), x)


def test_db_linear() -> None:
    assert u.db_to_linear(3.0) == pytest.approx(1.9952623149688795)
    assert u.linear_to_db(100.0) == pytest.approx(20.0)
    with pytest.raises(UnitError):
        u.linear_to_db(0.0)
    with pytest.raises(UnitError, match="dBm"):
        u.watt_to_dbm(-1.0)


def test_attenuation_definition() -> None:
    """0.2 dB/km over 50 km must be exactly 10 dB of power loss with exp(-alpha L)."""
    alpha = u.db_per_km_to_per_m(0.2)
    assert math.exp(-alpha * 50e3) == pytest.approx(0.1, rel=1e-12)
    assert u.per_m_to_db_per_km(alpha) == pytest.approx(0.2)


def test_wavelength_frequency() -> None:
    assert u.wavelength_to_frequency(1550e-9) == pytest.approx(SPEED_OF_LIGHT / 1550e-9)
    assert u.frequency_to_wavelength(u.wavelength_to_frequency(1310e-9)) == pytest.approx(1310e-9)
    with pytest.raises(UnitError):
        u.wavelength_to_frequency(0.0)
    with pytest.raises(UnitError):
        u.frequency_to_wavelength(-1.0)


def test_dispersion_conversions() -> None:
    """17 ps/(nm km) at 1550 nm is beta2 ~ -21.68 ps^2/km (textbook SMF value)."""
    d = u.to_si(17.0, "ps/(nm km)")
    beta2 = u.dispersion_to_beta2(d, 1550e-9)
    assert u.from_si(beta2, "ps^2/km") == pytest.approx(-21.6826, rel=1e-4)
    assert u.beta2_to_dispersion(beta2, 1550e-9) == pytest.approx(d)


def test_beta3_from_slope_matches_numerical_derivative() -> None:
    """beta3 = d(beta2)/d(omega); check the closed form against a finite difference."""
    lam0, d0, s = 1550e-9, 17e-6, u.to_si(0.058, "ps/(nm^2 km)")

    def beta2(lam: float) -> float:
        return u.dispersion_to_beta2(d0 + s * (lam - lam0), lam)

    def omega(lam: float) -> float:
        return 2 * math.pi * SPEED_OF_LIGHT / lam

    h = 1e-12
    numeric = (beta2(lam0 + h) - beta2(lam0 - h)) / (omega(lam0 + h) - omega(lam0 - h))
    assert u.dispersion_slope_to_beta3(d0, s, lam0) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1550, "nm", 1550e-9),
        (1.55, "µm", 1.55e-6),
        (1.55, "um", 1.55e-6),
        (193.4, "THz", 193.4e12),
        (10, "GHz", 1e10),
        (5, "ps", 5e-12),
        (100, "fs", 1e-13),
        (80, "km", 8e4),
        (2, "mW", 2e-3),
        (3, "µW", 3e-6),
        (0, "dBm", 1e-3),
        (17, "ps/nm/km", 17e-6),
        (10, "Gb/s", 1e10),
        (50, "ohm", 50.0),
        (-20, "ps^2/km", -20e-27),
    ],
)
def test_table_to_si(value: float, unit: str, expected: float) -> None:
    assert u.to_si(value, unit) == pytest.approx(expected)
    assert u.from_si(expected, unit) == pytest.approx(value, abs=1e-12)


def test_dimension_check() -> None:
    with pytest.raises(UnitError, match="dimension"):
        u.to_si(1.0, "nm", expect="power")
    assert "dBm" in u.known_units("power")


def test_unknown_unit() -> None:
    with pytest.raises(UnitError, match="Unknown unit"):
        u.to_si(1.0, "furlong")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1550 nm", (1550.0, "nm")),
        ("-3dBm", (-3.0, "dBm")),
        ("1e-3 W", (1e-3, "W")),
        (" 0.2 dB/km ", (0.2, "dB/km")),
        ("42", (42.0, "1")),
        (".5 V", (0.5, "V")),
    ],
)
def test_parse_quantity(text: str, expected: tuple[float, str]) -> None:
    assert u.parse_quantity(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "nm", "1550 nm nm", "__import__('os')", "2*3 W", "1550 parsecs", "1,5 nm"],
)
def test_parse_rejects_non_quantities(text: str) -> None:
    with pytest.raises(UnitError):
        u.parse_quantity(text)


def test_parse_to_si() -> None:
    assert u.parse_to_si("1550 nm", expect="length") == pytest.approx(1550e-9)
    with pytest.raises(UnitError):
        u.parse_to_si("1550 nm", expect="frequency")


def test_db_loss_presentation_unit() -> None:
    assert u.to_si(5.0, "dB loss") == pytest.approx(10 ** (-0.5))
    assert u.from_si(10 ** (-0.5), "dB loss") == pytest.approx(5.0)
    assert u.from_si(1.0, "dB loss") == pytest.approx(0.0, abs=1e-15)
    assert u.get_unit("dB loss").dimension == "ratio"
