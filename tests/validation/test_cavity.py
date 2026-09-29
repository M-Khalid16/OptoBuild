"""Validation of the round-trip cavity iteration (docs/physics_models.md 3.28).

Exact reference: a loop saturated gain -> Gaussian gate -> Gaussian filter ->
output coupler maps chirp-free Gaussians onto Gaussians; its fixed point
a* = (-c + sqrt(c^2 + c W^2))/2 and steady energy E* = E_sat (g0/L - 1) are
closed forms. Started from seeded noise, the iteration must reach them. The
map contracts towards the Gaussian ground state with ratio ~a*/(a* + c) per
round trip; after convergence to 1e-12 per round trip the remaining shape
error is < 1e-9 of the peak (tolerance), limited by round-off and the
periodic window (Gaussian tails < 1e-30 at the edges).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.numerics.grid import TimeGrid
from optobuild.physics.cavity import (
    gate_filter_fixed_point,
    gate_filter_steady_energy,
    gaussian_filter,
    gaussian_gate,
    output_coupler,
    saturable_absorber,
    saturated_gain,
)
from optobuild.solvers.cavity import run_cavity

GRID = TimeGrid(2048, 10e-15, -1024 * 10e-15)
TM, W = 2e-12, 2 * math.pi * 2e12
T_REL = (np.arange(2048) - 1024) * 10e-15


def _loop(g0: float, e_sat: float, t_out: float, eta: float = 1.0):  # type: ignore[no-untyped-def]
    return [
        lambda a: saturated_gain(a, GRID, g0, e_sat),
        lambda a: gaussian_gate(a, GRID, TM),
        lambda a: eta**0.5 * gaussian_filter(a, GRID, W),
        lambda a: output_coupler(a, t_out),
    ]


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1e-4 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))


@pytest.mark.parametrize("g0,e_sat,t_out,eta", [(3.0, 1e-12, 0.3, 1.0), (2.0, 5e-12, 0.1, 0.8)])
def test_gate_filter_cavity_reaches_the_exact_fixed_point(g0, e_sat, t_out, eta) -> None:  # type: ignore[no-untyped-def]
    res = run_cavity(_loop(g0, e_sat, t_out, eta), _noise(1), GRID, max_round_trips=5000,
                     tolerance=1e-12)  # fmt: skip
    assert res.converged
    a_star = gate_filter_fixed_point(TM, W)
    intensity = np.abs(res.field) ** 2
    expected = intensity.max() * np.exp(-2 * a_star * T_REL**2)
    assert np.max(np.abs(intensity - expected)) < 1e-9 * intensity.max()
    e_star = gate_filter_steady_energy(TM, W, g0, e_sat, t_out, eta)
    assert res.energies[-1] == pytest.approx(e_star, rel=1e-9)
    # output energy = T / (1 - T) x the energy kept in the cavity
    out_energy = float(np.sum(np.abs(res.output) ** 2) * GRID.dt)
    assert out_energy == pytest.approx(res.energies[-1] * t_out / (1 - t_out), rel=1e-12)


def test_below_threshold_the_field_decays() -> None:
    a = gate_filter_fixed_point(TM, W)
    loss = -math.log(0.7 * a / (a + 1 / TM**2))
    assert gate_filter_steady_energy(TM, W, 0.9 * loss, 1e-12, 0.3) == 0.0
    res = run_cavity(_loop(0.9 * loss, 1e-12, 0.3), _noise(2), GRID, max_round_trips=300)
    assert res.energies[-1] < 1e-6 * res.energies[0]


def test_non_convergence_is_reported() -> None:
    res = run_cavity(_loop(3.0, 1e-12, 0.3), _noise(3), GRID, max_round_trips=5, tolerance=1e-12)
    assert not res.converged and res.round_trips == 5 and res.energies.size == 5


def test_saturable_absorber_limits() -> None:
    a = np.array([1e-6, 1e3], dtype=complex)  # powers 1e-12 W and 1e6 W
    out = saturable_absorber(a, 0.4, 1.0, 0.05)
    assert abs(out[0]) ** 2 == pytest.approx(1e-12 * (1 - 0.4 - 0.05), rel=1e-9)
    assert abs(out[1]) ** 2 == pytest.approx(1e6 * (1 - 0.05), rel=1e-5)
