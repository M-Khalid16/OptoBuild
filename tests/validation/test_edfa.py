"""Validation of the two-level EDFA, its ASE/noise figure and the fiber ring laser
(docs/physics_models.md 3.26).

References: the Saleh-Jopson implicit solution (independent of the z
integration), the full-inversion quantum limit NF = 2 - 1/G, and the closed-form
ring-laser output, threshold and slope. Tolerances: RK4 with 200 steps has a
relative error < 1e-7 for these gains (measured 4e-8 against 400 steps for the
strongly absorbing 1535 nm beam); the implicit solution is solved to 1e-12.
"""

from __future__ import annotations

import math

import pytest

from optobuild.physics.edfa import (
    Beam,
    ErbiumFiber,
    ring_laser_closed_form,
    ring_laser_slope_efficiency,
    ring_laser_threshold,
    spontaneous_emission_factor,
)
from optobuild.solvers.edfa import amplify, ring_laser

FIBER = ErbiumFiber()
PUMP = Beam(980e-9, 2.2e-25, 0.0, 0.6)
SIGNAL = Beam(1550e-9, 2.6e-25, 3.4e-25, 0.4)
SIGNAL2 = Beam(1535e-9, 5.0e-25, 5.2e-25, 0.4)


@pytest.mark.parametrize(
    "powers",
    [(0.05, 1e-6), (0.1, 1e-4), (0.1, 1e-3), (0.02, 5e-3)],
)
def test_z_integration_matches_saleh_jopson(powers: tuple[float, float]) -> None:
    sol = amplify(FIBER, (PUMP, SIGNAL), powers)
    exact = FIBER.saleh_jopson((PUMP, SIGNAL), powers)
    assert sol.powers_out == pytest.approx(exact, rel=1e-7)


def test_two_signals_and_step_convergence() -> None:
    beams = (PUMP, SIGNAL, SIGNAL2)
    powers = (0.08, 1e-4, 3e-4)
    sol = amplify(FIBER, beams, powers)
    assert sol.powers_out == pytest.approx(FIBER.saleh_jopson(beams, powers), rel=1e-7)
    finer = amplify(FIBER, beams, powers, n_steps=400)
    assert sol.powers_out == pytest.approx(finer.powers_out, rel=1e-7)
    assert sol.ase_density == pytest.approx(finer.ase_density, rel=1e-7)


def test_full_inversion_gain_and_quantum_limited_noise_figure() -> None:
    """At 10 W pump the fiber is almost fully inverted. The gain coefficient is linear in
    n2, so ln G = Gamma n_t L [(sigma_a + sigma_e) mean(n2) - sigma_a] exactly; the deviation
    from full inversion is ln(G_full / G) = Gamma n_t L (sigma_a + sigma_e)(1 - mean n2), and
    n_sp - 1 = sigma_a (1 - n2) / (sigma_e n2) bounds the noise-figure excess."""
    sol = amplify(FIBER, (PUMP, SIGNAL), (10.0, 1e-7))
    n2 = sol.mean_upper_fraction
    sig_sum = SIGNAL.absorption_cross_section + SIGNAL.emission_cross_section
    gl = SIGNAL.overlap * FIBER.erbium_density * FIBER.length
    assert math.log(sol.gains[1]) == pytest.approx(
        gl * (sig_sum * n2 - SIGNAL.absorption_cross_section),
        rel=1e-7,  # RK4 error bound
    )
    assert 1 - n2 < 2e-4
    g_full = math.exp(gl * SIGNAL.emission_cross_section)
    # absolute tolerance: the RK4 bound 1e-7 on ln G carried into this small difference
    assert math.log(g_full / sol.gains[1]) == pytest.approx(
        gl * sig_sum * (1 - n2), abs=1e-7 * math.log(g_full)
    )
    excess = SIGNAL.absorption_cross_section * (1 - n2) / (SIGNAL.emission_cross_section * n2)
    assert sol.noise_figures[1] == pytest.approx(2 - 1 / sol.gains[1], rel=2 * excess + 1e-9)
    assert spontaneous_emission_factor(SIGNAL, 1.0) == 1.0


def test_partial_inversion_noise_figure_exceeds_quantum_limit() -> None:
    """Weak pumping leaves the fiber end poorly inverted: NF > 2 n_sp(mean n2) - ... > 3 dB."""
    sol = amplify(FIBER, (PUMP, SIGNAL), (0.02, 1e-3))
    assert sol.noise_figures[1] > 2.0
    n_sp = spontaneous_emission_factor(SIGNAL, sol.mean_upper_fraction)
    assert n_sp > 1


def test_ring_laser_numeric_equals_closed_form() -> None:
    t, eta = 0.5, 0.8
    for pump in (0.005, 0.02, 0.1):
        numeric, _ = ring_laser(FIBER, PUMP, SIGNAL, pump, t, eta)
        assert numeric == pytest.approx(
            ring_laser_closed_form(FIBER, PUMP, SIGNAL, pump, t, eta), rel=1e-7
        )
    p_th = ring_laser_threshold(FIBER, PUMP, SIGNAL, t, eta)
    assert ring_laser(FIBER, PUMP, SIGNAL, 0.99 * p_th, t, eta)[0] == 0.0
    assert ring_laser(FIBER, PUMP, SIGNAL, 1.01 * p_th, t, eta)[0] > 0.0
    slope = ring_laser_slope_efficiency(FIBER, PUMP, SIGNAL, t, eta)
    a, b = (ring_laser_closed_form(FIBER, PUMP, SIGNAL, p, t, eta) for p in (0.05, 0.06))
    assert (b - a) / 0.01 == pytest.approx(slope, rel=1e-12)  # linear above threshold
    assert slope < PUMP.wavelength / SIGNAL.wavelength  # below the quantum-defect limit
