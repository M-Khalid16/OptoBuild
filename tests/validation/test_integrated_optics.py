"""Validation of integrated-optics models, the S-matrix circuit solver, the thin-film TMM and
the resonance analysis (docs/physics_models.md 3.22-3.24).

Independent references: closed forms (rings, MZI, quarter-wave stacks,
Fresnel, CMT peak reflectance/bandwidth), the circuit solver against the
closed forms (different algebra: matrix elimination vs series summation),
and coupled-mode theory against the transfer-matrix method (different
physics approximations). Tolerances:
* identities computed two ways: 1e-12 (round-off of O(10) operations);
* CMT vs TMM: the synchronous (coupled-mode) approximation neglects terms of
  second order in the index modulation, (dn/n)^2 = 7e-7 here; we allow 10x;
* sampled-spectrum measurements (FSR, FWHM, delay): the stated
  interpolation/difference error for the grid used (see each test).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optobuild.analysis.resonances import find_resonances, free_spectral_range, group_delay
from optobuild.core.constants import SPEED_OF_LIGHT as C
from optobuild.core.errors import InvalidParameterError, NumericalStabilityError
from optobuild.physics.integrated_optics import (
    add_drop_ring,
    all_pass_fwhm_phase,
    all_pass_loaded_q,
    all_pass_ring,
    bragg_bandwidth,
    bragg_coupling_square,
    bragg_grating,
    bragg_peak_reflectance,
    coupler_matrix,
    mzi,
    ring_fsr_wavelength,
    waveguide_transmission,
)
from optobuild.physics.multilayer import quarter_wave_reflectance, stack_response
from optobuild.solvers.circuit import Circuit, four_port_coupler, two_port

LAM0 = 1.55e-6
NEFF, NG = 2.4, 4.2
RING_L = 2 * math.pi * 10e-6
ALPHA = math.log(10) / 10 * 300.0  # 3 dB/cm in 1/m


def _nu(lam_lo: float, lam_hi: float, n: int) -> np.ndarray:
    return np.linspace(C / lam_hi, C / lam_lo, n)  # ascending frequency


def _wg(nu: np.ndarray, length: float, loss: float = ALPHA) -> np.ndarray:
    return waveguide_transmission(nu, length, NEFF, NG, LAM0, loss)


# -- elements ------------------------------------------------------------------------------


def test_waveguide_group_delay_is_ng_l_over_c() -> None:
    """Linear phase in frequency: central differences are exact up to round-off."""
    nu = _nu(1.54e-6, 1.56e-6, 2001)
    tau = group_delay(nu, _wg(nu, 1e-3))
    assert tau == pytest.approx(np.full(nu.size, NG * 1e-3 / C), rel=1e-9)


def test_coupler_is_unitary_and_splits_power() -> None:
    c = coupler_matrix(0.3)
    assert np.allclose(c @ c.conj().T, np.eye(2), atol=1e-15)
    assert abs(c[1, 0]) ** 2 == pytest.approx(0.3) and abs(c[0, 0]) ** 2 == pytest.approx(0.7)
    with pytest.raises(ValueError):
        coupler_matrix(1.5)


# -- rings -----------------------------------------------------------------------------------


def _all_pass_circuit(nu: np.ndarray, k2: float, loss: float = ALPHA) -> np.ndarray:
    cir = Circuit()
    cir.add(four_port_coupler("dc", coupler_matrix(k2), nu.size))
    cir.add(two_port("ring", _wg(nu, RING_L, loss)))
    cir.connect("dc.out2", "ring.a")
    cir.connect("ring.b", "dc.in2")
    cir.expose("in", "dc.in1")
    cir.expose("out", "dc.out1")
    s, names = cir.smatrix()
    assert names == ("in", "out")
    return s


def test_all_pass_ring_circuit_equals_closed_form_and_is_reciprocal() -> None:
    nu = _nu(1.54e-6, 1.56e-6, 4001)
    s = _all_pass_circuit(nu, 0.1)
    np.testing.assert_allclose(s[:, 1, 0], all_pass_ring(_wg(nu, RING_L), 0.1), atol=1e-12)
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), atol=1e-12)


def test_all_pass_extrema_fsr_fwhm_and_q() -> None:
    """Grid: 4e5 points over 40 nm (12.5 MHz); FWHM ~ 10 GHz, so > 500 samples per FWHM
    and linear-interpolation error of the half-depth crossings < 1e-5 relative. Resonance
    positions are grid samples: FSR error <= one grid step."""
    k2 = 0.05
    nu = _nu(1.53e-6, 1.57e-6, 400_001)
    step = nu[1] - nu[0]
    h = all_pass_ring(_wg(nu, RING_L), k2)
    t = math.sqrt(1 - k2)
    a = math.exp(-0.5 * ALPHA * RING_L)
    power = np.abs(h) ** 2
    res = find_resonances(nu, power, "dip")
    assert len(res) >= 3
    assert min(power) == pytest.approx((t - a) ** 2 / (1 - t * a) ** 2, rel=1e-6)
    assert max(power) == pytest.approx((t + a) ** 2 / (1 + t * a) ** 2, rel=1e-9)
    assert free_spectral_range(res) == pytest.approx(C / (NG * RING_L), abs=step)
    fwhm_nu = all_pass_fwhm_phase(k2, a) * C / (2 * math.pi * NG * RING_L)
    for r in res:
        assert r.samples_per_fwhm > 500
        assert r.fwhm == pytest.approx(fwhm_nu, rel=1e-5)
        assert r.q_factor == pytest.approx(r.position / fwhm_nu, rel=1e-5)
    # the Lorentzian loaded-Q formula is the high-Q limit: relative error O(1 - ta) = 5 %
    q_lorentz = all_pass_loaded_q(C / res[1].position, NG, RING_L, k2, ALPHA)
    assert res[1].q_factor == pytest.approx(q_lorentz, rel=2 * (1 - t * a))
    # FSR in wavelength (closed form lambda^2 / (n_g L) at the resonance)
    lam = C / res[1].position
    # difference of two resonances vs the local form lambda^2/(n_g L): O(FSR/lambda) = 0.6 %
    fsr_lam = C / res[0].position - lam
    assert fsr_lam == pytest.approx(ring_fsr_wavelength(lam, NG, RING_L), rel=0.01)


def test_critical_coupling_gives_full_extinction() -> None:
    a = math.exp(-0.5 * ALPHA * RING_L)
    k2 = 1 - a * a  # t = a
    phi = np.array([0.0])  # exactly on resonance
    assert abs(all_pass_ring(a * np.exp(-1j * phi), k2)[0]) < 1e-15


def test_lossless_all_pass_group_delay_at_resonance() -> None:
    """tau = tau_rt (1 + t)/(1 - t) at resonance (lossless). Central differences on a grid
    with 2000 samples per FWHM: relative error ~ (dw/FWHM)^2 < 1e-5."""
    k2 = 0.1
    t = math.sqrt(1 - k2)
    tau_rt = NG * RING_L / C
    fsr = C / (NG * RING_L)
    fwhm = all_pass_fwhm_phase(k2, 1.0) / (2 * math.pi) * fsr
    nu0 = C / LAM0 + 3.7 * fsr  # any grid; locate a resonance below
    nu = np.linspace(nu0 - fsr / 2, nu0 + fsr / 2, int(2000 * fsr / fwhm))
    h = all_pass_ring(_wg(nu, RING_L, 0.0), k2)
    assert np.abs(h) == pytest.approx(1.0, abs=1e-12)  # all-pass
    tau = group_delay(nu, h)
    assert tau.max() == pytest.approx(tau_rt * (1 + t) / (1 - t), rel=1e-5)


def _add_drop_circuit(nu: np.ndarray, k1: float, k2: float, loss: float) -> np.ndarray:
    half = _wg(nu, RING_L / 2, loss)
    cir = Circuit()
    cir.add(four_port_coupler("c1", coupler_matrix(k1), nu.size))
    cir.add(four_port_coupler("c2", coupler_matrix(k2), nu.size))
    cir.add(two_port("top", half))
    cir.add(two_port("bottom", half))
    for a, b in (("c1.out2", "top.a"), ("top.b", "c2.in2"), ("c2.out2", "bottom.a")):
        cir.connect(a, b)
    cir.connect("bottom.b", "c1.in2")
    for name, ref in (("in", "c1.in1"), ("thru", "c1.out1"), ("add", "c2.in1")):
        cir.expose(name, ref)
    cir.expose("drop", "c2.out1")
    return cir.smatrix()[0]


def test_add_drop_circuit_equals_closed_form_and_conserves_energy() -> None:
    nu = _nu(1.54e-6, 1.56e-6, 4001)
    s = _add_drop_circuit(nu, 0.1, 0.05, ALPHA)
    th, dr = add_drop_ring(_wg(nu, RING_L / 2), 0.1, 0.05)
    np.testing.assert_allclose(s[:, 1, 0], th, atol=1e-12)
    np.testing.assert_allclose(s[:, 3, 0], dr, atol=1e-12)
    lossless = _add_drop_circuit(nu, 0.1, 0.1, 0.0)
    # unitary S (lossless, reflection-free): S^H S = I at every frequency
    eye = np.eye(4)[None]
    np.testing.assert_allclose(np.conj(np.swapaxes(lossless, 1, 2)) @ lossless, eye + 0 * lossless,
                               atol=1e-10)  # fmt: skip
    # symmetric lossless add-drop drops everything exactly on resonance (h = +-1)
    for h in (1.0, -1.0):
        th, dr = add_drop_ring(np.array([h]), 0.1, 0.1)
        assert abs(dr[0]) ** 2 == pytest.approx(1.0, abs=1e-12) and abs(th[0]) < 1e-12


# -- MZI --------------------------------------------------------------------------------------


def test_mzi_circuit_closed_form_and_fsr() -> None:
    dl = 100e-6
    nu = _nu(1.54e-6, 1.56e-6, 20001)
    h1, h2 = _wg(nu, 1e-3, 0.0), _wg(nu, 1e-3 + dl, 0.0)
    m = mzi(h1, h2)
    cir = Circuit()
    cir.add(four_port_coupler("c1", coupler_matrix(0.5), nu.size))
    cir.add(four_port_coupler("c2", coupler_matrix(0.5), nu.size))
    cir.add(two_port("arm1", h1))
    cir.add(two_port("arm2", h2))
    for a, b in (("c1.out1", "arm1.a"), ("c1.out2", "arm2.a")):
        cir.connect(a, b)
    cir.connect("arm1.b", "c2.in1")
    cir.connect("arm2.b", "c2.in2")
    for name, ref in (("in1", "c1.in1"), ("in2", "c1.in2"), ("bar", "c2.out1")):
        cir.expose(name, ref)
    cir.expose("cross", "c2.out2")
    s = cir.smatrix()[0]
    np.testing.assert_allclose(s[:, 2, 0], m[0, 0], atol=1e-12)
    np.testing.assert_allclose(s[:, 3, 0], m[1, 0], atol=1e-12)
    dphi = np.angle(h1 / h2)
    np.testing.assert_allclose(np.abs(m[1, 0]) ** 2, np.cos(dphi / 2) ** 2, atol=1e-12)
    res = find_resonances(nu, np.abs(m[1, 0]) ** 2, "peak")
    # peak positions are grid samples (error <= half a step each)
    assert free_spectral_range(res) == pytest.approx(C / (NG * dl), abs=nu[1] - nu[0])


# -- Bragg gratings: CMT vs TMM ----------------------------------------------------------------

NAVG, DN = 2.4, 2e-3
PERIOD = LAM0 / (2 * NAVG)
CMT_TOL = 10 * (DN / NAVG) ** 2


def _stack(n_periods: int, shift: bool = False) -> tuple[list[float], list[float]]:
    nh, nl = NAVG + DN / 2, NAVG - DN / 2
    idx, d = [nh, nl] * n_periods, [PERIOD / 2] * (2 * n_periods)
    if shift:
        idx, d = idx + [NAVG] + idx, d + [PERIOD / 2] + d
    return idx, d


def test_uniform_grating_cmt_matches_tmm_and_closed_forms() -> None:
    n = 1500
    lam = np.linspace(1.547e-6, 1.553e-6, 6001)
    _, _, big_r, big_t = stack_response(lam, *_stack(n), NAVG, NAVG)
    kappa = bragg_coupling_square(DN, LAM0)
    r, t = bragg_grating(C / lam, n * PERIOD, PERIOD, kappa, NAVG, NAVG, LAM0)
    np.testing.assert_allclose(np.abs(r) ** 2, big_r, atol=CMT_TOL)
    np.testing.assert_allclose(np.abs(r) ** 2 + np.abs(t) ** 2, 1.0, atol=1e-12)
    assert np.max(np.abs(r) ** 2) == pytest.approx(bragg_peak_reflectance(kappa, n * PERIOD))
    # bandwidth between the first zeros around the peak (grid 1 pm): within 2 grid steps
    power = np.abs(r) ** 2
    zeros = find_resonances(lam, power, "dip", min_depth=1e-3)
    centre = int(np.argmax(power))
    left = max(z.position for z in zeros if z.index < centre)
    right = min(z.position for z in zeros if z.index > centre)
    step = lam[1] - lam[0]
    assert right - left == pytest.approx(bragg_bandwidth(LAM0, NAVG, kappa, n * PERIOD),
                                         abs=2 * step)  # fmt: skip
    assert big_r + big_t == pytest.approx(np.ones_like(big_r), abs=1e-12)


def test_phase_shifted_grating_circuit_matches_tmm() -> None:
    """Two CMT gratings joined by a half-period gap in the circuit solver vs TMM of the
    whole pi-shifted stack: validates the solver with reflections and cavity phase."""
    n = 1500
    lam = np.linspace(1.547e-6, 1.553e-6, 3001)
    nu = C / lam
    _, _, _, big_t = stack_response(lam, *_stack(n, shift=True), NAVG, NAVG)
    kappa = bragg_coupling_square(DN, LAM0)
    r, t = bragg_grating(nu, n * PERIOD, PERIOD, kappa, NAVG, NAVG, LAM0)
    cir = Circuit()
    cir.add(two_port("g1", t, r))
    cir.add(two_port("g2", t, r))
    cir.add(two_port("gap", waveguide_transmission(nu, PERIOD / 2, NAVG, NAVG, LAM0)))
    cir.connect("g1.b", "gap.a")
    cir.connect("gap.b", "g2.a")
    cir.expose("in", "g1.a")
    cir.expose("out", "g2.b")
    s = cir.smatrix()[0]
    np.testing.assert_allclose(np.abs(s[:, 1, 0]) ** 2, big_t, atol=CMT_TOL)
    assert np.abs(s[np.argmin(np.abs(lam - LAM0)), 1, 0]) ** 2 == pytest.approx(1.0, abs=1e-9)


# -- TMM ---------------------------------------------------------------------------------------


def test_tmm_quarter_wave_stack_fresnel_and_absorption() -> None:
    nh, nl = 2.3, 1.45
    idx, d = [nh, nl] * 8, [LAM0 / 4 / nh, LAM0 / 4 / nl] * 8
    _, _, big_r, big_t = stack_response([LAM0], idx, d, 1.0, 1.52)
    assert big_r[0] == pytest.approx(quarter_wave_reflectance(nh, nl, 8, 1.0, 1.52), rel=1e-12)
    assert big_r[0] + big_t[0] == pytest.approx(1.0, abs=1e-12)
    _, _, r0, _ = stack_response([LAM0], [], [], 1.0, 1.5)  # bare interface: Fresnel
    assert r0[0] == pytest.approx(((1 - 1.5) / (1 + 1.5)) ** 2, rel=1e-12)
    _, _, ra, ta = stack_response([LAM0], [1.5 - 0.01j], [1e-6], 1.0, 1.0)  # absorbing
    assert ra[0] + ta[0] < 1.0


# -- solver errors -----------------------------------------------------------------------------


def test_circuit_rejects_bad_netlists_and_singular_resonance() -> None:
    cir = Circuit()
    cir.add(two_port("w", np.array([1.0])))
    with pytest.raises(InvalidParameterError, match="Unknown port"):
        cir.connect("w.a", "x.b")
    cir.connect("w.a", "w.b")  # a lossless loop: singular at resonance
    with pytest.raises(InvalidParameterError, match="already"):
        cir.expose("p", "w.a")
    with pytest.raises(NumericalStabilityError):
        cir.smatrix()
    with pytest.raises(InvalidParameterError, match="Duplicate"):
        cir.add(two_port("w", np.array([1.0])))
