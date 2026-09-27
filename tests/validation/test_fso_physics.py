"""Validation of the free-space optical channel physics against independent references.

Statistical tests use fixed seeds and 5-sigma tolerances derived from the
estimator's variance (binomial for probabilities, sample variance for means).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.integrate

from optobuild.core.units import per_m_to_db_per_km
from optobuild.physics.atmospheric import (
    LN_2_PERCENT,
    rain_attenuation,
    visibility_attenuation,
    visibility_exponent,
)
from optobuild.physics.free_space import (
    aperture_fraction,
    beam_radius,
    diffraction_limited_divergence,
    displacement_quadrature,
    mean_aperture_fraction,
)
from optobuild.physics.fso_channel import FSOChannelModel
from optobuild.physics.turbulence import (
    aperture_averaging_factor,
    gamma_gamma_cdf,
    gamma_gamma_parameters,
    gamma_gamma_pdf,
    gamma_gamma_scintillation_index,
    lognormal_cdf,
    rytov_variance,
    sample_gamma_gamma,
    sample_lognormal,
)

LAM = 1550e-9


# --- beam and aperture -------------------------------------------------------------------
def test_diffraction_limited_beam_equals_gaussian_beam_formula() -> None:
    w0, z = 5e-3, 2e3
    zr = math.pi * w0**2 / LAM
    theta = diffraction_limited_divergence(LAM, w0)
    assert beam_radius(z, w0, theta) == pytest.approx(w0 * math.sqrt(1 + (z / zr) ** 2), rel=1e-12)


@pytest.mark.parametrize(("a", "w"), [(0.05, 1.0), (0.3, 1.0), (1.0, 0.5)])
def test_centered_fraction_closed_form(a: float, w: float) -> None:
    assert aperture_fraction(a, w) == pytest.approx(1 - math.exp(-2 * a**2 / w**2), rel=1e-12)


@pytest.mark.parametrize(("a", "w", "r"), [(0.05, 1.0, 0.4), (0.3, 1.0, 0.25), (0.5, 0.4, 0.7)])
def test_offset_fraction_matches_2d_quadrature(a: float, w: float, r: float) -> None:
    """Independent reference: integrate the Gaussian intensity over the aperture."""

    def intensity(rho: float, phi: float) -> float:
        d2 = rho**2 + r**2 - 2 * rho * r * math.cos(phi)
        return 2 / (math.pi * w**2) * math.exp(-2 * d2 / w**2) * rho

    ref, _ = scipy.integrate.dblquad(intensity, 0, 2 * math.pi, 0, a, epsabs=1e-13, epsrel=1e-12)
    assert aperture_fraction(a, w, r) == pytest.approx(ref, rel=1e-8)


def test_small_aperture_and_large_aperture_limits() -> None:
    w, a, r = 1.0, 1e-3, 0.3
    on_axis = 2 * a**2 / w**2 * math.exp(-2 * r**2 / w**2)  # I(r) * pi a^2
    assert aperture_fraction(a, w, r) == pytest.approx(on_axis, rel=1e-5)
    assert aperture_fraction(10.0, w, r) == pytest.approx(1.0, abs=1e-12)


def test_rayleigh_mean_matches_farid_hranilovic_for_small_apertures() -> None:
    """E[h_p] = A0 xi^2/(xi^2+1) (Farid & Hranilovic 2007), accurate for a << w."""
    a, w, s = 0.02, 1.0, 0.3
    v = math.sqrt(math.pi) * a / (math.sqrt(2) * w)
    a0 = math.erf(v) ** 2
    w_eq2 = w**2 * math.sqrt(math.pi) * math.erf(v) / (2 * v * math.exp(-(v**2)))
    xi2 = w_eq2 / (4 * s**2)
    fh = a0 * xi2 / (xi2 + 1)
    assert mean_aperture_fraction(a, w, s) == pytest.approx(fh, rel=2e-3)


def test_displacement_quadrature_is_normalized() -> None:
    for static, sigma in ((0.0, 0.1), (0.5, 0.1), (0.05, 0.2)):
        _, wts = displacement_quadrature(static, sigma, 400)
        assert wts.sum() == pytest.approx(1.0, abs=1e-10)


# --- atmospheric attenuation ------------------------------------------------------------
@pytest.mark.parametrize("v", [200.0, 800.0, 3e3, 20e3, 80e3])
def test_koschmieder_definition(v: float) -> None:
    """At 550 nm the transmission over the visibility distance is exactly 2 %."""
    assert math.exp(-visibility_attenuation(v, 550e-9) * v) == pytest.approx(0.02, rel=1e-12)
    assert LN_2_PERCENT == pytest.approx(3.912, abs=5e-4)


def test_kim_and_kruse_exponents() -> None:
    assert [visibility_exponent(v * 1e3) for v in (60, 20, 3, 0.8, 0.3)] == pytest.approx(
        [1.6, 1.3, 0.16 * 3 + 0.34, 0.3, 0.0]
    )
    assert visibility_exponent(3e3, "kruse") == pytest.approx(0.585 * 3 ** (1 / 3))
    assert visibility_exponent(20e3, "kruse") == visibility_exponent(20e3, "kim")
    with pytest.raises(ValueError):
        visibility_exponent(1e3, "other")


def test_kim_reference_value_at_1550nm() -> None:
    """V = 1 km: q = 0.5, beta = 3.912/km * (1550/550)^-0.5 = 2.3303/km -> 10.12 dB/km."""
    alpha = visibility_attenuation(1e3, LAM)
    assert per_m_to_db_per_km(alpha) == pytest.approx(10.12, abs=0.01)
    # dense fog (q = 0): wavelength independent
    assert visibility_attenuation(100.0, LAM) == pytest.approx(
        visibility_attenuation(100.0, 850e-9)
    )


def test_rain_attenuation_formula() -> None:
    r = 25 / 3.6e6  # 25 mm/h in m/s
    assert per_m_to_db_per_km(rain_attenuation(r)) == pytest.approx(1.076 * 25**0.67)
    assert rain_attenuation(0.0) == 0.0


# --- turbulence ---------------------------------------------------------------------------
def test_rytov_scaling_laws() -> None:
    base = rytov_variance(1e-14, LAM, 1e3)
    assert rytov_variance(2e-14, LAM, 1e3) == pytest.approx(2 * base)
    assert rytov_variance(1e-14, LAM, 2e3) == pytest.approx(2 ** (11 / 6) * base)
    assert rytov_variance(1e-14, LAM / 2, 1e3) == pytest.approx(2 ** (7 / 6) * base)
    k = 2 * math.pi / LAM
    assert base == pytest.approx(1.23e-14 * k ** (7 / 6) * 1e3 ** (11 / 6))


def test_aperture_averaging_limits() -> None:
    assert aperture_averaging_factor(LAM, 1e3, 1e-9) == pytest.approx(1.0, abs=1e-9)
    vals = [aperture_averaging_factor(LAM, 1e3, d) for d in (0.01, 0.05, 0.2)]
    assert vals[0] > vals[1] > vals[2] > 0


@pytest.mark.parametrize("sr2", [0.2, 1.0, 5.0])
def test_gamma_gamma_pdf_moments(sr2: float) -> None:
    """Independent quadrature of the pdf: normalization, E[h] = 1, E[h^2] = 1 + sigma_I^2."""
    a, b = gamma_gamma_parameters(sr2)

    def moment(k: int) -> float:
        return scipy.integrate.quad(
            lambda h: h**k * gamma_gamma_pdf(h, a, b), 0, np.inf, limit=500
        )[0]

    assert moment(0) == pytest.approx(1.0, abs=1e-8)
    assert moment(1) == pytest.approx(1.0, abs=1e-8)
    assert moment(2) == pytest.approx(1 + gamma_gamma_scintillation_index(a, b), rel=1e-7)
    for h in (0.3, 1.0, 2.0):
        ref = scipy.integrate.quad(lambda x: gamma_gamma_pdf(x, a, b), 0, h, limit=500)[0]
        assert gamma_gamma_cdf(h, a, b) == pytest.approx(ref, abs=1e-7)


def test_gamma_gamma_weak_turbulence_limit() -> None:
    """For sigma_R^2 -> 0: 1/alpha + 1/beta -> (0.49 + 0.51) sigma_R^2 = sigma_R^2."""
    sr2 = 1e-3
    a, b = gamma_gamma_parameters(sr2)
    assert gamma_gamma_scintillation_index(a, b) == pytest.approx(sr2, rel=2e-3)


def test_samplers_match_distributions() -> None:
    rng = np.random.default_rng(11)
    n = 400_000
    for sampler, cdf, si2 in (
        (lambda: sample_lognormal(rng, 0.3, n), lambda h: lognormal_cdf(h, 0.3), 0.3),
        (
            lambda: sample_gamma_gamma(rng, *gamma_gamma_parameters(1.5), n),
            lambda h: gamma_gamma_cdf(h, *gamma_gamma_parameters(1.5)),
            gamma_gamma_scintillation_index(*gamma_gamma_parameters(1.5)),
        ),
    ):
        s = sampler()
        assert s.mean() == pytest.approx(1.0, abs=5 * math.sqrt(si2 / n))
        assert s.var() == pytest.approx(si2, rel=0.05)  # 4th-moment-limited; ~5 sigma
        for h in (0.2, 0.5, 1.0, 2.0):
            p = cdf(h)
            assert (s <= h).mean() == pytest.approx(p, abs=5 * math.sqrt(p * (1 - p) / n))


# --- complete channel ---------------------------------------------------------------------
MODELS = {
    "turbulence_only": FSOChannelModel(
        1500, LAM, 0.01, 0.5e-3, 0.1, turbulence="gamma_gamma", cn2=5e-14
    ),
    "lognormal": FSOChannelModel(1000, LAM, 0.01, 0.5e-3, 0.1, turbulence="lognormal", cn2=1e-15),
    "pointing_only": FSOChannelModel(
        1500, LAM, 0.01, 0.5e-3, 0.1, pointing_jitter=300e-6, pointing_offset=100e-6
    ),
    "combined": FSOChannelModel(
        2000,
        LAM,
        0.01,
        1e-3,
        0.08,
        visibility=5e3,
        rain_rate=5e-6,
        pointing_jitter=100e-6,
        turbulence="gamma_gamma",
        cn2=2e-14,
        beam_wander=True,
        tx_efficiency=0.8,
        rx_efficiency=0.7,
    ),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_channel_mean_and_outage_match_monte_carlo(name: str) -> None:
    m = MODELS[name]
    n = 300_000
    s = m.sample(np.random.default_rng(5), n)["gain"]
    mean = m.mean_gain()
    assert s.mean() == pytest.approx(mean, abs=5 * s.std() / math.sqrt(n))
    for frac in (0.3, 0.6, 0.9):
        p = m.outage_probability(frac * mean)
        tol = 5 * math.sqrt(max(p * (1 - p), 1 / n) / n)
        assert (s < frac * mean).mean() == pytest.approx(p, abs=tol), (name, frac)


def test_channel_budget_factors() -> None:
    m = MODELS["combined"]
    assert m.constant_gain == pytest.approx(
        0.8 * 0.7 * math.exp(-(visibility_attenuation(5e3, LAM) + rain_attenuation(5e-6)) * 2000)
    )
    assert m.mean_pointing_fraction() < m.geometric_fraction()
    with pytest.raises(ValueError):
        FSOChannelModel(1000, LAM, 0.01, turbulence="lognormal", cn2=0.0)
