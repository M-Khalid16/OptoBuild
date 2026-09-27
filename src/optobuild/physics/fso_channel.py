"""Complete FSO channel gain model (physics_models.md 3.15).

Power gain of a horizontal free-space link (quasi-static: constant over one
simulation window):

    h = eta_tx * eta_rx * exp(-(beta_fog + beta_rain) L) * h_p(r) * h_t

* eta_tx, eta_rx: transmitter/receiver optics transmissions (linear)
* beta_*: attenuation coefficients [1/m] (physics.atmospheric)
* h_p(r): fraction of the Gaussian beam collected by the aperture for radial
  displacement r (physics.free_space); r = |r_static + n|, n Gaussian with
  per-axis variance sigma_s^2 = (theta_j L)^2 + <r_c^2>/2 (pointing jitter
  theta_j [rad] per axis + beam wander)
* h_t: turbulence fading, mean 1 (physics.turbulence; none, log-normal or
  Gamma-Gamma)

The model owns both the analytic statistics (mean gain, outage) and the
sampler, so Monte Carlo and analysis use the same equations.
Outage probability for a gain threshold h_th:

    P_out = P(h < h_th) = E_r[ F_t( h_th / (c h_p(r)) ) ],  c = eta_tx eta_rx exp(-beta L)

evaluated by quadrature over the Rice density of r (smooth integrand), or
exactly as the Rice survival probability P(R > r_th) when there is no
turbulence (h_p decreases monotonically in r), or directly when r is
deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from optobuild.physics.atmospheric import rain_attenuation, visibility_attenuation
from optobuild.physics.free_space import (
    aperture_fraction,
    beam_radius,
    beam_wander_variance,
    diffraction_limited_divergence,
    displacement_quadrature,
    mean_aperture_fraction,
)
from optobuild.physics.turbulence import (
    aperture_averaging_factor,
    gamma_gamma_cdf,
    gamma_gamma_parameters,
    gamma_gamma_scintillation_index,
    lognormal_cdf,
    rytov_variance,
    sample_gamma_gamma,
    sample_lognormal,
)

TURBULENCE_MODELS = ("none", "lognormal", "gamma_gamma")


@dataclass(frozen=True)
class FSOChannelModel:
    """Parameters of a horizontal FSO link (all SI)."""

    distance: float
    wavelength: float
    beam_waist: float
    """Transmitted 1/e^2 beam radius w0 [m]."""
    divergence: float = 0.0
    """Far-field 1/e^2 half-angle [rad]; 0 = diffraction limited (lambda / (pi w0))."""
    aperture_diameter: float = 0.1
    tx_efficiency: float = 1.0
    rx_efficiency: float = 1.0
    visibility: float = math.inf
    """[m]; inf = no fog/haze attenuation."""
    attenuation_model: str = "kim"
    rain_rate: float = 0.0
    """[m/s]."""
    pointing_offset: float = 0.0
    """Static angular misalignment [rad]."""
    pointing_jitter: float = 0.0
    """Random angular jitter, standard deviation per axis [rad]."""
    turbulence: str = "none"
    cn2: float = 0.0
    """Refractive-index structure parameter [m^-2/3]."""
    beam_wander: bool = False
    aperture_averaging: bool = True

    def __post_init__(self) -> None:
        if self.turbulence not in TURBULENCE_MODELS:
            raise ValueError(f"turbulence must be one of {TURBULENCE_MODELS}.")
        if self.turbulence != "none" and self.cn2 <= 0:
            raise ValueError("A turbulence model needs cn2 > 0.")

    # --- deterministic parts ----------------------------------------------------------------
    @property
    def effective_divergence(self) -> float:
        """Divergence used [rad] (diffraction limit if 0 was given)."""
        dl = diffraction_limited_divergence(self.wavelength, self.beam_waist)
        return dl if self.divergence == 0.0 else self.divergence

    @property
    def beam_radius_rx(self) -> float:
        """w(L) at the receiver [m]."""
        return beam_radius(self.distance, self.beam_waist, self.effective_divergence)

    @property
    def atmospheric_coefficient(self) -> float:
        """beta_fog + beta_rain [1/m]."""
        fog = (
            0.0
            if math.isinf(self.visibility)
            else visibility_attenuation(self.visibility, self.wavelength, self.attenuation_model)
        )
        return fog + rain_attenuation(self.rain_rate)

    @property
    def atmospheric_transmission(self) -> float:
        return math.exp(-self.atmospheric_coefficient * self.distance)

    @property
    def static_offset(self) -> float:
        """Static beam displacement at the receiver [m]."""
        return self.pointing_offset * self.distance

    @property
    def sigma_axis(self) -> float:
        """Per-axis std of the random displacement [m] (jitter + beam wander)."""
        var = (self.pointing_jitter * self.distance) ** 2
        if self.beam_wander and self.cn2 > 0:
            var += 0.5 * beam_wander_variance(self.cn2, self.distance, self.beam_waist)
        return math.sqrt(var)

    @property
    def constant_gain(self) -> float:
        """eta_tx eta_rx exp(-beta L)."""
        return self.tx_efficiency * self.rx_efficiency * self.atmospheric_transmission

    def geometric_fraction(self) -> float:
        """h_p for the static offset only (no random displacement)."""
        return float(
            aperture_fraction(0.5 * self.aperture_diameter, self.beam_radius_rx, self.static_offset)
        )

    def mean_pointing_fraction(self) -> float:
        """E[h_p(r)] including random displacement."""
        return mean_aperture_fraction(
            0.5 * self.aperture_diameter, self.beam_radius_rx, self.sigma_axis, self.static_offset
        )

    def mean_gain(self) -> float:
        """E[h] (turbulence fading has mean 1)."""
        return self.constant_gain * self.mean_pointing_fraction()

    # --- turbulence -------------------------------------------------------------------------
    @property
    def rytov_variance(self) -> float:
        return 0.0 if self.cn2 == 0 else rytov_variance(self.cn2, self.wavelength, self.distance)

    @property
    def scintillation_index(self) -> float:
        """sigma_I^2 of the turbulence model in use (0 without turbulence)."""
        if self.turbulence == "none":
            return 0.0
        if self.turbulence == "lognormal":
            s = self.rytov_variance
            if self.aperture_averaging:
                s *= aperture_averaging_factor(
                    self.wavelength, self.distance, self.aperture_diameter
                )
            return s
        return gamma_gamma_scintillation_index(*gamma_gamma_parameters(self.rytov_variance))

    def turbulence_cdf(self, h: ArrayLike) -> NDArray[np.float64] | float:
        """P(h_t <= h) (vectorized in h)."""
        hh = np.asarray(h, dtype=float)
        if self.turbulence == "none":
            return np.where(hh >= 1.0, 1.0, 0.0)[()]
        if self.turbulence == "lognormal":
            return np.vectorize(lambda x: lognormal_cdf(x, self.scintillation_index))(hh)[()]
        return gamma_gamma_cdf(hh, *gamma_gamma_parameters(self.rytov_variance))

    # --- statistics -------------------------------------------------------------------------
    def outage_probability(self, threshold_gain: float, n_points: int = 200) -> float:
        """P(h < threshold_gain)."""
        c = self.constant_gain
        a = 0.5 * self.aperture_diameter
        w = self.beam_radius_rx
        if self.sigma_axis == 0.0:
            hp = float(aperture_fraction(a, w, self.static_offset))
            return float(self.turbulence_cdf(threshold_gain / (c * hp)))
        if self.turbulence == "none":
            # h_p(r) decreases monotonically in r: P(h < h_th) = P(R > r_th), exactly
            target = threshold_gain / c
            if target > float(aperture_fraction(a, w, 0.0)):
                return 1.0
            if target <= 0.0:
                return 0.0
            r_hi = w
            while float(aperture_fraction(a, w, r_hi)) >= target:
                r_hi *= 2.0
            r_th = scipy.optimize.brentq(
                lambda r: float(aperture_fraction(a, w, r)) - target,
                0.0,
                r_hi,
                xtol=1e-15,
                rtol=1e-13,
            )
            return float(
                scipy.stats.rice.sf(
                    r_th, self.static_offset / self.sigma_axis, scale=self.sigma_axis
                )
            )
        r, wts = displacement_quadrature(self.static_offset, self.sigma_axis, n_points)
        hp = np.asarray(aperture_fraction(a, w, r))
        return float(np.sum(wts * self.turbulence_cdf(threshold_gain / (c * hp))))

    def sample(self, rng: np.random.Generator, size: int) -> dict[str, NDArray[np.float64]]:
        """Random channel states: total gain and its factors (pointing, turbulence)."""
        a = 0.5 * self.aperture_diameter
        sig = self.sigma_axis
        if sig > 0:
            dx = self.static_offset + sig * rng.standard_normal(size)
            dy = sig * rng.standard_normal(size)
            hp = np.asarray(aperture_fraction(a, self.beam_radius_rx, np.hypot(dx, dy)))
        else:
            hp = np.full(size, self.geometric_fraction())
        if self.turbulence == "lognormal":
            ht = sample_lognormal(rng, self.scintillation_index, size)
        elif self.turbulence == "gamma_gamma":
            ht = sample_gamma_gamma(rng, *gamma_gamma_parameters(self.rytov_variance), size)
        else:
            ht = np.ones(size)
        return {"gain": self.constant_gain * hp * ht, "pointing": hp, "turbulence": ht}


__all__ = ["TURBULENCE_MODELS", "FSOChannelModel"]
