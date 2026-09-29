"""Atmospheric turbulence: scintillation statistics (physics_models.md 3.14).

Rytov variance (plane wave, Kolmogorov spectrum, horizontal path of constant Cn^2):

    sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6),   k = 2 pi / lambda

Irradiance fading h_t (normalized, E[h_t] = 1):

* log-normal (weak turbulence, sigma_R^2 < ~1): ln h_t ~ N(-s^2/2, s^2),
  s^2 = ln(1 + sigma_I^2), with scintillation index sigma_I^2 = sigma_R^2 (weak,
  plane wave), optionally reduced by aperture averaging.
* Gamma-Gamma (moderate to strong, Al-Habash, Andrews & Phillips 2001):

      alpha = [exp(0.49 sR^2 / (1 + 1.11 sR^(12/5))^(7/6)) - 1]^-1
      beta  = [exp(0.51 sR^2 / (1 + 0.69 sR^(12/5))^(5/6)) - 1]^-1
      f(h) = 2 (alpha beta)^((alpha+beta)/2) / (Gamma(alpha) Gamma(beta))
             h^((alpha+beta)/2 - 1) K_(alpha-beta)(2 sqrt(alpha beta h))
      sigma_I^2 = 1/alpha + 1/beta + 1/(alpha beta)

  (point receiver, plane wave, zero inner scale). Sampling: h = X Y with
  X ~ Gamma(alpha, 1/alpha), Y ~ Gamma(beta, 1/beta).

Aperture averaging (plane wave, weak turbulence, receiver diameter D):

    A = [1 + 1.062 k D^2 / (4 L)]^(-7/6),    sigma_I^2(D) = A sigma_I^2(0)

References: L. C. Andrews & R. L. Phillips, Laser Beam Propagation through
Random Media, 2nd ed., SPIE Press (2005), ch. 8-11; M. A. Al-Habash,
L. C. Andrews, R. L. Phillips, Opt. Eng. 40, 1554 (2001).
Limitations: constant Cn^2 along the path; no inner/outer-scale corrections;
the aperture-averaging factor is used only with the log-normal model
(Gamma-Gamma parameters above are for a point receiver).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray


def rytov_variance(cn2: float, wavelength: float, distance: float) -> float:
    """sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6) (plane wave)."""
    k = 2.0 * math.pi / wavelength
    return 1.23 * cn2 * k ** (7.0 / 6.0) * distance ** (11.0 / 6.0)


def aperture_averaging_factor(wavelength: float, distance: float, diameter: float) -> float:
    """A = [1 + 1.062 k D^2/(4L)]^(-7/6) (plane wave, weak turbulence)."""
    k = 2.0 * math.pi / wavelength
    return (1.0 + 1.062 * k * diameter**2 / (4.0 * distance)) ** (-7.0 / 6.0)


def gamma_gamma_parameters(rytov: float) -> tuple[float, float]:
    """(alpha, beta) of the Gamma-Gamma model for plane-wave Rytov variance ``rytov``."""
    if rytov <= 0:
        raise ValueError(f"Rytov variance must be > 0, got {rytov}.")
    s125 = rytov ** (6.0 / 5.0)  # sigma_R^(12/5)
    a = 1.0 / math.expm1(0.49 * rytov / (1.0 + 1.11 * s125) ** (7.0 / 6.0))
    b = 1.0 / math.expm1(0.51 * rytov / (1.0 + 0.69 * s125) ** (5.0 / 6.0))
    return a, b


def gamma_gamma_scintillation_index(alpha: float, beta: float) -> float:
    """sigma_I^2 = 1/alpha + 1/beta + 1/(alpha beta)."""
    return 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)


def gamma_gamma_pdf(h: ArrayLike, alpha: float, beta: float) -> NDArray[np.float64]:
    """Gamma-Gamma probability density of the normalized irradiance h > 0."""
    x = np.asarray(h, dtype=float)
    ab = alpha * beta
    log_c = (
        math.log(2.0)
        + 0.5 * (alpha + beta) * math.log(ab)
        - scipy.special.gammaln(alpha)
        - scipy.special.gammaln(beta)
    )
    z = 2.0 * np.sqrt(ab * x)
    # kve(v, z) = kv(v, z) exp(z): stable for large z
    return np.exp(log_c + (0.5 * (alpha + beta) - 1.0) * np.log(x) - z) * scipy.special.kve(
        alpha - beta, z
    )


def gamma_gamma_cdf(h: ArrayLike, alpha: float, beta: float) -> NDArray[np.float64] | float:
    """P(h_t <= h) = E_X[ F_Y(h / X) ] by quadrature over X's quantiles (vectorized in h).

    With X ~ Gamma(alpha, 1/alpha), Y ~ Gamma(beta, 1/beta): E[g(X)] =
    integral_0^1 g(F_X^-1(u)) du, evaluated with 400-point Gauss-Legendre on
    u in [1e-15, 1 - 1e-15].
    """
    hh = np.atleast_1d(np.asarray(h, dtype=float))
    gx = scipy.stats.gamma(alpha, scale=1.0 / alpha)
    gy = scipy.stats.gamma(beta, scale=1.0 / beta)
    u_lo, u_hi = 1e-15, 1.0 - 1e-15
    x, w = np.polynomial.legendre.leggauss(400)
    xs = gx.ppf(u_lo + 0.5 * (x + 1.0) * (u_hi - u_lo))
    ratio = np.where(hh[:, None] > 0, hh[:, None] / xs[None, :], 0.0)
    out = 0.5 * (u_hi - u_lo) * (gy.cdf(ratio) @ w)
    out = np.where(hh > 0, out, 0.0)
    return out if np.ndim(h) else float(out[0])


def lognormal_sigma2(scintillation_index: float) -> float:
    """Variance of ln h for a log-normal fading with given sigma_I^2 (mean 1)."""
    return math.log1p(scintillation_index)


def lognormal_cdf(h: float, scintillation_index: float) -> float:
    """P(h_t <= h), ln h ~ N(-s^2/2, s^2)."""
    if h <= 0:
        return 0.0
    s2 = lognormal_sigma2(scintillation_index)
    return float(scipy.stats.norm.cdf((math.log(h) + 0.5 * s2) / math.sqrt(s2)))


def sample_lognormal(
    rng: np.random.Generator, scintillation_index: float, size: int
) -> NDArray[np.float64]:
    """Samples of mean-1 log-normal irradiance."""
    s2 = lognormal_sigma2(scintillation_index)
    return np.exp(rng.normal(-0.5 * s2, math.sqrt(s2), size))


def sample_gamma_gamma(
    rng: np.random.Generator, alpha: float, beta: float, size: int
) -> NDArray[np.float64]:
    """Samples of mean-1 Gamma-Gamma irradiance (product of two unit-mean gammas)."""
    return rng.gamma(alpha, 1.0 / alpha, size) * rng.gamma(beta, 1.0 / beta, size)


__all__ = [
    "aperture_averaging_factor",
    "gamma_gamma_cdf",
    "gamma_gamma_parameters",
    "gamma_gamma_pdf",
    "gamma_gamma_scintillation_index",
    "lognormal_cdf",
    "lognormal_sigma2",
    "rytov_variance",
    "sample_gamma_gamma",
    "sample_lognormal",
]
