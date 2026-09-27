"""Free-space beam propagation, geometric and pointing loss (physics_models.md 3.12).

Gaussian beam (1/e^2 intensity radius w):

    w(L)^2 = w0^2 + (theta L)^2          theta: far-field half-angle divergence [rad]
    theta_dl = lambda / (pi w0)          diffraction limit (w(L) is then exact)

Received fraction of a Gaussian spot whose centre is displaced by r [m] from
the centre of a circular aperture of radius a [m]: with sigma = w/2 the
intensity is a 2-D normal density, so

    h(r) = P(|X| <= a) = F_ncx2( (a/sigma)^2 ; k = 2, lambda_nc = (r/sigma)^2 )

(non-central chi-square CDF, i.e. 1 - Q_1(r/sigma, a/sigma) with the Marcum Q
function). For r = 0: h = 1 - exp(-2 a^2 / w^2).

Random pointing: jitter (per-axis standard deviation sigma_s at the receiver
plane) and beam wander are modelled as independent zero-mean Gaussian
displacements, so r is Rayleigh distributed with per-axis variance
sigma_s^2 + <r_c^2>/2. Beam wander of a collimated beam in weak turbulence
(infinite outer scale): <r_c^2> = 2.42 Cn^2 L^3 w0^(-1/3)
(Andrews & Phillips, Laser Beam Propagation through Random Media, 2nd ed.,
2005, ch. 6).

Assumptions: far-field Gaussian beam in the paraxial approximation; receiver
aperture in the beam's transverse plane; obscurations and angle-of-arrival
effects not modelled. References: A. A. Farid & S. Hranilovic, J. Lightwave
Technol. 25, 1702 (2007); H. Kaushal & G. Kaddoum, IEEE Commun. Surv. Tutor.
19, 57 (2017).
"""

from __future__ import annotations

import functools
import math

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import ArrayLike, NDArray


def diffraction_limited_divergence(wavelength: float, waist: float) -> float:
    """theta_dl = lambda / (pi w0) [rad] (1/e^2 half angle)."""
    return wavelength / (math.pi * waist)


def beam_radius(distance: float, waist: float, divergence: float) -> float:
    """w(L) = sqrt(w0^2 + (theta L)^2) [m]."""
    return math.sqrt(waist**2 + (divergence * distance) ** 2)


def aperture_fraction(
    aperture_radius: float, beam_radius_m: float, offset: ArrayLike = 0.0
) -> NDArray[np.float64] | float:
    """Fraction of a Gaussian beam's power collected by a circular aperture."""
    sigma = 0.5 * beam_radius_m
    r = np.asarray(offset, dtype=float)
    # scipy.special.chndtr is the non-central chi-square CDF (identical to
    # scipy.stats.ncx2.cdf without the distribution-object overhead)
    return scipy.special.chndtr(
        (aperture_radius / sigma) ** 2, 2, np.maximum((r / sigma) ** 2, 1e-300)
    )[()]


def beam_wander_variance(cn2: float, distance: float, waist: float) -> float:
    """<r_c^2> = 2.42 Cn^2 L^3 w0^(-1/3) [m^2] (collimated beam, weak turbulence)."""
    return 2.42 * cn2 * distance**3 * waist ** (-1.0 / 3.0)


@functools.lru_cache(maxsize=8)
def _leggauss(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cached Gauss-Legendre nodes/weights on [-1, 1] (read-only arrays)."""
    x, w = np.polynomial.legendre.leggauss(n)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def displacement_pdf(r: ArrayLike, static_offset: float, sigma_axis: float) -> NDArray[np.float64]:
    """Density of the radial displacement r = |r_static + n|, n ~ N(0, sigma^2 I) (Rice)."""
    rr = np.asarray(r, dtype=float)
    return scipy.stats.rice.pdf(rr, static_offset / sigma_axis, scale=sigma_axis)


def displacement_quadrature(
    static_offset: float, sigma_axis: float, n_points: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Legendre nodes/weights covering the Rice density (to 12 sigma)."""
    r_lo = max(0.0, static_offset - 12.0 * sigma_axis)
    r_hi = static_offset + 12.0 * sigma_axis
    x, w = _leggauss(n_points)
    r = r_lo + 0.5 * (r_hi - r_lo) * (x + 1.0)
    return r, 0.5 * (r_hi - r_lo) * w * displacement_pdf(r, static_offset, sigma_axis)


def mean_aperture_fraction(
    aperture_radius: float,
    beam_radius_m: float,
    sigma_axis: float,
    static_offset: float = 0.0,
    n_points: int = 400,
) -> float:
    """E[h(r)] over the Rice-distributed displacement (Gauss-Legendre quadrature)."""
    if sigma_axis == 0.0:
        return float(aperture_fraction(aperture_radius, beam_radius_m, static_offset))
    r, w = displacement_quadrature(static_offset, sigma_axis, n_points)
    return float(np.sum(w * aperture_fraction(aperture_radius, beam_radius_m, r)))


__all__ = [
    "aperture_fraction",
    "beam_radius",
    "beam_wander_variance",
    "diffraction_limited_divergence",
    "displacement_pdf",
    "displacement_quadrature",
    "mean_aperture_fraction",
]
