"""Thin-film transfer-matrix method at normal incidence (physics_models.md 3.24).

Characteristic matrix of a homogeneous layer (index n_j, thickness d_j,
phase thickness delta_j = 2 pi n_j d_j / lambda), Macleod, *Thin-Film
Optical Filters*, 4th ed., sec. 2.2:

    M_j = [[cos delta_j, i sin delta_j / n_j], [i n_j sin delta_j, cos delta_j]]

[B, C]^T = (prod_j M_j) [1, n_s]^T for a stack between incidence medium n_0
and substrate n_s; then

    r = (n_0 B - C) / (n_0 B + C),  t = 2 n_0 / (n_0 B + C),
    R = |r|^2,  T = n_s |t|^2 / n_0  (real, lossless media: R + T = 1).

Quarter-wave stack (HL)^N at its design wavelength: admittance
Y = (n_H / n_L)^(2N) n_s, R = ((n_0 - Y) / (n_0 + Y))^2.

Assumptions: normal incidence, isotropic, non-magnetic, real or complex
(absorbing, n - i k in our convention) indices, abrupt interfaces.
Used as an independent reference for coupled-mode Bragg gratings.
Validation: tests/validation/test_integrated_optics.py.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

_C = NDArray[np.complex128]
_R = NDArray[np.float64]


def stack_response(
    wavelength: ArrayLike,
    indices: ArrayLike,
    thicknesses: ArrayLike,
    n_incident: complex,
    n_substrate: complex,
) -> tuple[_C, _C, _R, _R]:
    """(r, t, R, T) of a layer stack for each wavelength [m]; layers ordered from the
    incidence side."""
    lam = np.atleast_1d(np.asarray(wavelength, dtype=float))
    n = np.asarray(indices, dtype=complex)
    d = np.asarray(thicknesses, dtype=float)
    if n.shape != d.shape or n.ndim != 1:
        raise ValueError("indices and thicknesses must be 1-D arrays of equal length")
    m11 = np.ones(lam.size, dtype=complex)
    m12 = np.zeros(lam.size, dtype=complex)
    m21 = np.zeros(lam.size, dtype=complex)
    m22 = np.ones(lam.size, dtype=complex)
    for nj, dj in zip(n, d, strict=True):
        delta = 2 * math.pi * nj * dj / lam
        c, s = np.cos(delta), np.sin(delta)
        a11, a12, a21, a22 = c, 1j * s / nj, 1j * nj * s, c
        m11, m12, m21, m22 = (
            m11 * a11 + m12 * a21,
            m11 * a12 + m12 * a22,
            m21 * a11 + m22 * a21,
            m21 * a12 + m22 * a22,
        )
    b = m11 + m12 * n_substrate
    cc = m21 + m22 * n_substrate
    den = n_incident * b + cc
    r = (n_incident * b - cc) / den
    t = 2 * n_incident / den
    big_r = np.abs(r) ** 2
    big_t = np.real(n_substrate) / np.real(n_incident) * np.abs(t) ** 2
    return r, t, big_r, big_t


def quarter_wave_reflectance(
    n_high: float, n_low: float, pairs: int, n_incident: float, n_substrate: float
) -> float:
    """Peak reflectance of (HL)^N on a substrate at the design wavelength."""
    y = (n_high / n_low) ** (2 * pairs) * n_substrate
    return ((n_incident - y) / (n_incident + y)) ** 2


__all__ = ["quarter_wave_reflectance", "stack_response"]
