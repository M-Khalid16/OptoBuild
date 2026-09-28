"""Polarization optics: Jones matrices, differential group delay and PMD (physics_models.md 3.20).

Field: Jones vector A = (A_x, A_y) along axis 0 of an ``(2, N)`` envelope
[sqrt(W)]; P = |A_x|^2 + |A_y|^2. A lossless element is a unitary 2x2
matrix, possibly frequency dependent: A~_out(w) = J(w) A~_in(w), with w [rad/s]
the baseband angular frequency (our FFT convention, delay tau <-> exp(-i w tau)).

Elements (all with det J = 1 unless stated):

* rotation / general SOP transformation (azimuth theta, phase phi) [rad]::

      U(theta, phi) = [[cos(theta), -sin(theta) exp(-i phi)],
                       [sin(theta) exp(+i phi),  cos(theta)]]

* linear retarder (waveplate) with retardance delta and fast axis at
  azimuth theta: R(-theta) diag(exp(-i delta/2), exp(+i delta/2)) R(theta)
* differential group delay tau [s] between the principal axes (x slow, y fast):
  D(w) = diag(exp(-i w tau/2), exp(+i w tau/2))
* Haar-random element of SU(2): (a, b) a normalized complex Gaussian
  4-vector, U = [[a, -conj(b)], [b, conj(a)]] (uniform on the Poincare sphere)
* PMD, waveplate (retarded-plate) model: J(w) = D_N(w) U_N ... D_1(w) U_1
  with Haar-random U_k and section DGDs tau_k. With independent random
  coupling the PMD vector obeys Omega_k = tau_k e_k + R_k Omega_{k-1} with
  R_k isotropic, so exactly E[|Omega_N|^2] = sum tau_k^2 at every frequency;
  for many sections |Omega| is Maxwellian with mean
  <tau> = sqrt(8 / (3 pi)) sqrt(E[tau^2]) (Foschini & Poole, J. Lightwave
  Technol. 9, 1439 (1991); Gordon & Kogelnik, PNAS 97, 4541 (2000)).

DGD measurement (Jones-matrix eigenanalysis, Heffner, IEEE Photon. Technol.
Lett. 4, 1066 (1992)): for w1 < w2 the eigenvalues rho_1,2 of
J(w2) J(w1)^H give tau = |arg(rho_1 / rho_2)| / (w2 - w1) (exact for a
frequency-independent PMD vector, O(dw^2) otherwise).

Assumptions: lossless, no polarization-dependent loss, first-order
birefringence per section (frequency-independent tau_k), frequency-independent
coupling U_k, static (no temporal drift within a simulation window).
Validation: tests/validation/test_polarization.py.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.numerics.fft import inverse_spectrum, spectrum
from optobuild.numerics.grid import TimeGrid

MAXWELLIAN_MEAN_TO_RMS = math.sqrt(8.0 / (3.0 * math.pi))


def rotation(theta: float) -> NDArray[np.complex128]:
    """Real rotation of the polarization axes by ``theta`` [rad]."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def sop_transform(theta: float, phi: float = 0.0) -> NDArray[np.complex128]:
    """U(theta, phi): general SU(2) transformation with azimuth theta and phase phi [rad]."""
    c, s = math.cos(theta), math.sin(theta)
    e = complex(math.cos(phi), math.sin(phi))
    return np.array([[c, -s * e.conjugate()], [s * e, c]], dtype=complex)


def waveplate(retardance: float, axis_angle: float = 0.0) -> NDArray[np.complex128]:
    """Linear retarder: retardance delta [rad], fast axis at azimuth ``axis_angle`` [rad]."""
    d = np.diag([np.exp(-0.5j * retardance), np.exp(0.5j * retardance)])
    return rotation(-axis_angle) @ d @ rotation(axis_angle)


def random_unitary(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Haar-distributed element of SU(2)."""
    g = rng.standard_normal(4)
    g /= np.linalg.norm(g)
    a, b = complex(g[0], g[1]), complex(g[2], g[3])
    return np.array([[a, -b.conjugate()], [b, a.conjugate()]], dtype=complex)


def dgd_element(omega: ArrayLike, dgd: float) -> NDArray[np.complex128]:
    """D(w) = diag(exp(-i w tau/2), exp(+i w tau/2)), shape (2, 2, N)."""
    w = np.asarray(omega, dtype=float)
    out = np.zeros((2, 2, w.size), dtype=complex)
    out[0, 0] = np.exp(-0.5j * w * dgd)
    out[1, 1] = np.exp(0.5j * w * dgd)
    return out


def pmd_jones(
    omega: ArrayLike, section_dgds: ArrayLike, couplings: ArrayLike
) -> NDArray[np.complex128]:
    """Waveplate PMD model J(w) = D_N U_N ... D_1 U_1, shape (2, 2, N).

    ``section_dgds`` [s], shape (K,); ``couplings`` unitary matrices, shape (K, 2, 2).
    """
    w = np.asarray(omega, dtype=float)
    taus = np.asarray(section_dgds, dtype=float)
    us = np.asarray(couplings, dtype=complex)
    if us.shape != (taus.size, 2, 2):
        raise ValueError(f"need {taus.size} coupling matrices of shape (2, 2), got {us.shape}")
    j = np.broadcast_to(np.eye(2, dtype=complex)[:, :, None], (2, 2, w.size)).copy()
    for tau, u in zip(taus, us, strict=True):
        j = np.einsum("ij,jkn->ikn", u, j)
        j[0] *= np.exp(-0.5j * w * tau)
        j[1] *= np.exp(0.5j * w * tau)
    return j


def apply_jones(field: ArrayLike, jones: ArrayLike, grid: TimeGrid) -> NDArray[np.complex128]:
    """Apply a constant (2, 2) or frequency-dependent (2, 2, N) Jones matrix to a (2, N) field.

    A frequency-dependent matrix is given on ``grid.angular_frequency()`` (FFT order);
    the product is taken per frequency bin (circular in time, like any transfer function).
    """
    a = np.asarray(field, dtype=complex)
    j = np.asarray(jones, dtype=complex)
    if a.shape[0] != 2:
        raise ValueError(f"Jones matrices act on dual-polarization fields, got shape {a.shape}")
    if j.shape == (2, 2):
        return j @ a
    if j.shape != (2, 2, grid.n_samples):
        raise ValueError(f"Jones matrix shape {j.shape} does not match the grid")
    # the per-bin product commutes with the origin phase of spectrum()/inverse_spectrum()
    return inverse_spectrum(np.einsum("ikn,kn->in", j, spectrum(a, grid)), grid)


def dgd_from_jones(j1: ArrayLike, j2: ArrayLike, delta_omega: float) -> float:
    """Jones-matrix eigenanalysis: DGD [s] from J(w) and J(w + delta_omega)."""
    m = np.asarray(j2, dtype=complex) @ np.asarray(j1, dtype=complex).conj().T
    rho = np.linalg.eigvals(m)
    return float(abs(np.angle(rho[0] / rho[1])) / delta_omega)


def waveplate_section_dgd(mean_dgd: float, n_sections: int) -> float:
    """Equal section DGD giving mean DGD ``mean_dgd`` in the Maxwellian limit.

    E[tau^2] = N tau_s^2 and <tau> = sqrt(8/(3 pi)) sqrt(E[tau^2])
    => tau_s = <tau> / (sqrt(8/(3 pi)) sqrt(N)).
    """
    return mean_dgd / (MAXWELLIAN_MEAN_TO_RMS * math.sqrt(n_sections))


__all__ = [
    "MAXWELLIAN_MEAN_TO_RMS",
    "apply_jones",
    "dgd_element",
    "dgd_from_jones",
    "pmd_jones",
    "random_unitary",
    "rotation",
    "sop_transform",
    "waveplate",
    "waveplate_section_dgd",
]
