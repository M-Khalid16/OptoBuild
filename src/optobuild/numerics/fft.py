"""The project-wide Fourier-transform convention (ADR-0002).

All spectral computations in OptoBuild must go through this module so that the
sign, scaling and ordering conventions are defined exactly once.

Continuous convention (engineering sign, ordinary frequency f in Hz)::

    X(f) = integral x(t) exp(-i 2 pi f t) dt          (forward)
    x(t) = integral X(f) exp(+i 2 pi f t) df          (inverse)

With the optical field written as E(t) = Re{A(t) exp(+i 2 pi f_ref t)}, a
positive baseband frequency f therefore corresponds to the absolute optical
frequency f_ref + f (i.e. a *shorter* wavelength).

Discrete approximation on a :class:`~optobuild.numerics.grid.TimeGrid`::

    X_k = dt * sum_n x_n exp(-i 2 pi k n / N) * exp(-i 2 pi f_k t0)
        = dt * fft(x)[k] * exp(-i 2 pi f_k t0)
    x_n = df * sum_k X_k exp(+i 2 pi k n / N) * exp(+i 2 pi f_k t0)
        = ifft(X * exp(+i 2 pi f t0))[n] / dt

Units: if x is in sqrt(W) then X is in sqrt(W)*s = sqrt(J/Hz).

Parseval (exact for the discrete pair above)::

    sum_n |x_n|^2 dt  ==  sum_k |X_k|^2 df            [J]

Two-sided power spectral density of a signal that is periodic in the window
T = N dt (every simulated signal is)::

    S_k = |X_k|^2 / T                                 [W/Hz]
    sum_k S_k df == mean_n |x_n|^2                   [W]

Time is always the **last** array axis; leading axes (e.g. polarization) are
transformed independently. Arrays are in FFT order (DC first); use
:func:`to_display_order` only for plotting/reporting.
"""

from __future__ import annotations

import numpy as np
import scipy.fft
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import SamplingError
from optobuild.numerics.grid import TimeGrid


def _as_signal_array(x: ArrayLike, grid: TimeGrid) -> NDArray[np.complexfloating]:
    arr = np.asarray(x)
    if arr.ndim == 0 or arr.shape[-1] != grid.n_samples:
        raise SamplingError(
            f"Last axis of the array has length {arr.shape[-1] if arr.ndim else 0}, "
            f"but the grid has {grid.n_samples} samples.",
            hint="Time must be the last axis and match the TimeGrid length.",
        )
    return arr


def _origin_phase(grid: TimeGrid, sign: float) -> NDArray[np.complex128] | None:
    if grid.t0 == 0.0:
        return None
    return np.exp(sign * 2j * np.pi * grid.frequency() * grid.t0)


def spectrum(x: ArrayLike, grid: TimeGrid) -> NDArray[np.complex128]:
    """Approximate continuous Fourier transform X(f_k) of samples x(t_n).

    Parameters
    ----------
    x:
        Samples with time along the last axis (length ``grid.n_samples``).
    grid:
        Sampling grid; its ``t0`` sets the time origin of the transform.

    Returns
    -------
    ndarray
        X_k in FFT order, units [x] * s.
    """
    arr = _as_signal_array(x, grid)
    out = grid.dt * scipy.fft.fft(arr, axis=-1)
    phase = _origin_phase(grid, -1.0)
    if phase is not None:
        out *= phase
    return out


def inverse_spectrum(spec: ArrayLike, grid: TimeGrid) -> NDArray[np.complex128]:
    """Inverse of :func:`spectrum`: samples x(t_n) from X(f_k) given in FFT order."""
    arr = _as_signal_array(spec, grid)
    phase = _origin_phase(grid, +1.0)
    if phase is not None:
        arr = arr * phase
    return scipy.fft.ifft(arr, axis=-1) / grid.dt


def apply_transfer_function(x: ArrayLike, transfer: ArrayLike, grid: TimeGrid) -> NDArray:
    """Filter x by a frequency response H(f_k) (FFT order): y = F^-1{ H * F{x} }.

    This is a circular (periodic) convolution over the window; see the
    wrap-around discussion in docs/numerical_conventions.md. The result is
    independent of ``grid.t0``. Real input with Hermitian H returns real
    output up to round-off; callers decide whether to take the real part.
    """
    arr = _as_signal_array(x, grid)
    h = np.asarray(transfer)
    if h.shape[-1] != grid.n_samples:
        raise SamplingError(
            f"Transfer function has {h.shape[-1]} points but the grid has {grid.n_samples}.",
            hint="Evaluate H on grid.frequency() (FFT order).",
        )
    return scipy.fft.ifft(scipy.fft.fft(arr, axis=-1) * h, axis=-1)


def energy(x: ArrayLike, grid: TimeGrid) -> NDArray[np.float64] | float:
    """Time-domain energy sum |x_n|^2 dt, summed over the last axis ([J] for sqrt(W))."""
    arr = _as_signal_array(x, grid)
    return np.sum(np.abs(arr) ** 2, axis=-1) * grid.dt


def spectral_energy(spec: ArrayLike, grid: TimeGrid) -> NDArray[np.float64] | float:
    """Frequency-domain energy sum |X_k|^2 df (equals :func:`energy` by Parseval)."""
    arr = _as_signal_array(spec, grid)
    return np.sum(np.abs(arr) ** 2, axis=-1) * grid.df


def mean_power(x: ArrayLike, grid: TimeGrid) -> NDArray[np.float64] | float:
    """Window-averaged power mean |x_n|^2 ([W] for sqrt(W))."""
    arr = _as_signal_array(x, grid)
    return np.mean(np.abs(arr) ** 2, axis=-1)


def power_spectral_density(x: ArrayLike, grid: TimeGrid) -> NDArray[np.float64]:
    """Two-sided (baseband) PSD S_k = |X_k|^2 / T in FFT order ([W/Hz] for sqrt(W)).

    Normalized so that ``sum(S) * df == mean_power(x)``. This is the
    periodogram of one realization (no averaging, rectangular window); its
    variance does not decrease with N. Averaging/windowing belong to the
    analysis layer.
    """
    spec = spectrum(x, grid)
    return np.abs(spec) ** 2 / grid.duration


def to_display_order(*arrays: ArrayLike) -> tuple[NDArray, ...]:
    """Apply ``fftshift`` on the last axis (for plotting/reporting only)."""
    return tuple(np.fft.fftshift(np.asarray(a), axes=-1) for a in arrays)


__all__ = [
    "apply_transfer_function",
    "energy",
    "inverse_spectrum",
    "mean_power",
    "power_spectral_density",
    "spectral_energy",
    "spectrum",
    "to_display_order",
]
