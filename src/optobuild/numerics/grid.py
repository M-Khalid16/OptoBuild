"""Uniform time grid and its reciprocal frequency grid.

Conventions (docs/numerical_conventions.md, ADR-0002)::

    t_n   = t0 + n * dt,             n = 0 .. N-1           [s]
    fs    = 1 / dt                                          [Hz]
    T     = N * dt                   (window duration)      [s]
    df    = 1 / T                                           [Hz]
    f_k   = numpy.fft.fftfreq(N, dt)  (unshifted, "FFT order") [Hz]
    w_k   = 2 * pi * f_k                                    [rad/s]

Frequencies are *baseband* offsets relative to a signal's reference (carrier)
frequency; absolute optical frequency is ``nu = f_ref + f_k``.
Arrays are returned in FFT order; ``fftshift`` is applied only for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optobuild.core.errors import SamplingError


@dataclass(frozen=True)
class TimeGrid:
    """A uniform, finite, periodic sampling grid.

    Attributes
    ----------
    n_samples:
        Number of samples N (>= 2). Powers of two are fastest but not required.
    dt:
        Sampling interval [s], finite and > 0.
    t0:
        Time of the first sample [s]. In a retarded frame this carries the
        accumulated group delay (see docs/signal_model.md).
    """

    n_samples: int
    dt: float
    t0: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, (int, np.integer)):
            raise SamplingError(
                f"n_samples must be an integer, got {self.n_samples!r}.",
                hint="Pass the number of samples as an int.",
            )
        if self.n_samples < 2:
            raise SamplingError(
                f"n_samples must be >= 2, got {self.n_samples}.",
                hint="Increase the number of symbols or samples per symbol.",
            )
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise SamplingError(
                f"dt must be finite and > 0 s, got {self.dt!r}.",
                hint="Specify a positive sampling interval (1 / sample rate).",
            )
        if not math.isfinite(self.t0):
            raise SamplingError(f"t0 must be finite, got {self.t0!r}.")
        object.__setattr__(self, "n_samples", int(self.n_samples))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "t0", float(self.t0))

    @classmethod
    def from_sample_rate(cls, n_samples: int, sample_rate: float, t0: float = 0.0) -> TimeGrid:
        """Create a grid from a sample rate ``fs`` [Hz]."""
        if not (math.isfinite(sample_rate) and sample_rate > 0.0):
            raise SamplingError(
                f"sample_rate must be finite and > 0 Hz, got {sample_rate!r}.",
                hint="Specify a positive sample rate.",
            )
        return cls(n_samples=n_samples, dt=1.0 / sample_rate, t0=t0)

    # --- derived scalars -------------------------------------------------------------
    @property
    def sample_rate(self) -> float:
        """Sampling rate fs = 1/dt [Hz]; also the full simulated baseband span."""
        return 1.0 / self.dt

    @property
    def duration(self) -> float:
        """Window duration T = N dt [s] (the period of the implied periodic signal)."""
        return self.n_samples * self.dt

    @property
    def df(self) -> float:
        """Frequency resolution df = 1/T [Hz]."""
        return 1.0 / self.duration

    @property
    def nyquist_frequency(self) -> float:
        """fs/2 [Hz]: the largest representable |baseband frequency|."""
        return 0.5 * self.sample_rate

    # --- axes --------------------------------------------------------------------------
    def time(self) -> NDArray[np.float64]:
        """Sample times t_n = t0 + n dt [s]."""
        return self.t0 + self.dt * np.arange(self.n_samples, dtype=np.float64)

    def frequency(self, *, shifted: bool = False) -> NDArray[np.float64]:
        """Baseband frequencies f_k [Hz], FFT order unless ``shifted`` (display order)."""
        f = np.fft.fftfreq(self.n_samples, d=self.dt)
        return np.fft.fftshift(f) if shifted else f

    def angular_frequency(self, *, shifted: bool = False) -> NDArray[np.float64]:
        """Baseband angular frequencies w_k = 2 pi f_k [rad/s]."""
        return 2.0 * np.pi * self.frequency(shifted=shifted)

    def with_t0(self, t0: float) -> TimeGrid:
        """Return the same grid with a different start time (e.g. after a pure delay)."""
        return TimeGrid(self.n_samples, self.dt, t0)

    def is_compatible(self, other: TimeGrid, *, rtol: float = 1e-12) -> bool:
        """True if both grids have the same N and dt (t0 may differ)."""
        return self.n_samples == other.n_samples and math.isclose(
            self.dt, other.dt, rel_tol=rtol, abs_tol=0.0
        )


__all__ = ["TimeGrid"]
