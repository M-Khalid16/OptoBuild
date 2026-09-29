"""Fourth-order Runge-Kutta in the interaction picture (RK4IP) for the GNLSE (ADR-0020).

Hult, J. Lightwave Technol. 25, 3770 (2007). With dA~/dz = D A~ + N(A~) and
the half-step linear propagator P = exp(D h/2), one step is

    A_I = P A~,  k1 = P N(A~),  k2 = N(A_I + h k1/2),  k3 = N(A_I + h k2/2),
    k4 = N(P (A_I + h k3)),     A~(z+h) = P (A_I + h (k1 + 2 k2 + 2 k3)/6) + h k4/6

(global error O(h^4); the linear part is exact). Step control (optional,
``tolerance``): step doubling; the relative difference delta between one step
of h and two of h/2 estimates the local error; a step is accepted if
delta <= 2 tol, the result is the local extrapolation (16 u_fine - u_coarse)/15,
and h <- h min(2, max(0.5, (tol/delta)^(1/5))) (Hult 2007, sec. III; Sinkin
et al. 2003 for the step-doubling idea). Work is done on the spectrum
(FFT order, numerics.fft conventions).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import NumericalStabilityError
from optobuild.numerics.fft import inverse_spectrum, spectrum
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.ultrafast import NonlinearOperator, dispersion_operator, photon_number

MAX_STEPS = 200_000


@dataclass(frozen=True)
class GNLSEReport:
    n_steps: int
    rejected_steps: int
    min_step: float
    max_step: float
    photon_number_change: float
    """Relative change of the photon number over the propagation (0 without loss)."""
    energy_change: float


def _rk4ip_step(
    spec: NDArray[np.complex128],
    h: float,
    d: NDArray[np.complex128],
    op: NonlinearOperator,
) -> NDArray[np.complex128]:
    p = np.exp(d * h / 2)
    a_i = p * spec
    k1 = p * op(spec)
    k2 = op(a_i + h / 2 * k1)
    k3 = op(a_i + h / 2 * k2)
    k4 = op(p * (a_i + h * k3))
    return p * (a_i + h / 6 * (k1 + 2 * k2 + 2 * k3)) + h / 6 * k4


def propagate_gnlse(
    field: ArrayLike,
    grid: TimeGrid,
    length: float,
    center_frequency: float,
    *,
    betas: Sequence[float],
    gamma: float,
    alpha: float = 0.0,
    self_steepening: bool = True,
    raman_fraction: float = 0.18,
    tau1: float = 12.2e-15,
    tau2: float = 32e-15,
    n_steps: int | None = None,
    tolerance: float = 1e-8,
    initial_step: float | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[NDArray[np.complex128], GNLSEReport]:
    """Propagate a scalar envelope (N,) or (1, N) over ``length`` [m].

    ``n_steps`` fixes equal steps (no error control); otherwise adaptive with ``tolerance``.
    """
    a = np.asarray(field, dtype=complex)
    shape = a.shape
    a = a.reshape(-1)
    if a.size != grid.n_samples:
        raise ValueError("field length does not match the grid")
    op = NonlinearOperator(
        grid,
        gamma,
        center_frequency,
        self_steepening=self_steepening,
        raman_fraction=raman_fraction,
        tau1=tau1,
        tau2=tau2,
    )
    d = dispersion_operator(grid.angular_frequency(), betas, alpha)
    spec = spectrum(a, grid)
    n0 = photon_number(spec, grid, center_frequency)
    e0 = float(np.sum(np.abs(spec) ** 2))
    steps = rejected = 0
    h_min, h_max = math.inf, 0.0
    if length <= 0.0 or gamma == 0.0 and alpha == 0.0 and not any(betas):
        pass
    elif n_steps is not None:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        h = length / n_steps
        for _ in range(n_steps):
            spec = _rk4ip_step(spec, h, d, op)
            steps += 1
            if check_cancelled is not None and steps % 64 == 0:
                check_cancelled()
        h_min = h_max = h
    else:
        z = 0.0
        h = initial_step or length / 100
        while z < length * (1 - 1e-12):
            h = min(h, length - z)
            coarse = _rk4ip_step(spec, h, d, op)
            fine = _rk4ip_step(_rk4ip_step(spec, h / 2, d, op), h / 2, d, op)
            delta = float(np.linalg.norm(fine - coarse) / max(np.linalg.norm(fine), 1e-300))
            if not np.isfinite(delta):
                raise NumericalStabilityError(
                    "GNLSE step produced non-finite values.",
                    hint="Reduce the tolerance or check the grid resolution.",
                )
            if delta <= 2 * tolerance:
                spec = (16 * fine - coarse) / 15
                z += h
                steps += 1
                h_min, h_max = min(h_min, h), max(h_max, h)
            else:
                rejected += 1
            factor = (tolerance / delta) ** 0.2 if delta > 0 else 2.0
            h *= min(2.0, max(0.5, factor))
            if steps + rejected > MAX_STEPS:
                raise NumericalStabilityError(
                    f"GNLSE needed more than {MAX_STEPS} steps.",
                    hint="Relax the tolerance or shorten the fiber.",
                )
            if check_cancelled is not None and (steps + rejected) % 16 == 0:
                check_cancelled()
    n1 = photon_number(spec, grid, center_frequency)
    e1 = float(np.sum(np.abs(spec) ** 2))
    out = inverse_spectrum(spec, grid).reshape(shape)
    report = GNLSEReport(
        steps,
        rejected,
        h_min if steps else 0.0,
        h_max,
        (n1 - n0) / n0 if n0 else 0.0,
        (e1 - e0) / e0 if e0 else 0.0,
    )
    return out, report


__all__ = ["GNLSEReport", "propagate_gnlse"]
