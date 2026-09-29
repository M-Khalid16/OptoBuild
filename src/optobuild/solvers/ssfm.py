"""Symmetric split-step Fourier method for the scalar NLSE (ADR-0014).

One step of length h (Strang splitting, global error O(h^2)):

    A <- D(h/2) A          D: exact linear dispersion, frequency domain
    A <- N(h)   A          N: exact Kerr + loss step (physics.nonlinear)
    A <- D(h/2) A

with D(h) = exp(-i (beta2/2 w^2 + beta3/6 w^3) h) (physics.fiber.fiber_transfer_function
with alpha = 0; loss is inside N).

Step-size control (``max_phase``): the step is chosen so that the peak
nonlinear phase per step, gamma * P_peak * L_eff(h), does not exceed
``max_phase`` [rad] (Sinkin et al., J. Lightwave Technol. 21, 61 (2003),
"nonlinear phase rotation" method), and never exceeds ``max_step`` [m]. P_peak
is re-evaluated at every step. With gamma = 0 the method is exact for any
number of steps.

Numerical limitations (reported by the component as diagnostics): splitting
error O(h^2); periodic window (energy leaving one edge re-enters the other);
spectral broadening beyond +-fs/2 aliases.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import NumericalStabilityError
from optobuild.numerics.fft import apply_transfer_function
from optobuild.numerics.grid import TimeGrid
from optobuild.physics.fiber import fiber_transfer_function
from optobuild.physics.nonlinear import effective_length, kerr_loss_step

MAX_STEPS = 1_000_000


@dataclass(frozen=True)
class SSFMReport:
    """Statistics of one propagation."""

    n_steps: int
    max_step_phase: float
    """Largest nonlinear phase of a single step [rad]."""
    total_peak_phase: float
    """Sum over steps of the per-step peak nonlinear phase [rad] (upper bound of phi_NL)."""
    min_step: float
    max_step: float


def _step_for_phase(
    gamma: float, peak: float, alpha: float, max_phase: float, limit: float
) -> float:
    """Largest h <= limit with gamma * peak * L_eff(h) <= max_phase."""
    if gamma * peak == 0.0:
        return limit
    target = max_phase / (gamma * peak)  # required L_eff
    if alpha == 0.0:
        return min(limit, target)
    if alpha * target >= 1.0:
        return limit  # L_eff can never reach target: any step is fine
    return min(limit, -math.log1p(-alpha * target) / alpha)


def propagate_nlse(
    field: ArrayLike,
    grid: TimeGrid,
    length: float,
    *,
    alpha: float,
    beta2: float,
    beta3: float = 0.0,
    gamma: float,
    max_phase: float = 0.005,
    max_step: float = math.inf,
    n_steps: int | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[NDArray[np.complex128], SSFMReport]:
    """Propagate a (n_pol, N) envelope over ``length`` [m].

    Either ``n_steps`` (uniform steps) or adaptive steps limited by
    ``max_phase`` [rad] and ``max_step`` [m].
    """
    a = np.array(field, dtype=complex, copy=True)
    if n_steps is not None and n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if length < 0:
        raise ValueError(f"length must be >= 0 m, got {length}.")
    if length == 0.0:
        return a, SSFMReport(0, 0.0, 0.0, 0.0, 0.0)
    w = grid.angular_frequency()
    half_cache: dict[float, NDArray[np.complex128]] = {}

    def disperse(x: NDArray[np.complex128], h: float) -> NDArray[np.complex128]:
        if beta2 == 0.0 and beta3 == 0.0:
            return x
        if h not in half_cache:
            if len(half_cache) > 8:
                half_cache.clear()
            half_cache[h] = fiber_transfer_function(w, h, 0.0, beta2, beta3)
        return apply_transfer_function(x, half_cache[h], grid)

    z, steps, max_phi, total_phi = 0.0, 0, 0.0, 0.0
    hmin, hmax = math.inf, 0.0
    uniform = None if n_steps is None else length / n_steps
    while z < length * (1 - 1e-12):
        remaining = length - z
        peak = float(np.max(np.sum(np.abs(a) ** 2, axis=0)))
        if uniform is not None:
            h = min(uniform, remaining)
        else:
            h = _step_for_phase(gamma, peak, alpha, max_phase, min(max_step, remaining))
            if remaining - h < 1e-9 * length:
                h = remaining
        phi = gamma * peak * effective_length(h, alpha)
        a = disperse(a, 0.5 * h)
        a = kerr_loss_step(a, gamma, alpha, h)
        a = disperse(a, 0.5 * h)
        if not np.all(np.isfinite(a)):
            raise NumericalStabilityError(
                f"Non-finite field after {steps + 1} SSFM steps (z = {z + h:.4g} m).",
                hint="Reduce max_phase / the step size, or check the input power.",
            )
        z += h
        steps += 1
        max_phi = max(max_phi, phi)
        total_phi += phi
        hmin, hmax = min(hmin, h), max(hmax, h)
        if steps > MAX_STEPS:
            raise NumericalStabilityError(
                f"SSFM needed more than {MAX_STEPS} steps.",
                hint="Increase max_phase or check gamma and the peak power.",
            )
        if check_cancelled is not None and steps % 64 == 0:
            check_cancelled()
    return a, SSFMReport(steps, max_phi, total_phi, hmin, hmax)


__all__ = ["SSFMReport", "propagate_nlse"]
