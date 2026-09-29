"""Time integration of the semiconductor-laser rate equations (physics_models.md 3.25).

Drift: ``LaserParameters.rhs`` integrated with fixed-step RK4
(``solvers.ode``); the injection current is linearly interpolated between
its samples. Noise (optional): Langevin forces of ``physics.semiconductor_laser``
added after each step as Gaussian increments with covariance D h (Ito,
Euler-Maruyama), for photon number n_p = S V / Gamma, carrier number
n_c = N V and phase. With D_pp = 2 R n_p, D_cc = 2 (R n_p + n_c/tau_n),
D_pc = -2 R n_p the Cholesky factorization is exact and simple:

    dn_p = sqrt(2 R n_p h) z1,   dn_c = -dn_p + sqrt(2 n_c h / tau_n) z2,
    dphi = sqrt(R h / (2 n_p)) z3.

Photon density is kept >= 0 after the stochastic increment (the diffusion
approximation is invalid for n_p of order 1, i.e. far below threshold).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.physics.semiconductor_laser import LaserParameters
from optobuild.solvers.ode import integrate

MAX_STEP_OVER_PHOTON_LIFETIME = 0.25
"""Default RK4 step bound h <= tau_p / 4 (|lambda| h <= 1 for the fastest photon mode)."""


def default_substeps(params: LaserParameters, dt: float) -> int:
    return max(1, math.ceil(dt / (MAX_STEP_OVER_PHOTON_LIFETIME * params.photon_lifetime)))


def simulate(
    params: LaserParameters,
    current: ArrayLike,
    dt: float,
    *,
    substeps: int | None = None,
    initial: tuple[Any, Any, Any] | None = None,
    rng: np.random.Generator | None = None,
    ensemble: int | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(N, S, phi) at the current samples' times t_k = k dt.

    ``initial`` defaults to the steady state at current[0]. ``rng`` enables Langevin
    noise; ``ensemble`` integrates that many independent realizations at once
    (outputs of shape (ensemble, n)).
    """
    i_samples = np.asarray(current, dtype=float)
    n_out = i_samples.size
    m = substeps or default_substeps(params, dt)
    if initial is None:
        n0, s0 = params.steady_state(float(i_samples[0]))
        initial = (n0, s0, 0.0)
    y0: tuple[Any, ...] = tuple(initial)
    if ensemble is not None:
        y0 = tuple(np.full(ensemble, float(v)) for v in y0)
    i_list = i_samples.tolist()
    last = n_out - 1

    def f(t: float, y: tuple[Any, ...]) -> tuple[Any, ...]:
        u = t / dt
        k = min(int(u), last)
        frac = u - k
        cur = i_list[k] if k == last else i_list[k] + frac * (i_list[k + 1] - i_list[k])
        return params.rhs(cur, y[0], y[1])

    noise = None
    if rng is not None:
        shape = (ensemble,) if ensemble is not None else ()
        v, gam, tn = params.volume, params.confinement, params.carrier_lifetime
        beta = params.spontaneous_coupling

        def noise(t: float, y: tuple[Any, ...], step: float) -> tuple[Any, ...]:
            n, s = y[0], np.maximum(y[1], 0.0)
            n_p = np.maximum(s * v / gam, 1e-30)
            n_c = np.maximum(n * v, 0.0)
            r = beta * n_c / tn
            z = rng.standard_normal((3, *shape))
            d_np = np.sqrt(2 * r * n_p * step) * z[0]
            d_nc = -d_np + np.sqrt(2 * n_c * step / tn) * z[1]
            d_phi = np.sqrt(r * step / (2 * n_p)) * z[2]
            ds = d_np * gam / v
            ds = np.maximum(s + ds, 0.0) - y[1]  # keep S >= 0
            return d_nc / v, ds, d_phi

    states = integrate(f, y0, 0.0, dt, n_out, m, noise, check_cancelled)
    out = [np.array([st[j] for st in states], dtype=float) for j in range(3)]
    if ensemble is not None:
        out = [o.T for o in out]  # (ensemble, n)
    return out[0], out[1], out[2]


__all__ = ["default_substeps", "simulate"]
