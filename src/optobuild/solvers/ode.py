"""Fixed-step explicit integrators for small ODE/SDE systems (ADR-0019).

State: a tuple of floats or equally shaped NumPy arrays (an ensemble of
independent realizations integrates in one call). The right-hand side
``f(t, y) -> dy/dt`` uses only arithmetic, so it runs on floats (fast for one
realization in pure Python) and arrays alike.

* ``rk4_step``: classical 4th-order Runge-Kutta, local error O(h^5), global
  O(h^4) (Hairer, Norsett & Wanner, *Solving ODEs I*, sec. II.1).
* ``integrate``: fixed substeps between output times; optional additive
  stochastic increment ``noise(t, y, h) -> dy`` applied after each drift step
  (Lie splitting; with Ito-evaluated Gaussian increments this is the
  Euler-Maruyama scheme for the noise part, strong order 1/2).

Fixed steps keep the output on the simulation time grid and make runs
reproducible; the caller chooses h well below the fastest time constant
(stability of RK4 for real negative eigenvalues: |lambda| h < 2.78).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

State = tuple[Any, ...]
Rhs = Callable[[float, State], State]
Noise = Callable[[float, State, float], State]


def rk4_step(f: Rhs, t: float, y: State, h: float) -> State:
    """One classical RK4 step of size h."""
    k1 = f(t, y)
    k2 = f(t + h / 2, tuple(a + h / 2 * b for a, b in zip(y, k1, strict=True)))
    k3 = f(t + h / 2, tuple(a + h / 2 * b for a, b in zip(y, k2, strict=True)))
    k4 = f(t + h, tuple(a + h * b for a, b in zip(y, k3, strict=True)))
    return tuple(
        a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4, strict=True)
    )


def integrate(
    f: Rhs,
    y0: State,
    t0: float,
    dt: float,
    n_out: int,
    substeps: int = 1,
    noise: Noise | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> list[State]:
    """States at t0 + k dt, k = 0 .. n_out - 1 (y0 first), with ``substeps`` RK4 steps each."""
    if substeps < 1 or n_out < 1 or dt <= 0:
        raise ValueError("need substeps >= 1, n_out >= 1 and dt > 0")
    h = dt / substeps
    out = [y0]
    y, t = y0, t0
    for k in range(1, n_out):
        for _ in range(substeps):
            y = rk4_step(f, t, y, h)
            if noise is not None:
                y = tuple(a + b for a, b in zip(y, noise(t, y, h), strict=True))
            t += h
        t = t0 + k * dt  # no drift of the time axis by round-off
        out.append(y)
        if check_cancelled is not None and k % 4096 == 0:
            check_cancelled()
    return out


def columns(states: Sequence[State]) -> list[list[Any]]:
    """Transpose a list of state tuples into one list per state variable."""
    return [list(c) for c in zip(*states, strict=True)]


__all__ = ["columns", "integrate", "rk4_step"]
