"""Numerical solution of the two-level EDFA equations along z (physics_models.md 3.26).

The coupled flux equations of ``physics.edfa`` (all beams forward) and the
co-polarized forward ASE density at each signal beam are integrated from
z = 0 to L with fixed-step RK4 (``solvers.ode``). The fiber ring laser
steady state is found by bisection on the intracavity power using this
amplifier solution (valid with background loss, where no closed form exists).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from optobuild.physics.edfa import Beam, ErbiumFiber, noise_figure
from optobuild.solvers.ode import integrate

DEFAULT_STEPS = 200
"""RK4 steps along the fiber (error << 1e-6 relative for gains of a few 10 dB; see tests)."""


@dataclass(frozen=True)
class AmplifierSolution:
    """Output powers [W], gains (linear), forward co-polarized ASE density [W/Hz] and NF
    (linear) of each beam; mean upper-level fraction along the fiber."""

    powers_out: tuple[float, ...]
    gains: tuple[float, ...]
    ase_density: tuple[float, ...]
    noise_figures: tuple[float, ...]
    mean_upper_fraction: float


def amplify(
    fiber: ErbiumFiber,
    beams: tuple[Beam, ...],
    powers_in: tuple[float, ...],
    n_steps: int = DEFAULT_STEPS,
) -> AmplifierSolution:
    """Integrate pump/signal fluxes and ASE along the doped fiber."""
    if len(beams) != len(powers_in) or not beams:
        raise ValueError("need one input power per beam")
    q0 = tuple(p / b.photon_energy for b, p in zip(beams, powers_in, strict=True))
    nb = len(beams)

    def f(z: float, y: tuple[float, ...]) -> tuple[float, ...]:
        q = y[:nb]
        n2 = fiber.upper_fraction(beams, q)
        dq = tuple(fiber.gain_coefficient(b, n2) * qk for b, qk in zip(beams, q, strict=True))
        ds = tuple(fiber.ase_derivative(b, n2, sk) for b, sk in zip(beams, y[nb : 2 * nb],
                                                                    strict=True))  # fmt: skip
        return (*dq, *ds, n2)

    y0 = (*q0, *(0.0,) * nb, 0.0)  # last variable integrates n2 dz
    end = integrate(f, y0, 0.0, fiber.length, 2, substeps=n_steps)[-1]
    q_out = end[:nb]
    ase = end[nb : 2 * nb]
    gains = tuple(qo / qi if qi > 0 else math.nan for qo, qi in zip(q_out, q0, strict=True))
    nfs = tuple(
        noise_figure(g, s, b.photon_energy) if g > 0 else math.nan
        for g, s, b in zip(gains, ase, beams, strict=True)
    )
    return AmplifierSolution(
        tuple(q * b.photon_energy for q, b in zip(q_out, beams, strict=True)),
        gains,
        tuple(ase),
        nfs,
        end[-1] / fiber.length,
    )


def ring_laser(
    fiber: ErbiumFiber,
    pump: Beam,
    signal: Beam,
    pump_power: float,
    output_coupling: float,
    passive_transmission: float,
    n_steps: int = DEFAULT_STEPS,
) -> tuple[float, float]:
    """(output power [W], intracavity power at the EDF input [W]); zeros below threshold.

    Solves ln G(P_c) + ln((1 - T) eta) = 0; G decreases monotonically with P_c.
    """
    loss = -math.log((1 - output_coupling) * passive_transmission)

    def excess(pc: float) -> float:
        sol = amplify(fiber, (pump, signal), (pump_power, pc), n_steps)
        return math.log(sol.gains[1]) - loss

    tiny = 1e-12
    if excess(tiny) <= 0:
        return 0.0, 0.0
    hi = max(pump_power, 1e-6)
    while excess(hi) > 0:
        hi *= 2
    pc = brentq(excess, tiny, hi, xtol=1e-15, rtol=1e-12)
    return output_coupling * pc / ((1 - output_coupling) * passive_transmission), pc


__all__ = ["AmplifierSolution", "amplify", "ring_laser"]
