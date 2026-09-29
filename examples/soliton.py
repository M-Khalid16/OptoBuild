"""Soliton propagation with the split-step Fourier solver (Phase 4).

Runs the built-in soliton demo for N = 1 (stationary) and N = 2 (period
(pi/2) L_D) and prints how much the output pulse deviates from the input.
Everything printed is computed by the simulator.

Run:  python examples/soliton.py
"""

from __future__ import annotations

import math

import numpy as np

from optobuild.cli.demos import soliton_project
from optobuild.persistence import run_project


def report(order: float, length_in_ld: float) -> None:
    res = run_project(soliton_project(soliton_order=order, length_in_ld=length_in_ld))
    p_in = res.signal("pulse", "out").power()
    p_out = res.signal("fiber", "out").power()
    dev = float(np.max(np.abs(p_out - p_in)) / p_in.max())
    print(
        f"N = {order:g}, L = {length_in_ld:.4g} L_D: peak {p_in.max():.4g} W -> "
        f"{p_out.max():.4g} W, max |P_out - P_in| / P_peak = {dev:.2e}, "
        f"{res.result('fiber', 'n_steps')} SSFM steps, "
        f"nonlinear phase bound {res.result('fiber', 'nonlinear_phase_bound_rad'):.3g} rad"
    )
    for d in res.all_diagnostics():
        print(f"  {d}")


def main() -> None:
    report(1.0, 5.0)  # fundamental soliton: shape preserved
    report(2.0, 0.5 * math.pi)  # second-order soliton: back to input after one period
    report(2.0, 0.25 * math.pi)  # ... and strongly compressed half-way


if __name__ == "__main__":
    main()
