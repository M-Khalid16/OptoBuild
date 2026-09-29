"""Laser and amplifier models (Phase 8): semiconductor laser, EDFA, fiber ring laser, DML link.

Prints, from the simulator: the semiconductor-laser L-I curve (steady states of
the full rate equations) next to the ideal closed form, the relaxation
oscillation and Henry linewidth vs bias, the small-signal RIN peak; EDFA gain
and noise figure vs pump power; the erbium fiber ring laser output vs pump
(numerical z integration vs the closed form) with threshold and slope; and the
directly modulated 10 Gb/s link Q factor vs fiber length (chirp x dispersion;
the decision-directed Q of a 511-bit pattern, not necessarily monotonic in length).

Run:  python examples/lasers.py
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from optobuild.cli.demos import dml_link_project
from optobuild.persistence import run_project
from optobuild.physics.edfa import (
    Beam,
    ErbiumFiber,
    ring_laser_closed_form,
    ring_laser_slope_efficiency,
    ring_laser_threshold,
)
from optobuild.physics.semiconductor_laser import LaserParameters
from optobuild.solvers.edfa import amplify, ring_laser


def semiconductor_laser() -> None:
    p = LaserParameters()
    ideal = dataclasses.replace(p, spontaneous_coupling=0.0, gain_compression=0.0)
    print(f"Semiconductor laser: I_th = {p.threshold_current * 1e3:.2f} mA, "
          f"slope {p.slope_efficiency:.3f} W/A")  # fmt: skip
    print(f"{'I [mA]':>7s} {'P [mW]':>8s} {'P ideal':>8s} {'f_R [GHz]':>9s} "
          f"{'linewidth [MHz]':>15s} {'RIN peak [dB/Hz]':>16s}")  # fmt: skip
    f = np.linspace(0.1e9, 20e9, 400)
    for i in (0.045, 0.06, 0.08, 0.1):
        n, s = p.steady_state(i)
        w_r, _ = ideal.relaxation_oscillation(i)
        rin, _ = p.noise_spectra(i, f)
        print(f"{i * 1e3:7.1f} {p.output_power(s) * 1e3:8.3f} "
              f"{ideal.output_power(ideal.steady_state(i)[1]) * 1e3:8.3f} "
              f"{w_r / (2 * math.pi) / 1e9:9.2f} {p.henry_linewidth(i) / 1e6:15.2f} "
              f"{10 * math.log10(rin.max()):16.1f}")  # fmt: skip


def edfa() -> None:
    fiber = ErbiumFiber()
    pump = Beam(980e-9, 2.2e-25, 0.0, 0.6)
    signal = Beam(1550e-9, 2.6e-25, 3.4e-25, 0.4)
    print("\nEDFA (10 m, n_t = 5e24 m^-3), signal -20 dBm at 1550 nm")
    print(f"{'pump [mW]':>9s} {'gain [dB]':>9s} {'NF [dB]':>8s} {'<n2>':>6s}")
    for pp in (0.01, 0.02, 0.05, 0.1, 0.2):
        sol = amplify(fiber, (pump, signal), (pp, 1e-5))
        g_db, nf_db = (10 * math.log10(v) for v in (sol.gains[1], sol.noise_figures[1]))
        print(f"{pp * 1e3:9.0f} {g_db:9.2f} {nf_db:8.2f} {sol.mean_upper_fraction:6.3f}")
    t, eta = 0.5, 0.8
    print(f"\nErbium ring laser (T = {t}, passive transmission {eta}): threshold "
          f"{ring_laser_threshold(fiber, pump, signal, t, eta) * 1e3:.2f} mW, slope "
          f"{ring_laser_slope_efficiency(fiber, pump, signal, t, eta):.3f} W/W")  # fmt: skip
    for pp in (0.01, 0.05, 0.1):
        numeric = ring_laser(fiber, pump, signal, pp, t, eta)[0]
        closed = ring_laser_closed_form(fiber, pump, signal, pp, t, eta)
        print(f"  pump {pp * 1e3:5.0f} mW: output {numeric * 1e3:8.4f} mW "
              f"(closed form {closed * 1e3:8.4f} mW)")  # fmt: skip


def dml_link() -> None:
    print("\n10 Gb/s DML link (bias 45 mA, 1.5 V drive, EDFA preamp): Q vs fiber length")
    for length in (0.0, 10e3, 20e3, 40e3):
        r = run_project(dml_link_project(fiber_length_m=length))
        q = r.result("eye", "q_factor_decision_directed")
        print(f"  {length / 1e3:4.0f} km: Q = {q:6.2f}, "
              f"BER = {r.result('ber', 'ber'):.2e} ({r.result('ber', 'n_bits')} bits), "
              f"chirp p-p {r.result('dml', 'chirp_peak_to_peak_hz') / 1e9:.1f} GHz")  # fmt: skip


def main() -> None:
    semiconductor_laser()
    edfa()
    dml_link()


if __name__ == "__main__":
    main()
