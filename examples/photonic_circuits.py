"""Photonic-circuit wavelength sweeps with the S-matrix solver (Phase 7).

Builds circuits from elements (directional couplers, waveguides, Bragg
gratings) as netlists, solves them with ``solvers.circuit`` over a
wavelength sweep, and extracts resonances, FSR, loaded Q, extinction ratio,
insertion loss and group delay with ``analysis.resonances``. Each measured
figure is printed next to its closed form (docs/physics_models.md 3.22-3.24),
computed by the simulator from the same parameters. Also shows a two-ring
coupled-resonator filter, which has no simple closed form.

Run:  python examples/photonic_circuits.py
"""

from __future__ import annotations

import math

import numpy as np

from optobuild.analysis.resonances import find_resonances, free_spectral_range, group_delay
from optobuild.core.constants import SPEED_OF_LIGHT as C
from optobuild.physics.integrated_optics import (
    all_pass_fwhm_phase,
    bragg_bandwidth,
    bragg_coupling_square,
    bragg_grating,
    bragg_peak_reflectance,
    coupler_matrix,
    waveguide_transmission,
)
from optobuild.physics.multilayer import stack_response
from optobuild.solvers.circuit import Circuit, four_port_coupler, two_port

LAM0, NEFF, NG = 1550e-9, 2.4, 4.2
LOSS = 2.0 * math.log(10) / 10 * 100  # 2 dB/cm
NU = np.linspace(C / 1.57e-6, C / 1.53e-6, 40_001)  # 125 MHz steps, ~80 per ring FWHM


def wg(length: float, loss: float = LOSS) -> np.ndarray:
    return waveguide_transmission(NU, length, NEFF, NG, LAM0, loss)


def ring_circuit(radius: float, k_in: float, k_drop: float | None) -> Circuit:
    """All-pass (k_drop None) or add-drop ring netlist."""
    length = 2 * math.pi * radius
    cir = Circuit()
    cir.add(four_port_coupler("c1", coupler_matrix(k_in), NU.size))
    cir.expose("in", "c1.in1")
    cir.expose("through", "c1.out1")
    if k_drop is None:
        cir.add(two_port("ring", wg(length)))
        cir.connect("c1.out2", "ring.a")
        cir.connect("ring.b", "c1.in2")
        return cir
    cir.add(four_port_coupler("c2", coupler_matrix(k_drop), NU.size))
    cir.add(two_port("top", wg(length / 2)))
    cir.add(two_port("bottom", wg(length / 2)))
    cir.connect("c1.out2", "top.a")
    cir.connect("top.b", "c2.in2")
    cir.connect("c2.out2", "bottom.a")
    cir.connect("bottom.b", "c1.in2")
    cir.expose("drop", "c2.out1")
    return cir


def report_ring() -> None:
    radius, k2 = 10e-6, 0.05
    length = 2 * math.pi * radius
    s, names = ring_circuit(radius, k2, None).smatrix()
    h = s[:, names.index("through"), names.index("in")]
    res = find_resonances(NU, np.abs(h) ** 2, "dip")
    a = math.exp(-0.5 * LOSS * length)
    fwhm = all_pass_fwhm_phase(k2, a) * C / (2 * math.pi * NG * length)
    r = res[len(res) // 2]
    print("All-pass ring, R = 10 um, kappa^2 = 0.05, 2 dB/cm")
    print(f"  FSR      {free_spectral_range(res) / 1e9:10.2f} GHz   closed form "
          f"{C / (NG * length) / 1e9:10.2f} GHz")  # fmt: skip
    print(f"  FWHM     {r.fwhm / 1e9:10.3f} GHz   closed form {fwhm / 1e9:10.3f} GHz")
    print(f"  loaded Q {r.q_factor:10.0f}       closed form {r.position / fwhm:10.0f}")
    print(f"  extinction {r.extinction_ratio_db:8.2f} dB")
    tau = group_delay(NU, h)
    print(f"  group delay at resonance {tau[r.index] * 1e12:8.2f} ps "
          f"(waveguide round trip {NG * length / C * 1e12:.2f} ps)")  # fmt: skip


def report_add_drop() -> None:
    s, names = ring_circuit(10e-6, 0.1, 0.1).smatrix()
    drop = np.abs(s[:, names.index("drop"), names.index("in")]) ** 2
    res = find_resonances(NU, drop, "peak")
    r = res[len(res) // 2]
    print("Add-drop ring, kappa^2 = 0.1 / 0.1")
    print(f"  drop insertion loss at resonance {-10 * math.log10(r.extremum):6.3f} dB, "
          f"loaded Q {r.q_factor:8.0f}")  # fmt: skip


def report_coupled_rings() -> None:
    """Two identical rings in series between two buses (second-order filter)."""
    length = 2 * math.pi * 10e-6
    cir = Circuit()
    for k, name in enumerate(("c1", "c12", "c2")):
        cir.add(four_port_coupler(name, coupler_matrix((0.2, 0.02, 0.2)[k]), NU.size))
    for name in ("r1_top", "r1_bottom", "r2_top", "r2_bottom"):
        cir.add(two_port(name, wg(length / 2)))
    cir.connect("c1.out2", "r1_top.a")
    cir.connect("r1_top.b", "c12.in1")
    cir.connect("c12.out1", "r1_bottom.a")
    cir.connect("r1_bottom.b", "c1.in2")
    cir.connect("c12.out2", "r2_top.a")
    cir.connect("r2_top.b", "c2.in2")
    cir.connect("c2.out2", "r2_bottom.a")
    cir.connect("r2_bottom.b", "c12.in2")
    cir.expose("in", "c1.in1")
    cir.expose("through", "c1.out1")
    cir.expose("drop", "c2.out1")
    cir.expose("add", "c2.in1")
    s, names = cir.smatrix()
    drop = np.abs(s[:, names.index("drop"), names.index("in")]) ** 2
    thru = np.abs(s[:, names.index("through"), names.index("in")]) ** 2
    s1, n1 = ring_circuit(10e-6, 0.2, 0.2).smatrix()
    single_drop = np.abs(s1[:, n1.index("drop"), n1.index("in")]) ** 2
    i = int(np.argmax(drop))
    print("Two coupled rings (kappa^2 = 0.2 / 0.02 / 0.2), no closed form")
    print(f"  peak drop {drop[i]:.3f}, through at peak {thru[i]:.3e}, "
          f"energy check {drop[i] + thru[i]:.4f} (< 1: 2 dB/cm loss)")  # fmt: skip

    def width(p: np.ndarray, frac: float) -> float:
        k = int(np.argmax(p))
        above = p >= frac * p[k]
        lo = k
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = k
        while hi < p.size - 1 and above[hi + 1]:
            hi += 1
        return float(NU[hi] - NU[lo])

    ratio2 = width(drop, 0.1) / width(drop, 0.5)
    ratio1 = width(single_drop, 0.1) / width(single_drop, 0.5)
    print(f"  -10 dB / -3 dB width ratio {ratio2:.2f} (single ring {ratio1:.2f}; Lorentzian 3, "
          "maximally flat 2nd order 1.73)")  # fmt: skip


def report_bragg() -> None:
    navg, dn, n_periods = 2.4, 2e-3, 1500
    period = LAM0 / (2 * navg)
    kappa = bragg_coupling_square(dn, LAM0)
    lam = np.linspace(1.547e-6, 1.553e-6, 6001)
    r, _ = bragg_grating(C / lam, n_periods * period, period, kappa, navg, navg, LAM0)
    idx = [navg + dn / 2, navg - dn / 2] * n_periods
    _, _, big_r, _ = stack_response(lam, idx, [period / 2] * (2 * n_periods), navg, navg)
    print(f"Bragg grating, dn = {dn:g}, {n_periods} periods ({n_periods * period * 1e6:.0f} um)")
    print(f"  peak R: CMT {np.max(np.abs(r) ** 2):.5f}  TMM {big_r.max():.5f}  closed form "
          f"{bragg_peak_reflectance(kappa, n_periods * period):.5f}")  # fmt: skip
    print(f"  max |R_CMT - R_TMM| over the sweep {np.max(np.abs(np.abs(r) ** 2 - big_r)):.1e}")
    print(f"  bandwidth (first zeros) closed form "
          f"{bragg_bandwidth(LAM0, navg, kappa, n_periods * period) * 1e12:.1f} pm")  # fmt: skip


def main() -> None:
    report_ring()
    report_add_drop()
    report_coupled_rings()
    report_bragg()


if __name__ == "__main__":
    main()
