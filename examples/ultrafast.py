"""Ultrafast optics (Phase 9): supercontinuum generation and a mode-locked fiber laser.

1. The supercontinuum benchmark of Dudley, Genty & Coen (Rev. Mod. Phys. 78,
   1135 (2006), Fig. 3): 50 fs, 10 kW sech pulse at 835 nm in 15 cm of PCF with
   dispersion to beta10, Raman and self-steepening. Printed: spectral extent at
   -20/-40 dB, energy and photon-number change (Raman red-shift loses energy
   but conserves photons), solver steps.
2. A passively mode-locked soliton fiber ring laser iterated from a weak seed to
   its steady state: round trips, pulse FWHM, time-bandwidth product (sech^2
   limit 0.315), autocorrelation-deconvolved width, and the soliton number of
   the pulse arriving at the output coupler.

Everything printed is computed by the simulator. ``--quick`` shortens the PCF
to 3 cm (for tests).

Run:  python examples/ultrafast.py [--quick]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from optobuild.cli.demos import mode_locked_laser_project, supercontinuum_project
from optobuild.persistence import run_project


def supercontinuum(length_m: float) -> None:
    r = run_project(supercontinuum_project(length_m=length_m))
    fiber = r.nodes["pcf"].results
    spec = r.nodes["output_spectrum"].results
    lam = np.asarray(spec["wavelength_m"])
    p = np.asarray(spec["power_per_rbw_w"])
    print(f"Supercontinuum, {length_m * 100:g} cm PCF (Dudley et al. 2006 parameters)")
    for db in (20, 40):
        m = p > p.max() * 10 ** (-db / 10)
        lo, hi = lam[m].min() * 1e9, lam[m].max() * 1e9
        print(f"  -{db} dB spectral extent: {lo:7.1f} - {hi:7.1f} nm")
    print(f"  energy change {fiber['energy_change'] * 100:+.2f} %, photon-number change "
          f"{fiber['photon_number_change']:+.1e} ({fiber['n_steps']} steps)")  # fmt: skip
    for d in r.all_diagnostics():
        print(f"  diagnostic: {d}")


def mode_locked_laser() -> None:
    r = run_project(mode_locked_laser_project())
    res = r.nodes["laser"].results
    print("\nPassively mode-locked soliton fiber laser")
    print(f"  converged after {res['round_trips']} round trips: {res['converged']}")
    print(f"  pulse FWHM {res['pulse_fwhm_s'] * 1e15:.1f} fs, energy "
          f"{res['pulse_energy_j'] * 1e12:.1f} pJ, peak {res['peak_power_w']:.1f} W")  # fmt: skip
    print(f"  time-bandwidth product {res['time_bandwidth_product']:.3f} (sech^2 limit 0.315)")
    ac = r.result("autocorrelator", "autocorrelation_fwhm_s")
    deconvolved = r.result("autocorrelator", "deconvolved_fwhm_s")
    print(f"  autocorrelation FWHM {ac * 1e15:.1f} fs -> deconvolved (sech^2) "
          f"{deconvolved * 1e15:.1f} fs")  # fmt: skip
    t0 = res["pulse_fwhm_s"] / (2 * math.log(1 + math.sqrt(2)))
    p_before_coupler = res["peak_power_w"] / 0.3  # 30 % output coupling
    n = math.sqrt(1.3e-3 * p_before_coupler * t0**2 / 22e-27)
    print(
        f"  soliton number at the output coupler (gamma P0 T0^2/|beta2|)^1/2 = {n:.2f} "
        "(lumped gain and loss: the average soliton varies around the cavity)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="3 cm PCF instead of 15 cm")
    args = parser.parse_args(argv)
    supercontinuum(0.03 if args.quick else 0.15)
    mode_locked_laser()


if __name__ == "__main__":
    main()
