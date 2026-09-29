"""End-to-end 10 Gb/s NRZ-OOK reference link (Phase 2).

    PRBS11 -> NRZ (+-V_pi/2, 30 ps rise) -> MZM (quadrature, 5 dB IL, 30 dB ER) <- CW laser 0 dBm
    MZM -> 50 km SMF (0.2 dB/km, 17 ps/(nm km)) -> PIN (0.8 A/W) -> 4th-order Bessel 7.5 GHz
        -> decision -> BER analyzer
    taps: TX/RX optical power, TX optical spectrum (12.5 GHz RBW), eye diagram

Everything printed or saved here is computed by the simulator. The project is
saved as JSON next to this script, and the eye/spectrum data as .npz so they
can be plotted with any tool.

Run:  python examples/optical_link.py [--seed N] [--length-km L]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optobuild.analysis.ber import ber_from_q, q_factor
from optobuild.cli.demos import optical_link_project
from optobuild.persistence import load_project, run_project, save_project

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--length-km", type=float, default=50.0)
    parser.add_argument("--no-save", action="store_true", help="do not write output files")
    args = parser.parse_args(argv)

    project = optical_link_project(seed=args.seed, fiber_length_m=args.length_km * 1e3)
    if not args.no_save:
        project = load_project(save_project(project, HERE / "optical_link.json"))
    res = run_project(project)

    r = res.result
    print(
        f"seed {res.seed}, {r('ber', 'n_bits')} bits of PRBS11 at 10 Gb/s, "
        f"fiber {args.length_km:g} km"
    )
    print(f"TX power           {r('tx_power', 'average_power_dbm'):8.3f} dBm")
    print(f"RX power           {r('rx_power', 'average_power_dbm'):8.3f} dBm")
    print(f"fiber beta2        {r('fiber', 'beta2_s2_per_m') * 1e27:8.3f} ps^2/km")
    print(f"dispersive spread  {r('fiber', 'dispersive_spread_s') * 1e12:8.1f} ps (99% band)")
    print(f"filter NEB         {r('filter', 'noise_equivalent_bandwidth_hz') / 1e9:8.3f} GHz")
    print(
        f"counted errors     {r('ber', 'n_errors')} / {r('ber', 'n_bits')}  "
        f"(BER {r('ber', 'ber'):.3g}, 95% CI [{r('ber', 'ber_lower_95'):.2e}, "
        f"{r('ber', 'ber_upper_95'):.2e}])"
    )

    # Gaussian estimate from the decision samples grouped by the *transmitted* bits
    shift = r("ber", "alignment_shift_bits")
    ref = np.roll(res.signal("prbs", "out").bits, shift)
    mu1, mu0, s1, s0, q = q_factor(r("decision", "decision_samples"), ref)
    print(f"Q (data-aided)     {q:8.3f}  -> Gaussian BER estimate {ber_from_q(q):.2e}")
    print("   (assumes Gaussian noise and two levels; ISI makes it approximate)")

    diags = res.all_diagnostics()
    if diags:
        print("diagnostics:")
        for d in diags:
            print(f"  {d}")

    if not args.no_save:
        out = HERE / "optical_link_results.npz"
        np.savez_compressed(
            out,
            eye_traces_a=r("eye", "traces"),
            eye_time_s=r("eye", "time_s"),
            spectrum_frequency_hz=r("tx_spectrum", "frequency_hz"),
            spectrum_wavelength_m=r("tx_spectrum", "wavelength_m"),
            spectrum_power_per_rbw_w=r("tx_spectrum", "power_per_rbw_w"),
            spectrum_rbw_hz=r("tx_spectrum", "resolution_bandwidth_hz"),
        )
        print(f"saved project to {HERE / 'optical_link.json'} and data to {out}")


if __name__ == "__main__":
    main()
