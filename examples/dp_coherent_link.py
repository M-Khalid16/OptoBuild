"""Dual-polarization coherent link: DP-QPSK and DP-16QAM vs OSNR, with PMD (Phase 6b).

32 GBd per polarization, 80 km SMF (CD compensated in DSP), random PMD with
10 ps mean DGD (waveplate model, one realization per seed), ideal amplifier
setting the OSNR, 100 kHz TX and LO linewidths, 1 GHz LO offset, receiver
shot/thermal noise, 2x2 CMA (QPSK) or CMA+RDE (16-QAM) equalizer. For each
OSNR the measured pooled SNR, the per-polarization SNRs and the counted BER
and SER are printed next to the ASE-only limit SNR = 2 B_ref OSNR / (2 R_s)
and the exact AWGN SER at that limit. Differences are real impairments (receiver
noise, laser phase noise, modulator nonlinearity and carrier leakage,
equalizer and blind-phase-search penalties), not fitting. Everything printed
is computed by the simulator.

Run:  python examples/dp_coherent_link.py
"""

from __future__ import annotations

import math

from optobuild.analysis.constellations import theoretical_ser
from optobuild.cli.demos import dp_coherent_link_project
from optobuild.persistence import run_project
from optobuild.physics.noise import osnr_to_snr


def main() -> None:
    print(
        f"{'format':9s} {'OSNR':>5s} {'SNR':>6s} {'SNR x':>6s} {'SNR y':>6s} {'SNR ASE':>8s} "
        f"{'BER':>9s} {'SER':>9s} {'SER AWGN@ASE':>13s} {'DGD [ps]':>9s}"
    )
    for fmt, osnrs in (("qpsk", (13.0, 15.0)), ("16qam", (20.0, 22.0))):
        for osnr_db in osnrs:
            r = run_project(dp_coherent_link_project(modulation=fmt, osnr_db=osnr_db))
            snr = osnr_to_snr(10 ** (osnr_db / 10), 32e9, n_pol_signal=2)
            keys = ("snr_db", "snr_db_x", "snr_db_y", "ber", "ser")
            a = {k: r.result("analyzer", k) for k in keys}
            print(
                f"DP-{fmt:6s} {osnr_db:5.1f} {a['snr_db']:6.2f} {a['snr_db_x']:6.2f} "
                f"{a['snr_db_y']:6.2f} {10 * math.log10(snr):8.2f} {a['ber']:9.2e} {a['ser']:9.2e} "
                f"{float(theoretical_ser(fmt, snr)):13.2e} {r.result('pmd', 'dgd_s') * 1e12:9.2f}"
            )


if __name__ == "__main__":
    main()
