"""Single-polarization coherent link: QPSK and 16-QAM vs OSNR (Phase 6).

32 GBd, 80 km SMF with DSP chromatic-dispersion compensation, an ideal
amplifier compensating the span loss and setting the OSNR, 100 kHz TX and LO
linewidths, 1 GHz LO offset, receiver shot/thermal noise. For each OSNR
the measured (data-aided) SNR is printed next to the ASE-only limit
SNR = 2 B_ref OSNR / R_s, and the counted error rates next to the exact AWGN
values at that limit. Differences are real impairments (receiver noise,
laser phase noise, IQ-modulator sine nonlinearity and carrier leakage at the
40 dB child extinction ratio, blind phase search),
not fitting. Everything printed is computed by the simulator.

Run:  python examples/coherent_link.py
"""

from __future__ import annotations

import math

from optobuild.analysis.constellations import theoretical_ber, theoretical_ser
from optobuild.cli.demos import coherent_link_project
from optobuild.persistence import run_project
from optobuild.physics.noise import osnr_to_snr


def main() -> None:
    print(
        f"{'format':7s} {'OSNR':>6s} {'SNR meas':>9s} {'SNR ASE':>8s} "
        f"{'BER meas':>9s} {'SER meas':>9s} {'SER AWGN@ASE-limit':>19s}"
    )
    for fmt, osnrs in (("qpsk", (10.0, 12.0, 14.0)), ("16qam", (18.0, 20.0, 22.0))):
        for osnr_db in osnrs:
            r = run_project(coherent_link_project(modulation=fmt, osnr_db=osnr_db))
            snr = osnr_to_snr(10 ** (osnr_db / 10), 32e9)
            snr_meas, ber, ser = (r.result("analyzer", k) for k in ("snr_db", "ber", "ser"))
            print(
                f"{fmt:7s} {osnr_db:6.1f} {snr_meas:9.2f} {10 * math.log10(snr):8.2f} "
                f"{ber:9.2e} {ser:9.2e} {float(theoretical_ser(fmt, snr)):19.2e}"
            )
    print(
        "\nQPSK BER at the ASE limit for comparison:",
        ", ".join(
            f"{o:g} dB: {float(theoretical_ber('qpsk', osnr_to_snr(10 ** (o / 10), 32e9))):.1e}"
            for o in (10.0, 12.0, 14.0)
        ),
    )


if __name__ == "__main__":
    main()
