"""Study workflows (Phase 10): sweep, Monte Carlo, optimization and a report.

1. Sweep the amplifier OSNR of the coherent QPSK link and compare the measured
   SNR with the ASE limit 2 B_ref OSNR / R_s (the gap grows with OSNR as receiver
   noise, laser phase noise and BPS matter more; at low OSNR it is within the
   estimator's statistical error).
2. Monte Carlo of the SNR at one OSNR over independent noise trials (mean,
   standard error).
3. Optimize the DSP's chromatic-dispersion compensation for maximum SNR; the
   optimum should sit at the fiber's accumulated dispersion D L.
4. Write an HTML report of the optimized link.

Everything printed is computed by the simulator. ``--quick`` uses a shorter
PRBS and fewer points (for tests).

Run:  python examples/workflows.py [--quick] [--report PATH]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from optobuild.cli.demos import coherent_link_project
from optobuild.optimization.optimizer import Objective, Variable, optimize
from optobuild.persistence import run_project
from optobuild.physics.noise import osnr_to_snr
from optobuild.reporting.html import run_report
from optobuild.sweeps.parameter_sweep import Axis, Probe, monte_carlo, sweep


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report", metavar="PATH", help="write the HTML report here")
    args = parser.parse_args(argv)
    order = 11 if args.quick else 15
    project = coherent_link_project(prbs_order=order)
    osnr_db = (10.0, 14.0) if args.quick else (10.0, 12.0, 14.0, 16.0, 18.0)

    print("1. OSNR sweep (coherent QPSK, 32 GBd, 80 km)")
    res = sweep(project, [Axis("ase", "osnr", tuple(10 ** (x / 10) for x in osnr_db))],
                [Probe("analyzer", "snr_db"), Probe("analyzer", "ber")])  # fmt: skip
    for x, snr, ber in zip(osnr_db, res.mean("analyzer.snr_db"), res.mean("analyzer.ber"),
                           strict=True):  # fmt: skip
        limit = 10 * math.log10(osnr_to_snr(10 ** (x / 10), 32e9))
        print(f"   OSNR {x:5.1f} dB: SNR {snr:6.2f} dB (ASE limit {limit:6.2f} dB), BER {ber:.2e}")

    print("2. Monte Carlo at OSNR 12 dB")
    project.graph.set_parameters("ase", osnr=10**1.2)
    n = 4 if args.quick else 20
    est = monte_carlo(project, [Probe("analyzer", "snr_db")], n)["analyzer.snr_db"]
    lo, hi = est.interval()
    print(f"   SNR {est.mean:.3f} dB +- {est.standard_error:.3f} dB (SE), "
          f"95 % interval [{lo:.3f}, {hi:.3f}] dB over {n} trials")  # fmt: skip

    print("3. Optimize the CD compensation for maximum SNR")
    dl = 17e-6 * 80e3  # accumulated dispersion D L [s/m]
    opt = optimize(project, [Variable("dsp", "cd_compensation", 0.8 * dl, 1.2 * dl, 0.9 * dl)],
                   Objective("analyzer", "snr_db", "max"), xatol=1e-4 * dl)  # fmt: skip
    best = opt.x["dsp.cd_compensation"]
    print(f"   optimum {best * 1e3:.2f} ps/nm (D L = {dl * 1e3:.2f} ps/nm), "
          f"SNR {opt.value:.2f} dB after {opt.n_evaluations} evaluations")  # fmt: skip

    if args.report:
        result = run_project(opt.project)
        Path(args.report).write_text(run_report(opt.project, result), encoding="utf-8")
        print(f"4. Report written to {args.report}")


if __name__ == "__main__":
    main()
