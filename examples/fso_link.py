"""Free-space optical link: budget, outage and Monte Carlo BER (Phase 5).

1. Analytic link budget of the demo FSO channel (1.5 km, haze V = 4 km,
   Gamma-Gamma turbulence Cn^2 = 5e-14 m^-2/3, pointing jitter, beam wander).
2. Channel-gain statistics of the simulated component over Monte Carlo
   trials compared with the analytic mean and outage probability.
3. BER of the 10 Gb/s NRZ-OOK link accumulated over the same trials.

Everything printed is computed by the simulator.
Run:  python examples/fso_link.py [--trials N]
"""

from __future__ import annotations

import argparse

import numpy as np

from optobuild.analysis.link_budget import fso_link_budget
from optobuild.cli.demos import fso_link_project
from optobuild.engine import FeedForwardExecutor, ResultCache
from optobuild.persistence import run_project


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FSO link example")
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args(argv)

    project = fso_link_project(prbs_order=11)
    fso = project.graph.node("fso")
    model = fso.model(1550e-9)
    tx_power = run_project(project).result("tx_power", "average_power_w")  # simulated
    sensitivity = 1e-6  # -30 dBm: illustrative threshold for the outage figure
    print(fso_link_budget(model, tx_power, sensitivity).table())

    ex = FeedForwardExecutor(ResultCache())
    gains, errors, bits = [], 0, 0
    for k in range(args.trials):
        res = run_project(project, executor=ex, trial=k)
        gains.append(res.result("fso", "channel_gain"))
        errors += res.result("ber", "n_errors")
        bits += res.result("ber", "n_bits")
    g = np.array(gains)
    th = 0.5 * model.mean_gain()
    print(f"\n{args.trials} channel realizations:")
    n = g.size
    p_sim = float((g < th).mean())
    print(
        f"  mean gain      simulated {g.mean():.4e} +- {g.std() / np.sqrt(n):.1e} (1 s.e.)"
        f"   analytic {model.mean_gain():.4e}"
    )
    print(
        f"  P(h < E[h]/2)  simulated {p_sim:.4f} +- {np.sqrt(p_sim * (1 - p_sim) / n):.4f}"
        f"   analytic {model.outage_probability(th):.4f}"
    )
    print(f"  BER over all trials: {errors} errors / {bits} bits = {errors / bits:.3e}")


if __name__ == "__main__":
    main()
