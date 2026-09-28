"""Timing of OptoBuild's computational hot paths (not part of the test suite).

Measures, on the current machine, the cost per unit of work of:
FFT (baseline), RK4IP GNLSE step, SSFM step, semiconductor-laser RK4 step
(pure-Python loop), 2x2 MIMO equalizer update (Python loop), one mode-locked
laser round trip, and a whole coherent-link run; plus the wall-clock speedup
of a parameter sweep with process workers. Prints a Markdown table.

Run:  python benchmarks/bench_hotpaths.py
"""

from __future__ import annotations

import math
import os
import platform
import time
from collections.abc import Callable

import numpy as np
import scipy

from optobuild.analysis.coherent_dsp import mimo_equalize
from optobuild.cli.demos import coherent_link_project, reference_project
from optobuild.numerics.fft import spectrum
from optobuild.numerics.grid import TimeGrid
from optobuild.persistence import run_project
from optobuild.physics.semiconductor_laser import LaserParameters
from optobuild.solvers.gnlse import propagate_gnlse
from optobuild.solvers.laser_dynamics import simulate
from optobuild.solvers.ssfm import propagate_nlse
from optobuild.sweeps.parameter_sweep import Axis, Probe, monte_carlo, sweep


def best_of(fn: Callable[[], object], repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> None:
    rows: list[tuple[str, str, float, str]] = []
    grid = TimeGrid(8192, 1e-15, -4096e-15)
    field = (1 / np.cosh(grid.time() / 30e-15)).astype(complex) * 100
    n_fft = 200
    t = best_of(lambda: [spectrum(field, grid) for _ in range(n_fft)])
    rows.append(("FFT (numerics.fft.spectrum)", "N = 8192", t / n_fft * 1e6, "us / transform"))

    def gnlse() -> None:
        propagate_gnlse(field, grid, 1e-3, 3.6e14, betas=[-11.8e-27], gamma=0.11, n_steps=50)

    rows.append(("GNLSE RK4IP step (Raman + shock)", "N = 8192", best_of(gnlse) / 50 * 1e3,
                 "ms / step"))  # fmt: skip

    def ssfm() -> None:
        propagate_nlse(field[None, :], grid, 1e-3, alpha=0.0, beta2=-11.8e-27, gamma=0.11,
                       n_steps=200)  # fmt: skip

    rows.append(("SSFM step (Kerr + beta2)", "N = 8192", best_of(ssfm) / 200 * 1e3, "ms / step"))
    laser = LaserParameters()
    current = np.full(4000, 0.06)
    dt = 1e-12
    steps = 4000 * math.ceil(dt / (0.25 * laser.photon_lifetime))
    t_laser = best_of(lambda: simulate(laser, current, dt))
    rows.append(("Laser rate equations, RK4 (Python loop)", f"{steps} steps",
                 t_laser / steps * 1e6, "us / step"))  # fmt: skip
    rng = np.random.default_rng(1)
    y = rng.standard_normal((2, 2 * 4096)) + 1j * rng.standard_normal((2, 2 * 4096))
    y /= np.sqrt(np.mean(np.abs(y) ** 2))
    updates = 4096 * 6
    t_mimo = best_of(lambda: mimo_equalize(y, "qpsk"), 1)
    rows.append(("2x2 MIMO CMA update (Python loop)", "15 taps", t_mimo / updates * 1e6,
                 "us / update"))  # fmt: skip
    from optobuild.cli.demos import mode_locked_laser_project

    p_ml = mode_locked_laser_project()
    t_ml = best_of(lambda: run_project(p_ml), 1)
    trips = run_project(p_ml).result("laser", "round_trips")
    rows.append(("Mode-locked laser round trip", "N = 1024, 20 fiber steps",
                 t_ml / trips * 1e3, "ms / round trip"))  # fmt: skip
    p_coh = coherent_link_project(prbs_order=15)
    rows.append(("Coherent QPSK link, full run", "32767 symbols x 4 sps",
                 best_of(lambda: run_project(p_coh), 1), "s / run"))  # fmt: skip
    # a realistic sweep: coherent link, 4 OSNR points x 2 noise trials (~1 s per run)
    axes = [Axis("ase", "osnr", tuple(10 ** (x / 10) for x in (10.0, 12.0, 14.0, 16.0)))]
    probes = [Probe("analyzer", "snr_db")]
    p_sw = coherent_link_project(prbs_order=15)
    serial = best_of(lambda: sweep(p_sw, axes, probes, trials=2), 1)
    workers = min(4, os.cpu_count() or 1)
    parallel = best_of(lambda: sweep(p_sw, axes, probes, trials=2, workers=workers), 1)
    rows.append(("Coherent sweep 4 OSNR x 2 trials, serial", "shared cache", serial, "s"))
    rows.append((f"  same with {workers} process workers", f"speedup {serial / parallel:.2f}x",
                 parallel, "s"))  # fmt: skip
    mc_s = best_of(lambda: monte_carlo(p_sw, probes, 16), 1)
    mc_p = best_of(lambda: monte_carlo(p_sw, probes, 16, workers=workers), 1)
    rows.append(("Coherent Monte Carlo 16 trials, serial", "shared cache", mc_s, "s"))
    rows.append((f"  same with {workers} process workers", f"speedup {mc_s / mc_p:.2f}x", mc_p,
                 "s"))  # fmt: skip
    tiny = [Axis("noise", "sigma", tuple(np.linspace(0.1, 1.0, 8)))]
    t_tiny_s = best_of(lambda: sweep(reference_project(), tiny, [Probe("noisy", "rms")],
                                     trials=400), 1)  # fmt: skip
    t_tiny_p = best_of(lambda: sweep(reference_project(), tiny, [Probe("noisy", "rms")],
                                     trials=400, workers=workers), 1)  # fmt: skip
    rows.append(("Tiny sweep (reference, 3200 runs), serial", "", t_tiny_s, "s"))
    rows.append((f"  same with {workers} process workers", f"speedup {t_tiny_s / t_tiny_p:.2f}x",
                 t_tiny_p, "s"))  # fmt: skip
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__},"
          f" {os.cpu_count()} CPUs, {platform.machine()}\n")  # fmt: skip
    print("| hot path | size | time | unit |\n|---|---|---|---|")
    for name, size, value, unit in rows:
        print(f"| {name} | {size} | {value:.3g} | {unit} |")


if __name__ == "__main__":
    main()
