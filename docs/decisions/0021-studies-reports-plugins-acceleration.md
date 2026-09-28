# ADR-0021: Studies (sweeps, Monte Carlo, optimization), reports, plugins and acceleration

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 10 turns single runs into studies: parameter sweeps, Monte Carlo
statistics, optimization of design parameters, shareable reports, third-party
components, and faster execution. The core rules still apply: no fabricated
results, reproducibility, no pickle for user data, no native dependencies
without a benchmark.

## Decision
1. **Sweeps and Monte Carlo** (`sweeps.parameter_sweep`) run a copy of the
   project over a Cartesian grid of component parameters × trial indices and
   collect scalar results. Trial seeding (root seed, component name, trial)
   makes points use common random numbers and makes parallel and serial runs
   bit-identical.
2. **Parallelism by processes** (`workers > 1`, "spawn" start method): the
   project travels as its JSON text, workers return only the probed scalars
   (plain floats). Grid points, and trials when points are fewer than
   workers, are the tasks. Serial runs share one result cache.
3. **Optimization** (`optimization.optimizer`): bounded Brent for one
   variable, Nelder–Mead (bounded) or Powell otherwise (SciPy);
   objective averaged over fixed trials (common random numbers, deterministic
   objective); history kept for reports. Local optimizers only.
4. **Reports** (`reporting`): self-contained HTML with inline SVG rendered
   without plotting libraries; all text escaped, no scripts, no external
   resources; the project file embedded for provenance; optional timestamp
   (off = byte-reproducible). Curve builders moved from `gui.plotdata` to
   `reporting.figures` so reports and the GUI draw the same data (the GUI
   module re-exports them).
5. **Plugins** (`plugins.discovery`): entry-point group
   `optobuild.components`; loading is explicit (`--plugins`, `plugin_registry`),
   failures are reported per plugin and a plugin can never replace an
   existing `type_id`. Project files only reference `type_id` strings, never
   code.
6. **Acceleration**, decided from `benchmarks/bench_hotpaths.py`
   (results in `benchmarks/RESULTS.md`):
   * the FFT time-origin phase factor was recomputed on every transform (4× the
     FFT cost); it is now memoized per grid → GNLSE step 8.1 → 2.8 ms,
     mode-locked round trip 11.8 → 5.9 ms;
   * process parallelism pays off only when per-task work dominates the
     ~1 s worker start-up and the serial path's cache reuse is small: 1.4× for
     a 16-trial coherent Monte Carlo on 4 CPUs, none for small sweeps;
   * **Numba is not adopted.** The remaining pure-Python loops (laser RK4
     ~7 µs/step, MIMO update ~10 µs/update) dominate only the dual-polarization
     DSP (~2–4 s per run) and long laser transients; they are the candidates
     if a workload becomes dominated by them (re-benchmark first).

## Consequences
* Validated: exact sweep values and noise scaling on the reference pipeline;
  Monte Carlo mean/std vs the noise model (5σ); parallel = serial bitwise;
  optimizer vs analytic least-squares optima (including an active bound)
  and ring tuning to resonance; report escaping and self-containment; plugin
  loading and failure isolation.
* A sweep uncovered a simulation artefact (one corrupted symbol at the edge of
  the periodic window with non-periodic laser phase noise); the coherent
  analyzer gained `guard_symbols`, used by the demos.
* Limitations: local optimization only (no global/Bayesian methods), no
  adaptive stopping of Monte Carlo on a target confidence, parallel workers
  need built-in components (plugins are not re-loaded in workers), reports
  are HTML only (no PDF).
