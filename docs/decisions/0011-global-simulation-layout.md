# ADR-0011: Global simulation layout

* Status: Accepted
* Date: 2026-09-27
* Supersedes: decision 1 of ADR-0010 (per-source grids only). Decisions 2 and
  3 of ADR-0010 (bit-timing metadata, explicit reference connection) remain.

## Context
With per-source grids (ADR-0010) the CW laser's `n_samples`/`sample_rate`
had to be kept consistent with the PRBS length, bit rate and samples per bit
by hand. Link simulators solve this with global "layout" parameters.

## Decision
1. `numerics.layout.SimulationLayout(bit_rate, n_bits, samples_per_bit)`
   defines the shared grid: `fs = R_b · sps`, `N = n_bits · sps`, `t0 = 0`.
2. A project may carry one layout (`simulation.layout` in schema v2); the
   executor receives it (`run(..., layout=...)`, or `run_project(project)`)
   and exposes it as `RunContext.layout`. The layout is part of every cache key.
3. Sources opt in per instance with the shared parameter
   `timing_source ∈ {"parameters", "layout"}`, default `"parameters"` so
   existing projects behave identically. In layout mode:
   * PRBS: `n_bits`, `bit_rate` from the layout (own values ignored);
   * NRZ: `samples_per_bit` from the layout; the incoming sequence must match
     the layout's bit count and rate (`SamplingError` otherwise);
   * CW laser: grid from the layout.
   A layout-mode source without a layout raises `SamplingError` with a hint.
4. Project schema v2: `simulation.layout` added; the GUI-only top-level key
   `layout` of v1 is renamed `schematic`. v1 files are migrated on load by a
   pure function (`persistence.project.MIGRATIONS`).

## Alternatives considered
* *Parameter binding/expressions (`n_samples = layout.n_bits * ...`)*:
  more general, but requires an expression language (and must never use
  `eval`); deferred until sweeps need it.
* *Making layout mode the default*: would silently change the meaning of
  existing projects.

## Consequences
* One place changes bit rate / pattern length / oversampling for a whole link.
* Components still validate grid compatibility, so mixed setups fail loudly.
* Source component versions bumped to 1.1.0 (new parameter; numerical output
  unchanged in `"parameters"` mode).
