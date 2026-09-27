# ADR-0012: Stochastic components and Monte Carlo trials

* Status: Accepted
* Date: 2026-09-27
* Refines: ADR-0005 (cache keys), ADR-0008 (trial spawn keys)

## Context
BER estimates below ~1e-4 need many noise realizations. Re-running the whole
graph per realization wastes time on deterministic parts (PRBS, NRZ, MZM,
fiber), and if the seed/trial were in every cache key nothing could be reused.

## Decision
1. Components declare `stochastic: ClassVar[bool]` (default `False`).
   Only stochastic components may use `context.rng`; the engine raises if a
   non-stochastic component touches it (this protects cache correctness).
   Built-in stochastic components: `PINPhotodiode`, `GaussianNoise`.
2. Cache keys include the root seed and trial index **only for stochastic
   components**; downstream nodes inherit the dependence through their input
   keys. Deterministic sub-graphs are therefore reused across seeds and trials.
3. `FeedForwardExecutor.run(..., trial=k)` selects realization k: each
   stochastic component's generator is `(root seed, name, k)`. `trial=None`
   is the default realization `(root seed, name)`.
4. `sweeps.monte_carlo.monte_carlo_ber(project, n_trials)` sums `n_errors`
   and `n_bits` of a BER analyzer over trials and reports an exact
   Clopper-Pearson interval on the totals. CLI: `optobuild ber`.

## Alternatives considered
* *Detect randomness use at run time and key afterwards*: keys must be known
  before running to look up the cache.
* *Longer single runs (more bits)*: also valid; memory grows with N, and FFT
  cost grows as N log N; trials keep memory bounded.

## Consequences
* Changing the seed of a deterministic graph now reuses cached results
  (previously everything was recomputed); results are identical either way.
* A new stochastic component must set `stochastic = True`, or it fails loudly.
* All trials share the transmitted pattern; stated in the module docstring.
