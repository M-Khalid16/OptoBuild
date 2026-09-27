# ADR-0008: Reproducible randomness

* Status: Accepted
* Date: 2026-09-27

## Context
Noise and random patterns must be reproducible (FR-010, NFR-06), and adding,
removing or reordering unrelated components should not change the random
stream of an existing component.

## Decision
* A project has one integer **root seed**.
* Each component instance gets its own `numpy.random.Generator(PCG64)` built
  from `SeedSequence(entropy=root_seed, spawn_key=(stable_id(name),))`, where
  `stable_id` is derived from a cryptographic hash (SHA-256) of the instance
  name — not Python's salted `hash()`.
* For Monte Carlo trial `k`, the spawn key is extended with `k`.
* Global `numpy.random` state is never used by library code.
* Deterministic sources (PRBS) do not consume randomness; their pattern is
  set by parameters (order, initial state).

## Alternatives considered
* *Single global generator consumed in execution order*: streams change when
  the graph changes or when execution is parallelized.
* *Per-component user-specified seeds only*: tedious; still allowed as an
  optional override parameter.

## Consequences
* Renaming a component changes its noise realization (documented).
* Results are reproducible given the same NumPy version (bit-exactness across
  NumPy versions is not guaranteed by NumPy for all distributions; recorded
  as a known limitation).
