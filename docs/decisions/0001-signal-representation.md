# ADR-0001: Signal representation

* Status: Accepted
* Date: 2026-09-27

## Context
Every component exchanges signals; changing the representation later would
touch every model (FR-001, FR-002). Optical carriers (~193 THz) cannot be
sampled directly at useful simulation lengths; communication, photonic and
ultrafast models all work naturally with slowly varying envelopes.

## Decision
* Optical signals are **sampled complex envelopes** relative to a reference
  frequency `f_ref`: `E = Re{A exp(+i 2π f_ref t)}`, `A` in sqrt(W) so that
  `P = Σ_p |A_p|²`.
* Array layout: **time is the last axis**; optical field shape `(n_pol, N)`
  with `n_pol ∈ {1, 2}` (Jones vector).
* Uniform sampling on an immutable `TimeGrid(n_samples, dt, t0)`; delay is
  bookkept in `t0` (retarded frame).
* Signals are **immutable** dataclasses with read-only arrays and a read-only
  metadata mapping.
* Distinct types: `OpticalSignal`, `ElectricalSignal` (real, V or A),
  `DigitalSequence` (bits + rate), `SymbolSequence` (symbols + rate);
  `SignalKind` enumerates the port-level domain.
* Noise is realized in the samples (Monte Carlo) or tracked as a PSD
  (`NoiseRepresentation`), never both for the same source.
* Signals to be combined must share N, dt and `f_ref`; no implicit resampling.

## Alternatives considered
* *Real passband sampling of the optical field*: physically direct but needs
  > 400 THz sampling; infeasible and unnecessary.
* *Mutable signals / in-place processing*: faster in places but breaks caching,
  fan-out and reproducibility; can be optimised internally later.
* *Time on the first axis*: conflicts with NumPy/SciPy FFT defaults and
  broadcasting over polarization/channels.
* *Always 2 polarizations*: doubles cost for Phase 2 links; `n_pol=1` is
  explicit and polarization-aware components require 2.
* *Per-channel signals for WDM*: complicates nonlinear coupling (Phase 4);
  a common grid is standard practice. Revisit if memory becomes limiting.

## Consequences
* Bandwidth of an optical signal is limited to `fs` around `f_ref`; wideband
  (multi-THz) simulations need large N.
* Absolute time is preserved via `t0` without moving the waveform.
* Phase 1 must implement validation (shapes, finiteness, `f_ref > 0`).
