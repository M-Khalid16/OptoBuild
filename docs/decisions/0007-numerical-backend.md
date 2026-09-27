# ADR-0007: Numerical backend strategy

* Status: Accepted
* Date: 2026-09-27

## Context
Performance will matter for SSFM, Monte Carlo and cavities, but premature
native code slows development and hampers validation (NFR-08, NFR-09).

## Decision
* Core numerics: **NumPy** arrays (float64/complex128) and **SciPy**.
* All FFTs go through `optobuild.numerics.fft`, implemented on `scipy.fft`
  (which supports multithreading via `workers` and pluggable backends such as
  pyFFTW or CuPy later) — a single place to swap implementations.
* Physics/solver functions accept and return arrays and avoid Python loops
  over samples; they use the NumPy API only, to keep a future array-API /
  CuPy port feasible.
* **Numba** only after a benchmark in `benchmarks/` shows a hotspot that
  vectorization cannot fix; the pure-NumPy version stays as the reference
  implementation and validation oracle.
* No CUDA, C++, Rust or GPU frameworks in the foundation phases.

## Alternatives considered
* *JAX / PyTorch backend*: autodiff attractive for optimization (Phase 10),
  but heavy dependencies and different semantics (immutability, x64 flags);
  revisit in Phase 10.
* *C++ core with Python bindings*: highest speed, much higher cost of
  validation and contribution.

## Consequences
* Simple installation (`pip install`), fast iteration.
* Acceleration is an optimization behind stable interfaces, not a rewrite.
