# Roadmap

Development proceeds strictly in order. A phase starts only when the previous
phase's exit criteria are met and the test suite passes. Jumping ahead is
allowed only for an *architectural dependency* (e.g. an interface that a
later phase needs must not be painted into a corner), recorded in an ADR.

## Phase 0 — Architecture and conventions ✅ (this milestone)

* Documentation: requirements, architecture, signal model, component API,
  numerical conventions, physics-model template, testing strategy, roadmap.
* ADRs 0001–0008.
* Package skeleton with layered subsystems; `pyproject.toml`; pytest + ruff.
* Implemented and tested: `TimeGrid`, FFT/PSD convention, exception
  hierarchy, diagnostics, exact SI constants, component *interface*
  (`Component`, `PortSpec`, `ParameterSpec`, `RunContext`).
* Architecture tests: import layering, GUI-without-physics.

Exit: docs internally consistent; tests and ruff pass; package installs. ✅

## Phase 1 — Foundation implementation ✅

Milestones (each a separate commit with tests):

1. `core.units` — explicit conversion functions + whitelisted unit table for
   parsing user input at boundaries.
2. `core.rng` — seed derivation (ADR-0008); `core.logging` setup.
3. `signals` — `OpticalSignal`, `ElectricalSignal`, `DigitalSequence`,
   `SymbolSequence`, `NoiseRepresentation` (immutability, validation).
4. Parameter validation in `Component.validate()` from `ParameterSpec`.
5. `components.registry`.
6. `graph` — nodes, connections, port-kind and fan-out rules, topology,
   cycle detection.
7. `engine` — `RunContext`, feed-forward executor, intermediate-signal
   inspection, progress, cancellation, basic cache and invalidation.
8. `persistence` — JSON project save/load, schema version.
9. `cli` + `examples/` — simple multi-component propagation using
   test/utility components only (gain, delay, probe; no device physics).

Status: complete (v0.1.0). `SymbolSequence`, tracked `NoiseRepresentation`
and HDF5 result storage were deferred until a component needs them.

Exit criteria (all must hold): component instances can be created; parameter
validation works; ports are typed; valid connections accepted and invalid
rejected; graph built programmatically; topology validated; execution order
determined; a signal propagates across several components; project config
saves and loads; seeds reproducible; test suite passes; package installs;
CLI example executes.

## Phase 2 — First validated optical link ✅

PRBS → NRZ → MZM ← CW laser; MZM → fiber (attenuation, delay, dispersion) →
PIN → electrical LPF → decision → BER; plus power meter, OSA, eye diagram.
All models per [physics_models.md](physics_models.md) §3 with the validation
catalogue of [testing_strategy.md](testing_strategy.md) §3; one end-to-end
example; sampling diagnostics (aliasing, window wrap-around, pattern
periodicity, samples/symbol).

Status: complete (v0.2.0). Example: `examples/optical_link.py`,
`optobuild demo optical_link`. Validation summary in
[physics_models.md](physics_models.md) §3. Deferred: global layout
parameters (ADR-0010), tracked-noise representation, HDF5 results.

## Phase 2.1 — Link hardening ✅

* Global simulation layout, project schema v2 with v1 migration (ADR-0011).
* HDF5 result storage with embedded project provenance (`--save-results`).
* Stochastic-component declaration, seed/trial-aware caching and Monte
  Carlo BER accumulation (`optobuild ber`) (ADR-0012).

Status: complete (v0.3.0).

## Later phases

| Phase | Topic |
|---|---|
| 3 | GUI and schematic editor (PySide6, PyQtGraph); forms from `ParameterSpec` |
| 4 | Advanced fiber propagation: SSFM/NLSE, step-size and nonlinear-phase diagnostics, spectral truncation, wrap-around |
| 5 | FSO channel and link-budget analysis |
| 6 | Coherent optical communication (dual-pol, DSP, EVM, constellations, OSNR) |
| 7 | Photonic circuit simulation (transfer/S-matrix) |
| 8 | CW, semiconductor and fiber laser modelling |
| 9 | Ultrafast pulse propagation and mode-locked cavities (iterative executor) |
| 10 | Optimization, Monte Carlo, reports, plugins, acceleration |
