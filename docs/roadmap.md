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

## Phase 3 — GUI and schematic editor ✅

Exit criteria (all met, tested headless in `tests/integration/test_gui_qt.py`
and `tests/unit/test_gui_*.py`):

* components from a palette grouped by category; drag-to-connect with the
  graph's type and fan-out rules (rejections explained in the status bar);
* parameter forms generated from `ParameterSpec` with display units
  (dBm, nm, GHz, km, dB/km, ps/(nm km), dB loss, ...), values stored in SI;
  invalid input rejected with the component's message and reverted;
* rename, move, delete, undo/redo; open/save projects; built-in demos;
* simulation settings (seed, global layout);
* background runs with progress and cancellation, result caching between runs;
* results: scalar table, eye diagram, optical spectrum, waveforms of every
  output port, diagnostics; stale-result indicator; HDF5 export;
* no physics in the GUI (import rules enforced by test).

Status: complete (v0.4.0). ADR-0013. Screenshot: `docs/images/gui_optical_link.png`.

## Phase 4 — Nonlinear fiber propagation (SSFM) ✅

* Scalar NLSE (loss, β2, β3, Kerr SPM) by symmetric split-step Fourier with
  exact Kerr+loss sub-step; adaptive nonlinear-phase step control or fixed
  steps (ADR-0014).
* Diagnostics: per-step nonlinear phase, spectral truncation, window
  wrap-around, non-finite fields.
* Optical pulse source (Gaussian/sech); soliton demo and example.
* Validation: linear limit, exact SPM, SPM chirp sign, fundamental soliton,
  N = 2 soliton vs closed form, period, convergence order, energy.

Status: complete (v0.5.0). Deferred to Phase 9: Raman, self-steepening,
vector/multichannel propagation, higher-order adaptive schemes.

## Phase 5 — FSO channel and link budget ✅

* Gaussian-beam geometry, exact aperture collection with static and random
  pointing (Rice), beam wander; Kim/Kruse visibility and rain attenuation;
  Rytov variance, log-normal and Gamma-Gamma fading, aperture averaging.
* `FSOChannel` component (quasi-static, one state per Monte Carlo trial),
  analytic mean gain, outage probability, link budget and margin
  (`optobuild fso-budget`), FSO link demo and example (ADR-0015).
* Validation against closed forms, independent quadrature, the Koschmieder
  definition, Farid–Hranilovic, moment integrals and Monte Carlo.

Status: complete (v0.6.0). Deferred: phase screens / coherent FSO, temporal
fading, slant paths with Cn²(h), snow, angle-of-arrival.

## Phase 6 — Coherent optical communication (single polarization) ✅

* Symbols, Gray constellations, RC/RRC shaping; IQ modulator from nested
  MZMs; laser linewidth; amplifier with output OSNR; intradyne receiver with
  balanced detection; DSP (CD compensation, two-pass FOE, matched filter,
  Oerder–Meyr timing, BPS); EVM/SNR/SER/BER analyzer; constellation view
  in the GUI (ADR-0016).
* Validation against exact AWGN theory, the OSNR→SNR relation, receiver-noise
  theory and synthetic DSP tests.

Status: complete (v0.7.0).

## Phase 6b — Dual-polarization coherent systems ✅

* Jones-matrix optics: PBS/PBC, polarization controller with DGD, random PMD
  (waveplate model) with DGD eigenanalysis.
* Hybrid phase/amplitude imbalance and I/Q skew in the receiver; GSOP and
  deskew in the DSP.
* Dual-polarization DSP: joint timing, 2×2 CMA/RDE butterfly equalizer with
  singularity-free initialization, two-pass FOE, BPS; analyzer with
  polarization-swap resolution; DP demo and example (ADR-0017).

Status: complete (v0.8.0). Known limitation: blind DP-64QAM convergence is
unreliable (needs training/pilot-aided equalization, deferred).

## Phase 7 — Photonic circuit simulation ✅

* Integrated-optics models: dispersive lossy waveguides, directional
  couplers, all-pass and add-drop microrings, MZIs, uniform Bragg gratings
  (coupled-mode theory), thin-film transfer-matrix method.
* Frequency-domain S-matrix circuit solver for arbitrary netlists with
  feedback and reflections; wavelength sweeps.
* Resonance analysis: FSR, FWHM, loaded Q, extinction, insertion loss,
  group delay.
* Signal-flow components (rings, MZI, Bragg grating) for time-domain links;
  ring-filter demo; GUI device-response view; example with coupled rings
  (ADR-0018).

Status: complete (v0.9.0). Deferred: mode solver, thermal/electro-optic
phase-shifter models, apodized/chirped gratings as components, netlists in
project files, back-reflection feedback in time-domain links.

## Phase 8 — Laser and amplifier modelling ✅

* Semiconductor laser rate equations (gain compression, spontaneous
  emission, α-factor chirp) with Langevin noise; directly modulated laser
  component; fixed-step RK4/SDE solver.
* Two-level EDFA (co-pumped) with ASE and physical noise figure; EDFA
  component.
* Erbium fiber ring laser steady state; fiber ring laser source.
* DML link demo, laser example (ADR-0019).

Status: complete (v0.10.0). Deferred: multimode/side-mode lasers, thermal
effects, EDFA ASE self-saturation and backward pumping, laser dynamics of
fiber cavities (Phase 9).

## Later phases

| Phase | Topic |
|---|---|
| 9 | Ultrafast pulse propagation and mode-locked cavities (iterative executor) |
| 10 | Optimization, Monte Carlo, reports, plugins, acceleration |
