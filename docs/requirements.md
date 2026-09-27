# Requirements

Identifiers are stable; tests and ADRs may reference them.
Priority: **M** = must (current or next phase), **L** = later phase (number given).

## 1. Scope

A professional-grade optical and photonics simulation platform (inspired by
OptiSystem, VPIphotonics, photonic-circuit simulators, FSO link tools and
ultrafast-laser modelling environments), built incrementally on a validated
numerical core.

## 2. Functional requirements

### Foundation (Phases 0–1)

| ID | Requirement | Pri |
|---|---|---|
| FR-001 | Represent optical signals as sampled complex envelopes with reference frequency, polarization (1 or 2), SI units, metadata. | M |
| FR-002 | Represent electrical (real V/A), digital (bits) and symbol sequences. | M |
| FR-003 | Single, documented FFT/PSD convention used by all modules. | M (done) |
| FR-004 | Centralized unit conversion (nm, µm, GHz, THz, ps, fs, km, dBm, dB/km, ps/(nm·km), W, mW, µW); no computation from strings. | M |
| FR-005 | Components with type id, version, name, category, typed ports, parameter schema, validation, execution, docs hook. | M (interface done) |
| FR-006 | Directed simulation graph: registration, connection validation, port-kind checks, topological order, cycle detection. | M |
| FR-007 | Feed-forward execution engine with intermediate-signal inspection, progress and cancellation. | M |
| FR-008 | Cached execution with invalidation of downstream nodes on parameter change. | M (Phase 1 basic, refine later) |
| FR-009 | Project save/load (JSON; YAML optional) with schema version. | M |
| FR-010 | Reproducible randomness from a project seed, independent of execution order. | M |
| FR-011 | CLI running a project/example headlessly. | M |

### First optical link (Phase 2)

| ID | Requirement | Pri |
|---|---|---|
| FR-020 | CW laser: wavelength/frequency, power, phase. | M |
| FR-021 | PRBS generator (deterministic; configurable order). | M |
| FR-022 | NRZ generator: symbol rate, samples/symbol, levels. | M |
| FR-023 | MZM: V_π, bias, insertion loss, extinction ratio, analytic transfer. | M |
| FR-024 | Linear fiber: attenuation, delay, chromatic dispersion. | M |
| FR-025 | PIN: responsivity, shot, dark-current and thermal noise. | M |
| FR-026 | Electrical low-pass filter: bandwidth, order. | M |
| FR-027 | Decision circuit and BER analyzer with direct error counting. | M |
| FR-028 | Optical power meter, optical spectrum analyzer, eye-diagram data. | M |

### Later phases (summary; detailed when the phase starts)

| ID | Requirement | Pri |
|---|---|---|
| FR-100 | GUI with schematic editor and auto-generated parameter forms. | L3 |
| FR-110 | SSFM/NLSE nonlinear propagation with step/phase/spectral diagnostics; higher-order dispersion, SPM, Raman, self-steepening. | L4 |
| FR-120 | FSO: geometric spreading, divergence, apertures, pointing loss, atmospheric attenuation (visibility, fog, haze, rain), turbulence (Cn², scintillation, log-normal, Gamma-Gamma, beam wander), received power, SNR, BER, outage probability, link margin. | L5 |
| FR-130 | Modulation: OOK, RZ/NRZ, PPM, M-PPM, PAM, DPSK, QPSK, OFDM; coherent receivers; EVM, constellations, OSNR. | L6 |
| FR-140 | Photonic circuits: waveguides, couplers, MZI, rings (all-pass, add-drop), Bragg gratings, phase shifters, filters, wavelength sweeps, S-/transfer-matrix, resonance, Q, FSR, insertion loss, group delay. | L7 |
| FR-150 | Lasers: CW cavities, semiconductor rate equations, fiber lasers, gain saturation, threshold, slope efficiency, longitudinal modes, FSR. | L8 |
| FR-160 | Ultrafast: pulse sources, cavity propagation, mode locking, saturable absorbers, dispersion compensation. | L9 |
| FR-170 | Parameter sweeps, Monte Carlo, optimization, automated reports, plugins, acceleration. | L10 |
| FR-180 | Link-budget analysis. | L5 |

## 3. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | **Correctness**: every physics model has documented equations, units, assumptions, validity range, numerics, limitations, references, and validation tests (physics_models.md §1). |
| NFR-02 | **No fabricated results**: examples/reports only show simulator output; no hand-drawn or random placeholder curves. |
| NFR-03 | **Transparency**: numerical limitations surface as diagnostics (aliasing, window, resolution, sampling, statistics). |
| NFR-04 | **GUI independence**: the engine runs headless; GUI contains no physics (enforced by test). |
| NFR-05 | **Layering**: no upward imports (enforced by test); no circular imports. |
| NFR-06 | **Reproducibility**: same project + seed + versions ⇒ identical results. |
| NFR-07 | **Safety**: no `eval`, no pickle for user projects; project files are data only. |
| NFR-08 | **Portability**: pure Python + NumPy/SciPy for the core; Python ≥ 3.10. |
| NFR-09 | **Performance**: acceptable for 2^15–2^20 samples per signal on a laptop; acceleration only after profiling (ADR-0007). |
| NFR-10 | **Maintainability**: type hints, docstrings, small modules, explicit errors with hints, ruff-clean. |
| NFR-11 | **Runnable after every milestone**: `pip install -e .` and `pytest` pass on every commit to the main branch. |
