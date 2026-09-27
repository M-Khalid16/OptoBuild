# Architecture

Status: **Phase 0 (foundation)**. This document is normative: code that
contradicts it is a bug in either the code or this document, and must be
resolved by changing one of them explicitly (with an ADR for major changes).

## 1. Goals that drive the architecture

1. **Scientific correctness first.** Every physical result must be traceable
   to a documented equation, a documented numerical method and a validation
   test ([physics_models.md](physics_models.md), [testing_strategy.md](testing_strategy.md)).
2. **One equation, one place.** Physics lives in `physics/` and `solvers/`.
   Components, the GUI, sweeps and reports call those functions; they never
   re-implement them.
3. **Engine independent of GUI.** Everything must run headless from Python or
   the CLI. The GUI (Phase 3) is a client of the engine.
4. **Stable conventions before features.** Signal representation, FFT
   convention and units ([signal_model.md](signal_model.md),
   [numerical_conventions.md](numerical_conventions.md)) are fixed before
   components are added.
5. **Reproducibility.** Given a project file and its seed, a simulation is
   bit-for-bit repeatable on the same platform/library versions.

## 2. Layered structure

The package `optobuild` is split into subsystems arranged in layers. A module
may import from its **own or lower** layers only. This is enforced by
`tests/integration/test_architecture_layers.py`.

```
 L9  cli        gui         plugins                (entry points, user interfaces)
 L8  persistence  sweeps  optimization  reporting  (orchestration around the engine)
 L7  engine                                        (executes graphs)
 L6  graph                                         (graph model, topology, validation)
 L5  components                                    (thin wrappers: ports + params + run)
 L4  solvers                                       (propagation / integration algorithms)
 L3  physics            analysis                   (pure models / pure measurements)
 L2  signals                                       (immutable signal types)
 L1  numerics                                      (grids, FFT convention, generic numerics)
 L0  core                                          (constants, units, errors, diagnostics, rng)
```

Additional rule: `gui` must not import `physics`, `solvers` or `numerics`.
It displays results produced by `analysis`/`engine` and edits graphs through
`graph`/`components` metadata.

### Why this layering

* `components` sits *above* `physics`/`solvers`/`analysis` so it can call them,
  and *below* `graph` so the graph can type-check ports using component
  metadata without circular imports.
* The **run context** protocol that components receive is defined in
  `components` (L5) and implemented by `engine` (L7): components never import
  the engine (dependency inversion).
* `analysis` is a peer of `physics`: measurements (BER, eye, spectra) are pure
  functions of signals and must not depend on how the signal was produced.
* `numerics` is separate from `solvers`: the FFT convention and grids are
  needed by `signals` (L2), while propagation solvers need `physics` (L3).

## 3. Subsystem responsibilities

| Subsystem | Responsibility | Must NOT contain |
|---|---|---|
| `core` | exact SI constants (`constants`), unit conversion (`units`), exception hierarchy (`errors`), `Diagnostic` (`diagnostics`), seed derivation (`rng`), logging helpers (`log`) | any signal or physics semantics |
| `numerics` | `TimeGrid`, FFT convention (`numerics.fft`), later windows, resampling, generic filters, sampling diagnostics | device physics |
| `signals` | `OpticalSignal`, `ElectricalSignal`, `DigitalSequence`, `SymbolSequence`, noise representation, `SignalKind` | algorithms beyond trivial accessors (power, wavelength) |
| `physics` | equations: fiber attenuation/dispersion, MZM transfer, photodetection, noise PSDs; later atmosphere, lasers, photonic elements | ports, parameters schemas, GUI, file I/O |
| `analysis` | BER counting, Q-factor, eye diagram data, spectra, power, later EVM/OSNR/constellation/link budget | signal generation |
| `solvers` | linear frequency-domain propagation, SSFM (P4), ODE (P8), cavity round-trip (P9) | component metadata |
| `components` | `Component` ABC (`base`), `PortSpec`/`ParameterSpec` (`spec`), `ComponentRegistry` (`registry`), built-in list (`library`), reference blocks (`reference`), concrete blocks grouped by category | equations (delegate to physics) |
| `graph` | graph data model, connections, port-kind checking, topology, cycle detection | execution |
| `engine` | `RunContext` implementation, DAG executor, cache & invalidation, progress, cancellation; later iterative executor | physics |
| `persistence` | project JSON/YAML, schema versions & migrations, HDF5 results | pickle of user data |
| `sweeps`, `optimization` | parameter sweeps, Monte Carlo, optimizers built on the engine | physics |
| `reporting` | Matplotlib figures, report generation | physics |
| `plugins` | entry-point discovery and registration of third-party components | — |
| `cli` | command-line entry points | physics |
| `gui` | PySide6 application, schematic editor, auto-generated parameter forms | physics, numerics |

## 4. Repository layout

```
pyproject.toml            packaging, pytest, ruff configuration
CLAUDE.md                 rules for automated contributors
mkdocs.yml                documentation site
src/optobuild/
    core/ numerics/ signals/ physics/ analysis/ solvers/ components/
    graph/ engine/ persistence/ sweeps/ optimization/ reporting/
    plugins/ cli/ gui/
tests/
    unit/ integration/ validation/ regression/
benchmarks/               performance measurements (not collected by pytest)
examples/                 runnable end-to-end scripts
docs/                     this documentation, decisions/ (ADRs)
```

Nested subpackages are created **when their first module is written**, not in
advance, to avoid empty scaffolding. The planned nesting is:

```
physics/    fiber/ modulation/ detection/ noise/ sources/ | atmospheric/ laser/ photonics/
analysis/   ber, qfactor, eye, spectrum, power | evm, osnr, constellation, link_budget, pulse
solvers/    linear_propagation | ssfm/ ode/ cavity/
components/ sources/ modulators/ fiber/ detectors/ electrical/ analyzers/
            | amplifiers/ passive/ dsp/ fso/ photonics/ laser/
engine/     context, executor, cache | iterative
```

### Changes relative to the initially proposed structure

| Proposed | Adopted | Reason |
|---|---|---|
| `optical_sim` | `optobuild` | Matches the repository/project name; avoids a generic name likely to clash on PyPI. Rename is mechanical if desired. |
| `utils/` | `core/` | A "utils" bucket attracts unrelated code; `core` has a defined scope (constants, units, errors, diagnostics, rng). |
| `engine/graph`, `engine/execution` | top-level `graph/` and `engine/` | The graph model is needed by persistence and the GUI without pulling in the executor; separates "what" from "how". |
| FFT inside `solvers/fft` | `numerics/fft` | Signals (L2) need the FFT convention; solvers are L4. |
| — | `physics/modulation`, `physics/detection` | MZM and photodiode physics had no home in the proposal. |
| — | `components/electrical` | Electrical filters, decision circuits, amplifiers. |
| `reports/` | `reporting/` | Naming consistency (subsystem names are activities/domains). |
| — | `cli/`, `gui/` placeholders | Makes the "GUI contains no physics" rule testable from day one. |
| `benchmarks/` under `tests/` | top-level `benchmarks/` | Benchmarks are not correctness tests and must not slow the suite. |

## 5. Data flow of a simulation

```
Project file (JSON) --persistence--> Graph (components + connections)
      |                                     |
      |                     graph.validate(): ports, kinds, cycles, params
      v                                     v
  root seed ------------------------> engine.Executor
                                          | topological order
                                          | for each node: RunContext(rng, logger, cancel, progress)
                                          |   component.run(inputs) -> physics/solvers/analysis
                                          v
                              immutable Signals per output port (inspectable, cacheable)
                                          |
                            analysis results -> reporting / GUI / HDF5
```

## 6. Extension points

* **New component**: subclass `Component`, declare metadata, delegate to
  physics; register with the registry (Phase 1) or via a plugin entry point
  (Phase 10). See [component_api.md](component_api.md).
* **New physics model**: add a pure function in `physics/` with the
  documentation block required by [physics_models.md](physics_models.md) and a
  validation test.
* **Alternative numerical backend**: all FFTs go through `numerics.fft`;
  array code uses the NumPy API (ADR-0007).

## 7. Architecture risks and mitigations

| Risk | Mitigation |
|---|---|
| Physics duplicated in components/GUI | Layer test; review rule "components contain no equations"; physics functions are the only place tests validate. |
| Sign/normalization errors in spectral code | Single FFT module, analytic-pair validation tests, documented envelope convention (ADR-0002). |
| Hidden unit conversions | SI everywhere internally; conversion only in `core.units` at boundaries (ADR-0003). |
| Non-reproducible randomness | Per-component generators derived from the project seed by a stable hash of the component name (ADR-0008). |
| Feedback systems forced into a DAG executor | Separate iterative executor for cavities (ADR-0005). |
| Memory growth with long sequences / many cached signals | Cache is bounded and optional; signals immutable so sharing is safe; HDF5 offload later. |
| Over-scaffolding (many empty modules) | Subpackages created only with their first real module. |
| Premature optimisation / native deps | NumPy/SciPy only; Numba only after benchmarks (ADR-0007). |
