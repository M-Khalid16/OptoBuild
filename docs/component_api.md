# Component API

Status: **implemented (Phase 1)** in `optobuild.components.base`,
`.spec`, `.registry`, `.reference`. Decision records: ADR-0004, ADR-0009.

## 1. Contract

A component is a **stateless, deterministic, thin wrapper**:

```python
class MachZehnderModulator(Component):
    """Push-pull MZM. Equations: docs/physics_models.md#mzm."""

    type_id = "optobuild.modulator.mzm"          # unique, stable, dotted lowercase
    version = "1.0.0"                             # bump when numerical output changes
    display_name = "Mach-Zehnder modulator"
    category = ComponentCategory.MODULATOR
    input_ports = (
        PortSpec("optical_in", SignalKind.OPTICAL, "CW carrier"),
        PortSpec("drive", SignalKind.ELECTRICAL, "drive voltage [V]"),
    )
    output_ports = (PortSpec("optical_out", SignalKind.OPTICAL),)
    parameter_specs = (
        ParameterSpec("v_pi", ParameterType.FLOAT, default=4.0, unit="V",
                      minimum=0.0, minimum_inclusive=False, symbol="V_pi"),
        # stored as linear power transmission (1.0 = lossless), shown in dB
        ParameterSpec("insertion_loss", ParameterType.FLOAT, default=1.0, unit="1",
                      display_unit="dB", minimum=0.0, minimum_inclusive=False,
                      maximum=1.0, symbol="IL"),
        ...
    )

    def run(self, inputs, context):
        field = physics.modulation.mzm_field_transfer(...)   # equation lives in physics
        return {"optical_out": inputs["optical_in"].with_field(field)}
```

(Illustrative; the MZM is implemented in Phase 2.)

| Element | Where | Rule |
|---|---|---|
| unique type identifier | `type_id` | dotted lowercase, checked at class creation; persisted in project files; never reused for a different model |
| component version | `version` | semantic version of the *model*; part of the cache key and project file |
| readable name | `display_name` (class), `name` (instance) | instance names unique within a graph |
| category | `category: ComponentCategory` | palette grouping and docs |
| typed ports | `input_ports`, `output_ports` (`PortSpec`) | names unique per direction; kind from `SignalKind` |
| parameter schema | `parameter_specs` (`ParameterSpec`) | SI `unit`, optional `display_unit`, range, choices, symbol, description |
| parameter validation | `__init__` (schema) + `validate() -> list[Diagnostic]` (cross-parameter) | invalid parameters raise `InvalidParameterError` at construction, listing every problem; warnings kept in `diagnostics` (ADR-0009) |
| execution | `run(inputs, context) -> {port: signal}` | pure function of parameters, inputs and `context.rng` |
| metadata / docs | class docstring via `documentation()`; physics docs linked | equations documented once, in physics docs |

## 2. Ports and connections

* One **input** port accepts exactly one connection. Components needing a
  variable number of inputs (N-way combiner) generate ports from a parameter
  by overriding `inputs()` / `outputs()`.
* Connections require **equal `SignalKind`** on both ends; otherwise
  `PortTypeMismatchError` explaining which conversion component to insert.
* **Fan-out**:
  * `ELECTRICAL`, `DIGITAL`, `SYMBOLS` outputs may fan out freely
    (ideal high-impedance probing / copying of data).
  * An `OPTICAL` output may feed **at most one non-tap input** plus any number
    of **tap** inputs (`PortSpec.tap=True`: power meter, OSA, eye analyzer).
    Duplicating optical power into two physical paths would violate energy
    conservation; use a splitter component.
* Unconnected non-optional inputs are a graph validation error.
* Unconnected outputs are allowed (result simply not consumed).

Supported topologies: 1-in/1-out, multi-input (combiners, modulators with
electrical drive), multi-output (splitters, demultiplexers), sinks
(analyzers, zero outputs), sources (zero inputs). Feedback/cavity topologies
are **not** expressed as graph cycles in the feed-forward executor (ADR-0005).

## 3. Parameters

* Stored values are SI numbers (or bool/int/choice/string). `display_unit`
  is advisory for the GUI and reports; conversions use `core.units`.
* dB-valued physical quantities (insertion loss, extinction ratio) are stored
  **linear** (e.g. power ratio) with `display_unit="dB"`, so that physics code
  never has to know about dB. The Phase 1 validator enforces the range on the
  stored (SI/linear) value.
* Parameters are immutable per instance; changing a parameter creates a new
  instance (or goes through the graph API, which invalidates cached
  downstream results).
* The GUI (Phase 3) generates forms from `ParameterSpec`; components never
  contain GUI code.

## 4. Run context

`RunContext` (a `Protocol` in `components.base`) provides:

| Member | Purpose |
|---|---|
| `rng` | `numpy.random.Generator` private to this component instance, derived from (project seed, name[, trial]) (ADR-0008); **only for components declaring `stochastic = True`** (ADR-0012) |
| `layout` | the run's `SimulationLayout` (bit rate, bits, samples per bit) or `None` (ADR-0011) |
| `logger` | `logging.Logger` named `optobuild.run.<component name>` |
| `check_cancelled()` | raises `SimulationCancelledError`; long loops call it periodically |
| `report_progress(fraction, message)` | progress in [0, 1] |
| `record(key, value)` | store an analysis result (number, string, array, tuple, dataclass); arrays are copied read-only (ADR-0009) |
| `warn(diagnostic)` | attach a run-time `Diagnostic` (e.g. aliasing risk) to the node's results |

Components must not create their own generators, read global state, or
perform I/O other than logging. A component that draws random numbers sets
the class attribute `stochastic = True`; the engine refuses `rng` access
otherwise, because only stochastic components have the seed and trial in
their cache keys.

Source components that can follow the global layout declare the shared
parameter `TIMING_SOURCE_SPEC` (`timing_source` = `parameters` | `layout`) and
obtain the layout with `require_layout(context, name)`.

## 5. Errors and diagnostics

* Parameter problems: construction raises `InvalidParameterError` naming
  every invalid value with its SI unit and range; a component instance is
  therefore always valid. `validate()` warnings are kept on the instance and
  reported by `SimulationGraph.validate()`; graphs with `ERROR` diagnostics
  (e.g. unconnected inputs) are refused by the executor.
* Wrong input kinds at run time (should be prevented by graph validation):
  `SignalTypeError`.
* Numerical problems discovered during `run` (e.g. aliasing): warnings via
  diagnostics; unrecoverable ones raise `SamplingError` /
  `NumericalStabilityError`.
* Any other exception raised inside `run` is wrapped by the engine in
  `ComponentExecutionError` naming the component instance.

## 6. Registry and plugins

`ComponentRegistry` maps `type_id -> class` and is used by persistence to
instantiate components from project files. Registries are ordinary objects;
`builtin_registry()` returns a fresh registry holding
`components.library.BUILTIN_COMPONENTS` (no global mutable singleton).
Plugins register through Python entry points (group `optobuild.components`,
`optobuild.plugins.discovery`, ADR-0021): an installed package exports a
`Component` subclass or an iterable of them, e.g.

```toml
[project.entry-points."optobuild.components"]
my_parts = "my_package.optobuild_plugin:COMPONENTS"
```

Loading is explicit (`plugin_registry()`, CLI `--plugins`); a plugin that
fails or reuses an existing `type_id` is reported and skipped. Plugins are
ordinary Python code: install only packages you trust. Project files never
name code, only `type_id` strings.

## 6a. Reference components

`components.reference` provides deterministic, trivially predictable blocks
(`RampSource`, `Gain`, `Adder`, `GaussianNoise`, `Recorder`) so the graph,
engine, cache, persistence and seeding can be validated exactly, independent
of optical physics (`optobuild demo reference`).

## 7. Testing requirements for each component

1. Metadata test: ports/parameters as documented.
2. Validation test: invalid parameters produce `ERROR` diagnostics with hints.
3. Delegation test: output equals the physics function applied to the input
   (guarantees no duplicated physics).
4. Physics validation lives with the physics function (tests/validation).
