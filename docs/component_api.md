# Component API

Status: **interface defined** (`optobuild.components.base`,
`optobuild.components.spec`); parameter validation, registry and concrete
components arrive in Phase 1–2. Decision record: ADR-0004.

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
| parameter validation | `validate() -> list[Diagnostic]` | schema checks (Phase 1, in the base class) + cross-parameter physics checks (subclass) |
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
| `rng` | `numpy.random.Generator` private to this component instance, derived from the project seed (ADR-0008) |
| `logger` | `logging.Logger` named `optobuild.run.<component name>` |
| `check_cancelled()` | raises `SimulationCancelledError`; long loops call it periodically |
| `report_progress(fraction, message)` | progress in [0, 1] |

Components must not create their own generators, read global state, or
perform I/O other than logging.

## 5. Errors and diagnostics

* Parameter problems: `validate()` returns `Diagnostic` records; the engine
  refuses to run a graph with `ERROR` diagnostics, and attaches warnings to
  results.
* Wrong input kinds at run time (should be prevented by graph validation):
  `SignalTypeError`.
* Numerical problems discovered during `run` (e.g. aliasing): warnings via
  diagnostics; unrecoverable ones raise `SamplingError` /
  `NumericalStabilityError`.
* Any other exception raised inside `run` is wrapped by the engine in
  `ComponentExecutionError` naming the component instance.

## 6. Registry and plugins

Phase 1 adds a `ComponentRegistry` mapping `type_id -> class`, used by
persistence to instantiate components from project files. Registration is
explicit (no import-time side effects on a global mutable registry beyond the
built-in library); plugins (Phase 10) register through Python entry points
(`optobuild.components` group).

## 7. Testing requirements for each component

1. Metadata test: ports/parameters as documented.
2. Validation test: invalid parameters produce `ERROR` diagnostics with hints.
3. Delegation test: output equals the physics function applied to the input
   (guarantees no duplicated physics).
4. Physics validation lives with the physics function (tests/validation).
