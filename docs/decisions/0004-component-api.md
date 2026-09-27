# ADR-0004: Component API

* Status: Accepted
* Date: 2026-09-27

## Context
Components are the extension surface (FR-005, FR-100, FR-170): they must be
typed, serialisable, GUI-describable and must not duplicate physics.

## Decision
* Abstract base class `Component` with class-level metadata: `type_id`
  (dotted lowercase, validated), `version` (model semver), `display_name`,
  `category`, `input_ports`/`output_ports` (`PortSpec`), `parameter_specs`
  (`ParameterSpec`). Instance-level `inputs()`/`outputs()` may derive ports
  from parameters (N-way devices).
* `run(inputs, context) -> Mapping[port, signal]` is stateless and
  deterministic given parameters, inputs and `context.rng`.
* `RunContext` is a `Protocol` defined in the components layer and implemented
  by the engine (dependency inversion).
* `validate() -> list[Diagnostic]`: schema checks in the base class (Phase 1),
  physics cross-checks in subclasses.
* Ports are typed by `SignalKind`; connections require equal kinds.
  Optical outputs may feed ≤ 1 non-tap input plus any number of tap
  (monitoring) inputs; splitting power needs a splitter component.
* Components delegate all equations to `physics`/`solvers`/`analysis`.

## Alternatives considered
* *Function-based components (decorated functions)*: lighter, but metadata
  (ports, versions, schemas) becomes ad hoc; can be added later as sugar that
  generates a `Component` subclass.
* *Stateful components with `reset()`*: required only for iterative/cavity
  simulation; handled by the iterative executor (ADR-0005) instead of
  complicating every component.
* *Unrestricted optical fan-out*: silently violates energy conservation.

## Consequences
* GUI forms, persistence and docs are generated from the same metadata.
* Cache keys can include `(type_id, version, parameters)`.
