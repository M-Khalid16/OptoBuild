# ADR-0009: Construction-time parameter validation and recorded results

* Status: Accepted
* Date: 2026-09-27
* Amends: ADR-0004 (component API)

## Context
Phase 1 implementation raised two questions ADR-0004 left open:
(1) when parameter validation happens, and (2) how sinks/analyzers (BER
analyzer, power meter), which produce *results* rather than signals, return
them without abusing output ports.

## Decision
1. `Component.__init__` validates every parameter against its
   `ParameterSpec` (type, finiteness, range, choices, required) and then calls
   `validate()` for cross-parameter checks. Any `ERROR` raises
   `InvalidParameterError` listing *all* problems. A component instance is
   therefore always valid and holds canonical SI values; parameter changes go
   through `with_parameters()` / `SimulationGraph.set_parameters()`, which
   re-validate. `WARNING` diagnostics are kept in `component.diagnostics`.
2. `RunContext` gains `record(key, value)` for named results and
   `warn(diagnostic)` for run-time diagnostics. The engine stores them per
   node in `SimulationResult` (`result(node, key)`), makes recorded arrays
   read-only, and caches them together with output signals.

## Alternatives considered
* *Deferred validation (store invalid values, validate before running)*:
  convenient for a GUI with half-edited forms, but every consumer would need
  to handle invalid components. A GUI can keep draft values outside the
  component until they validate.
* *Analyzer outputs as special port kinds*: would pollute the port type system
  and the connection rules with non-signal data.

## Consequences
* Errors surface at the point where the value is entered.
* Results are part of the deterministic, cacheable node output.
