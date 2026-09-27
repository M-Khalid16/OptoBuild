# ADR-0005: Graph execution architecture

* Status: Accepted
* Date: 2026-09-27

## Context
Most systems are feed-forward (transmitter → channel → receiver). Lasers and
mode-locked cavities (Phases 8–9) contain feedback loops. FR-006–FR-008.

## Decision
* `graph` (L6) holds the **model**: nodes (component instances), directed
  connections `(src node, out port) → (dst node, in port)`, validation and
  topology. It has no execution logic.
* `engine` (L7) holds **executors**:
  * `FeedForwardExecutor`: validates the graph, computes a deterministic
    topological order (Kahn's algorithm; ties broken by insertion order),
    runs each node once, stores output signals for inspection.
    A cycle raises `SimulationCycleError` naming the nodes and pointing to
    the iterative executor.
  * `IterativeExecutor` (Phase 8/9): runs an explicitly declared loop
    (a subgraph with a designated loop-closing connection) for round trips
    until a documented convergence criterion or max iterations; each round
    trip is itself a feed-forward pass. Not implemented before it is needed.
* Caching: key = hash of `(type_id, version, canonical parameters,
  input-signal fingerprints, derived seed)`; parameter change invalidates the
  node and its downstream closure. Cache is bounded and optional.
* Progress: per-node and intra-component fractions via `RunContext`.
  Cancellation: cooperative (`check_cancelled()`), checked between nodes and
  inside long loops.
* Errors inside a component are wrapped in `ComponentExecutionError`.

## Alternatives considered
* *Dataflow/streaming (block-by-block) execution like real-time SDR*: needed
  for very long sequences; possible later as another executor over the same
  graph model. Whole-signal execution is simpler to validate first.
* *Treating cycles with implicit unit delays*: hides physics (round-trip time
  must come from the cavity model).
* *Parallel execution*: independent branches could run concurrently; deferred
  until profiling shows benefit (ADR-0007).

## Consequences
* Feed-forward results are deterministic and inspectable per port.
* Cavity simulation gets an explicit, testable convergence contract.
