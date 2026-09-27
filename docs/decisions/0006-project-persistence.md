# ADR-0006: Project persistence format

* Status: Accepted
* Date: 2026-09-27

## Context
Projects must be saved/loaded (FR-009), diffable, safe (NFR-07), and survive
model evolution. Results can be large (millions of complex samples).

## Decision
* **Project file = JSON** (UTF-8), optionally YAML (same data model, needs
  `pyyaml`). Top-level keys:
  `format` (`"optobuild-project"`), `schema_version` (integer),
  `optobuild_version`, `metadata`, `simulation` (global settings, e.g.
  default sample rate, seed), `components` (list of
  `{name, type_id, version, parameters}` with SI values), `connections`
  (list of `{from: [node, port], to: [node, port]}`), `layout` (GUI-only,
  ignored by the engine).
* Loading: strict validation; unknown `type_id` or unsupported
  `schema_version` → `ProjectFormatError` with a hint. Component version
  mismatches produce a diagnostic (and migrations when defined).
* Schema migrations: pure functions `migrate_vN_to_vN+1(dict) -> dict`.
* **Results = HDF5** (`h5py`, optional extra), one group per node/port with
  datasets and attributes (grid, `f_ref`, units, provenance). Small results
  may be written as JSON/`.npz`.
* **Never pickle** user projects or results; never `eval`.

## Alternatives considered
* *Pickle*: arbitrary code execution on load; unstable across versions.
* *YAML only*: needs a dependency; JSON is stdlib and adequate.
* *Binary project format*: not diffable/reviewable.

## Consequences
* Projects are human-readable and reviewable in git.
* Registry must resolve `type_id` → class (Phase 1).
