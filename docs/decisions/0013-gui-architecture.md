# ADR-0013: GUI architecture

* Status: Accepted
* Date: 2026-09-27

## Context
Phase 3 adds a desktop editor (PySide6, pyqtgraph). Requirements: no physics
in the GUI (NFR-04), forms generated from `ParameterSpec` (FR-100), engine
usable headless, GUI testable without a display.

## Decision
1. **Two tiers inside `optobuild.gui`:**
   * Qt-free models, unit-tested without Qt: `forms` (fields from
     `ParameterSpec`, display-unit conversion via `core.units`), `document`
     (validated edits, snapshot undo/redo, schematic positions, observers),
     `plotdata` (plot arrays from recorded results; rescaling only).
   * Qt views in `gui.qt`: palette, schematic canvas, parameter panel,
     results tabs, simulation-settings dialog, background runner, main window.
2. **All edits go through `ProjectDocument`**, which applies the same graph
   and component validation as scripts. Undo/redo store the whole project in
   its persistence form, so undo restores exactly what save/load would.
3. **Runs use a snapshot** of the project in a `QThread` with the engine's
   progress/cancel hooks and a persistent result cache; the GUI never mutates
   what a running simulation reads.
4. **Import rules** (enforced by `test_architecture_layers.py`): `gui` may not
   import `physics`, `solvers` or `numerics`, **except** the data-only module
   `optobuild.numerics.layout` (the GUI must edit the layout). Only
   `optobuild.gui` may import Qt/pyqtgraph; `gui.forms/document/plotdata` must
   not import Qt at all.
5. **Presentation units**: display units are declared on `ParameterSpec`; a
   `dB loss` presentation unit shows linear transmissions (e.g. MZM insertion
   loss) as positive losses. Stored values stay SI/linear.
6. Headless tests run with `QT_QPA_PLATFORM=offscreen` and are skipped when
   PySide6/pyqtgraph are not installed.

## Alternatives considered
* *Web front-end*: easier distribution, but plotting of large arrays and a
  schematic editor are more work; PySide6 was chosen in the requirements.
* *Qt model/view classes for the document*: would tie the document to Qt and
  prevent Qt-free testing.
* *Command objects for undo*: finer-grained, but each command needs an exact
  inverse; snapshots are simpler and provably consistent for projects of
  this size (serialization cost is milliseconds).

## Consequences
* The GUI can be replaced or complemented (e.g. a web view) by reusing the
  Qt-free tier.
* Snapshot undo stores whole projects (bounded to 200 steps).
* PySide6 needs system GL/EGL libraries on Linux (documented in the README).
