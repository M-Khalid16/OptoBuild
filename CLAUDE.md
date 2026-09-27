# Project rules for automated contributors

OptoBuild is a scientific simulator. Correctness and traceability outrank
feature count. Read `docs/architecture.md` and `docs/roadmap.md` first.

## Phase discipline
* Work only on the current phase in `docs/roadmap.md`. Do not start a later
  phase (GUI, SSFM, FSO, coherent, photonic circuits, lasers, ultrafast)
  until the current phase's exit criteria hold and all tests pass.
* Keep the repo runnable after every commit: `pip install -e ".[dev]"`,
  `pytest`, `ruff check .`, `ruff format --check .` must pass.
* Small, logically separated commits. Inspect `git status` first; never
  overwrite unrelated user work.

## Architecture
* Respect the layers in `docs/architecture.md` §2 (enforced by
  `tests/integration/test_architecture_layers.py`). No upward or circular
  imports; absolute imports only.
* Equations live once, in `physics/` or `solvers/`. Components are thin
  wrappers; the GUI contains no physics and never imports physics/solvers/numerics.
* All FFTs through `optobuild.numerics.fft`; conventions in
  `docs/numerical_conventions.md` (ADR-0002). Time is the last array axis.
* Internal values are SI; unit conversion only in `optobuild.core.units`
  at boundaries. Store dB quantities linearly. Never compute from strings.
* Randomness only via the component's `context.rng` (ADR-0008); never the
  global `numpy.random` state.
* No `eval`, no pickle for user data, no global mutable simulation state.
* No new native/GPU dependencies; Numba only after a benchmark justifies it.
* Major design changes require a new ADR in `docs/decisions/`.

## Scientific quality — do not fake physics
* Never fabricate results, curves or numbers; never use random data as a
  stand-in for a model. Examples show only simulator output.
* Every physics model documents: equation, symbols, SI units, assumptions,
  validity range, numerical implementation, limitations, references, and has
  a validation test against an analytical/independent reference
  (`docs/physics_models.md` §1).
* Tolerances are justified from the error source, never tuned to pass.
* Surface numerical limitations as `Diagnostic`s (aliasing, window
  wrap-around, resolution, samples/symbol, statistical BER limits).
* Report limitations and failing tests honestly.

## Code style
* Type hints, docstrings with units, dataclasses/enums, small modules.
* Raise the domain exceptions in `optobuild.core.errors` with a `hint=`
  telling the user how to fix the problem.
* Tests go in `tests/{unit,integration,validation,regression}` with the code.
