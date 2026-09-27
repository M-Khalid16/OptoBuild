# OptoBuild

A scientifically validated optical and photonics simulation platform, built
incrementally: foundation → first validated optical link → GUI → SSFM → FSO →
coherent systems → photonic circuits → lasers → ultrafast → optimization/HPC.

**Current status: Phase 0 (architecture and conventions).** There are no
physics models yet. What exists and is tested:

* the time/frequency grid and the project-wide FFT / PSD convention
  (`optobuild.numerics`), validated against analytic Fourier pairs;
* exact SI constants, the exception hierarchy and diagnostics (`optobuild.core`);
* the component *interface* (`optobuild.components.base`, `.spec`);
* architecture tests enforcing the layer structure.

## Install and test

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Documentation

Start with [docs/architecture.md](docs/architecture.md), then
[docs/signal_model.md](docs/signal_model.md),
[docs/numerical_conventions.md](docs/numerical_conventions.md),
[docs/component_api.md](docs/component_api.md),
[docs/physics_models.md](docs/physics_models.md),
[docs/testing_strategy.md](docs/testing_strategy.md),
[docs/requirements.md](docs/requirements.md),
[docs/roadmap.md](docs/roadmap.md) and the
[architecture decision records](docs/decisions/README.md).

## License

MIT — see [LICENSE](LICENSE).
