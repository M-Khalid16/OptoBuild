# OptoBuild

A scientifically validated optical and photonics simulation platform, built
incrementally: foundation → first validated optical link → GUI → SSFM → FSO →
coherent systems → photonic circuits → lasers → ultrafast → optimization/HPC.

**Current status: Phase 2 complete (v0.2.0)** — the first validated
end-to-end optical link:

```
PRBS -> NRZ -> Mach-Zehnder modulator <- CW laser
MZM -> linear fiber -> PIN photodiode -> low-pass filter -> decision -> BER analyzer
taps: optical power meters, optical spectrum analyzer, eye diagram
```

Every model is documented with equations, SI units, assumptions and limits in
[docs/physics_models.md](docs/physics_models.md) and validated against
analytical references (CW normalization, MZM transfer, fiber attenuation,
Gaussian-pulse dispersion with the exact complex field, PIN current and noise,
filter responses, deterministic BER counting, BER of the whole link vs.
Gaussian theory, seed reproducibility).

Framework (Phase 1): immutable signals, typed component ports and parameter
schemas, registry, graph validation, deterministic topological execution with
caching/progress/cancellation, per-component seeded RNG, strict JSON/YAML
projects, CLI.

## Quick start

```bash
pip install -e ".[dev]"
optobuild components                      # list components
optobuild demo optical_link               # run the reference 10 Gb/s link
python examples/optical_link.py           # same, with a readable report + data export
optobuild run examples/optical_link.json  # run a saved project
optobuild run examples/optical_link.json --save-results out.h5   # HDF5 (needs h5py)
optobuild ber demo:optical_link --trials 20                      # Monte Carlo BER
```

Since v0.3.0: a global simulation layout (bit rate, pattern length, samples
per bit) drives all sources, results can be stored in HDF5 together with the
project that produced them, and BER can be accumulated over independent noise
trials with exact confidence intervals.

## Tests

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
