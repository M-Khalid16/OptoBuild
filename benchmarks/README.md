# Benchmarks

Performance measurements live here, outside `tests/`, so they are never part
of the correctness suite. Numba or other native acceleration may be introduced
only after a benchmark in this directory demonstrates the need (ADR-0007,
ADR-0021).

* `bench_hotpaths.py` — per-unit cost of the computational hot paths and the
  speedup of process-parallel sweeps/Monte Carlo. Results of the reference
  machine: [RESULTS.md](RESULTS.md).

Run: `python benchmarks/bench_hotpaths.py`
