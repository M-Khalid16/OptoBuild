# Testing strategy

## 1. Test levels

| Directory | Marker | Purpose | Example |
|---|---|---|---|
| `tests/unit` | `unit` | One function/class in isolation, including error paths. | `TimeGrid` rejects `dt <= 0` |
| `tests/integration` | `integration` | Connected subsystems; architecture rules. | layer-import rule; graph → engine → signals |
| `tests/validation` | `validation` | Comparison with **analytical or independent reference** solutions. | FFT of a Gaussian vs. closed form |
| `tests/regression` | `regression` | Protect behaviour already established by a validation test. | stored eye/BER data of the reference link |
| `benchmarks/` | — (not collected) | Performance; gate for acceleration decisions. | FFT size scaling |

Markers are applied automatically from the directory (`tests/conftest.py`), so
`pytest -m validation` runs all analytical checks. Warnings are errors
(`filterwarnings = error`) so numerical warnings (overflow, invalid value)
cannot pass unnoticed.

## 2. Rules

1. **Tests accompany code.** No module without tests in the same change.
2. **Tolerances are justified, never tuned.** Each validation test states the
   expected error source (discretization, truncation, statistics, round-off)
   and chooses a tolerance from it; the justification is in the docstring.
3. **Statistical tests** use fixed seeds *and* a tolerance derived from the
   estimator's distribution (e.g. binomial confidence for BER, chi-square for
   variance), so they would pass for almost every seed — the seed makes them
   deterministic, not correct.
4. **Deterministic seeds**: tests use explicit seeds; `numpy.random` global
   state is never used.
5. **Regression data** may be added only after the corresponding validation
   test passes, with generating script, seed and version recorded
   (`tests/regression/README.md`). Formats: `.npz`/JSON, never pickle.
6. **Conventions are tested**, not just documented: FFT sign, scaling,
   Parseval, PSD normalization, grid order.
7. **Architecture is tested**: import layering and "no physics in GUI".
8. **Error messages** are tested for the presence of the offending value and
   a hint where practical.

## 3. Validation catalogue

| Area | Reference | Status |
|---|---|---|
| FFT scaling vs. direct DFT sum | definition | ✅ `tests/unit/test_fft_conventions.py` |
| FFT sign (positive tone → positive bin) | definition | ✅ |
| Parseval energy | theorem | ✅ |
| PSD integrates to mean power; CW in DC bin | definition | ✅ |
| Gaussian Fourier pair; shift theorem; analytic energy | closed form | ✅ `tests/validation/test_fft_analytic_pairs.py` |
| CW power normalization | `mean|A|² = P0` | Phase 2 |
| MZM transfer curve, ER, quadrature point | closed form (physics_models §3.4) | Phase 2 |
| Fiber attenuation | `P0 exp(−αL)` | Phase 2 |
| Gaussian dispersive broadening | `T1/T0 = sqrt(1+(L/L_D)²)` | Phase 2 |
| PIN mean current | `I = R P + I_d` | Phase 2 |
| PIN noise variance | `(G_shot + G_th) B_eq` | Phase 2 |
| Filter −3 dB point, NEB | analytic | Phase 2 |
| BER counting | injected errors | Phase 2 |
| BER vs. AWGN theory | `½ erfc(Q/√2)` | Phase 2 |
| Reproducibility | identical seeds ⇒ identical arrays; different component names ⇒ independent streams | Phase 1 |

## 4. Commands

```bash
pip install -e ".[dev]"
pytest                    # full suite
pytest -m validation      # analytical validation only
pytest -m "not slow"      # skip long tests
ruff check . && ruff format --check .
```
