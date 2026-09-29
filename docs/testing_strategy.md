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
| CW power normalization | `mean(abs(A)²) = P0` | ✅ `tests/validation/test_sources.py` |
| MZM transfer curve, ER, quadrature point | closed form (physics_models §3.4) | ✅ `test_modulation.py` |
| Fiber attenuation | `P0 exp(−αL)` | ✅ `test_fiber.py` |
| Gaussian dispersive broadening (full complex field) | `T1/T0 = sqrt(1+(L/L_D)²)` | ✅ `test_fiber.py` |
| PIN mean current | `I = R P + I_d` | ✅ `test_detection.py` |
| PIN noise variance | `(G_shot + G_th) B_eq` | ✅ `test_detection.py` |
| Filter −3 dB point, NEB, impulse response | analytic | ✅ `test_filters.py` |
| BER counting | injected errors | ✅ `test_ber.py` |
| BER vs. Gaussian theory (whole link, incl. ISI) | `Σ ½ erfc(abs(m_k−th)/(√2σ))` | ✅ `test_link_ber.py` |
| Reproducibility | identical seeds ⇒ identical arrays; different component names ⇒ independent streams | ✅ `tests/integration/test_reproducibility.py`, `test_optical_link.py` |
| Monte Carlo BER (independent trials) | `n_trials × Σ ½ erfc(…)` | ✅ `test_monte_carlo_ber.py` |
| HDF5 results round trip | bit-exact signals and results | ✅ `tests/integration/test_results_hdf5.py` |
| SSFM: linear limit, exact SPM, SPM sign, solitons (N = 1, N = 2 closed form), order 2, energy | analytic | ✅ `tests/validation/test_ssfm.py` |
| FSO: aperture collection, Kim/Koschmieder, rain, Rytov, log-normal and Gamma-Gamma statistics, channel mean/outage vs Monte Carlo | closed form, independent quadrature, Monte Carlo | ✅ `tests/validation/test_fso_physics.py`, `tests/integration/test_fso_component.py` |
| Polarization: SU(2) elements, Haar isotropy, DGD centroids and eigenanalysis, PMD second moment and random flight | exact, Monte Carlo, independent simulation | ✅ `tests/validation/test_polarization.py` |
| Dual-pol DSP and link: CMA/RDE vs known channels, GSOP, joint FOE; DP SNR/BER vs OSNR theory with PMD, CD, LO offset; impairment compensation | exact AWGN, closed forms | ✅ `tests/validation/test_dual_pol_dsp.py`, `test_dp_coherent_link.py` |
| Studies: exact sweep values, Monte Carlo statistics vs noise model, parallel = serial, optimizer vs analytic optima; reports (escaping, self-contained); plugins | closed forms, statistics | ✅ `tests/integration/test_sweeps.py`, `tests/validation/test_optimization.py`, `tests/integration/test_reports.py`, `test_plugins.py` |
| Ultrafast: GNLSE solitons, shock solution, photon number, SSFS, RK4IP order; pulse metrics; exact Gaussian cavity fixed point | closed forms, conservation laws | ✅ `tests/validation/test_gnlse.py`, `test_pulses.py`, `test_cavity.py`, `tests/integration/test_ultrafast_components.py` |
| Lasers: rate-equation closed forms, turn-on, relaxation, chirp identity, RK4 order, Langevin RIN/FM spectra vs linear response (Henry); EDFA vs Saleh–Jopson, quantum-limited NF, ring laser closed forms | closed forms, independent method, Monte Carlo | ✅ `tests/validation/test_semiconductor_laser.py`, `test_edfa.py`, `tests/integration/test_laser_components.py` |
| Photonic circuits: rings/MZI closed forms vs S-matrix solver, reciprocity, unitarity, FSR/FWHM/Q/delay, CMT vs TMM (uniform and π-shifted gratings), quarter-wave stacks | closed forms, independent method (TMM) | ✅ `tests/validation/test_integrated_optics.py`, `tests/integration/test_photonic_components.py` |
| Coherent: constellations, RC/RRC zero ISI, IQ modulator, hybrid receiver noise, phase noise, OSNR, FOE/BPS/timing, end-to-end SNR/BER/SER vs theory | exact AWGN, closed forms, Monte Carlo | ✅ `tests/validation/test_constellations.py`, `test_pulse_shaping.py`, `test_coherent_physics.py`, `test_coherent_dsp.py`, `test_coherent_link.py` |

GUI tests: Qt-free models in `tests/unit/test_gui_*.py`; the Qt views run
headless (`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py`) in
`tests/integration/test_gui_qt.py`, including real mouse drags; they are
skipped when PySide6/pyqtgraph are not installed
(`pip install -e ".[dev,gui]"`).

## 4. Commands

```bash
pip install -e ".[dev]"
pytest                    # full suite
pytest -m validation      # analytical validation only
pytest -m "not slow"      # skip long tests
ruff check . && ruff format --check .
```
