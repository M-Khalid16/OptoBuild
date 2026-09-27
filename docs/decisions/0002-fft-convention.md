# ADR-0002: FFT and spectral-density convention

* Status: Accepted
* Date: 2026-09-27

## Context
Sign and scaling errors in Fourier transforms are among the most common
simulator bugs (wrong-direction dispersion, spectra mirrored in wavelength,
PSDs off by N or dt). FR-003 requires one convention.

## Decision
* Continuous forward kernel `exp(−i 2π f t)`, inverse `exp(+i 2π f t)`,
  ordinary frequency `f` [Hz]; angular `ω = 2π f` where physics uses it.
* Discrete: `X = dt · fft(x) · exp(−i 2π f t0)`, `x = ifft(X · exp(+i 2π f t0)) / dt`.
* Frequency grid `fftfreq(N, dt)`, **FFT order in storage**; `fftshift` only
  for display.
* Parseval: `Σ|x|² dt = Σ|X|² df`. Two-sided envelope PSD `S = |X|²/T`;
  one-sided electrical PSD doubles positive-frequency bins.
* White noise: complex two-sided `N0` → per-sample variance `N0 fs`;
  real one-sided `G` → `G fs/2`.
* Implemented once in `optobuild.numerics.fft` (on `scipy.fft`); all other
  code must use it.

## Alternatives considered
* *Physics convention (`exp(−iω0 t)` carrier, `exp(+iωt)` forward kernel,
  Agrawal)*: common in nonlinear optics but opposite to NumPy/SciPy and to
  communications texts; would require sign flips at every FFT call. Formulas
  from that literature are converted once, in the physics function, with the
  conversion documented (numerical_conventions.md §2).
* *Unscaled DFT everywhere*: forces every caller to remember dt/N factors.
* *Storing shifted spectra*: error-prone with odd N and with transfer
  functions; display-only shifting is safer.

## Consequences
* Positive baseband frequency ↔ higher optical frequency ↔ shorter wavelength.
* Dispersion operator is `exp(−i(β2 ω²/2 + β3 ω³/6) L)` in our convention.
* Conventions are pinned by unit and analytic-validation tests.
