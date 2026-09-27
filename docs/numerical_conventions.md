# Numerical conventions

Status: **normative; implemented** in `optobuild.numerics.grid` and
`optobuild.numerics.fft`, and pinned by `tests/unit/test_fft_conventions.py`,
`tests/unit/test_grid.py` and `tests/validation/test_fft_analytic_pairs.py`.
Decision record: ADR-0002.

## 1. Sampling grid

```
t_n  = t0 + n dt,   n = 0, ..., N-1         [s]
fs   = 1/dt                                  [Hz]
T    = N dt          (window duration)       [s]
df   = 1/T = fs/N                            [Hz]
f_k  = fftfreq(N, dt)                        [Hz]   (FFT order)
w_k  = 2 pi f_k                              [rad/s]
```

* **FFT order** (`f_0 = 0`, positive frequencies, then negative) is the
  storage order for every spectral array. `fftshift` is applied **only** for
  display (`numerics.fft.to_display_order`, `TimeGrid.frequency(shifted=True)`).
* For even N the Nyquist bin `-fs/2` appears once, at index N/2, with a
  negative sign (NumPy convention). Odd N is supported.
* Simulated signals are **periodic** with period T. Every linear filter is a
  circular convolution (see §6).

## 2. Fourier-transform convention

Continuous transform (engineering sign, ordinary frequency):

```
X(f) = ∫ x(t) exp(-i 2π f t) dt           x(t) = ∫ X(f) exp(+i 2π f t) df
```

Discrete implementation (`numerics.fft.spectrum` / `inverse_spectrum`):

```
X_k = dt · FFT{x}_k · exp(-i 2π f_k t0)
x_n = IFFT{ X · exp(+i 2π f t0) }_n / dt
```

where FFT/IFFT are the NumPy/SciPy unnormalized-forward, 1/N-inverse DFTs.
The `exp(∓i 2π f t0)` factor makes `X_k` approximate the continuous transform
with the *true* time origin, so analytic transforms can be compared directly
(validated against the Gaussian pair and the shift theorem).
For linear filtering (`apply_transfer_function`) the origin phase cancels.

Units: if `x` is in sqrt(W), `X` is in sqrt(W)·s.

### Relation to the optical field

With `E(t) = Re{A(t) exp(+i 2π f_ref t)}`, a component `exp(+i 2π f t)` of the
envelope oscillates at absolute frequency `f_ref + f`. Hence:

* positive baseband frequency ⇒ higher optical frequency ⇒ shorter wavelength;
* a spectrum plotted versus `f_ref + f_k` needs **no** sign flip.

### Relation to Agrawal's (physics) convention

Much of the fiber literature (e.g. G. P. Agrawal, *Nonlinear Fiber Optics*)
uses `E = Re{A exp(-i ω0 t)}` with forward kernel `exp(+i ω t)`. Converting an
equation from that convention to ours flips the sign of `i` in
frequency-domain operators. Example (linear dispersion, retarded frame):

```
ours:     Ã(z, ω) = Ã(0, ω) · exp( -i (β2/2 ω² + β3/6 ω³) z )
Agrawal:  Ã(z, ω) = Ã(0, ω) · exp( +i (β2/2 ω² + β3/6 ω³) z )
```

The Kerr term flips the same way: ours `−iγ|A|²A`, Agrawal `+iγ|A|²A`
(validated against the conjugate of his closed-form N = 2 soliton).

Derivation (ours): a monochromatic component propagates as
`exp(i(ω_a t − β(ω_a) z))`; expanding `β` about `ω0` and moving to
`T = t − β1 z` leaves `exp(−i(β2 ω²/2 + β3 ω³/6) z)`. Both forms predict the
same intensity evolution; chirp signs differ, so every physics function
states which convention its formula is written in (always ours in code).

## 3. Energy and power (Parseval)

Exact for the discrete pair above (validated to 1e-12):

```
E = Σ_n |x_n|² dt = Σ_k |X_k|² df                [J]  (x in sqrt(W))
P_avg = E / T = mean_n |x_n|²                    [W]
```

For polarization-resolved fields, sum additionally over the polarization axis.

## 4. Spectral-density conventions

**Optical / complex-envelope signals — two-sided baseband PSD**

```
S_k = |X_k|² / T                                 [W/Hz]
Σ_k S_k df = P_avg
power per bin: S_k df = |FFT{x}_k / N|²          [W]
```

This is the periodogram of one realization with a rectangular window (no
averaging), `numerics.fft.power_spectral_density`. "Two-sided" here means over
the full baseband span `[-fs/2, fs/2)`, which for an optical envelope is the
entire optical spectrum around `f_ref` (it is not symmetric).

**Optical spectrum analyzer display (Phase 2)** integrates `S` over a
resolution bandwidth `RBW` (e.g. 0.1 nm ↔ `Δν = c Δλ / λ²`) and reports
power per RBW, typically in dBm. The RBW used must be reported with the plot.
Wavelength axis: `λ_k = c / (f_ref + f_k)` (non-uniform, decreasing with k).

**Electrical (real) signals — one-sided PSD**

```
G_k = 2 S_k  for 0 < f_k < fs/2 ;  G_0 = S_0 ;  G_{N/2} = S_{N/2} (even N)
Σ_{f_k >= 0} G_k df = mean x²                     [V²/Hz → V²] or [A²/Hz → A²]
```

**White-noise generation**

| Noise | Specification | Per-sample variance |
|---|---|---|
| complex optical white noise | two-sided PSD `N0` [W/Hz] over the full band `fs` | `σ² = N0 fs` (σ²/2 per quadrature) |
| real electrical white noise | one-sided PSD `G` [A²/Hz] | `σ² = G fs / 2` |

So e.g. thermal current noise with one-sided PSD `4 k_B T_K / R_L` over the
simulation bandwidth `fs/2` has `σ² = (4 k_B T_K / R_L)(fs/2)`; the actual
noise bandwidth is then set by the receiver filter that follows.

## 5. dB conventions

* Power ratios: `dB = 10 log10(P2/P1)`; field/voltage ratios `20 log10`.
* `dBm = 10 log10(P / 1 mW)`.
* Attenuation coefficient: `α [1/m] = α_dB/km · ln(10) / 10 / 1000`
  (power attenuation, `P(L) = P(0) e^{-αL}`); field attenuation is `α/2`.
* Dispersion: `β2 = −D λ² / (2π c)` with `D` in s/m² (1 ps/(nm·km) = 1e-6 s/m²).

All conversions are implemented once in `core.units` (Phase 1).

## 6. Numerical limitations and required diagnostics

These are inherent to the representation and must be reported, not hidden.
Implemented diagnostics (Phase 2): `sampling.aliasing_risk` (NRZ generator,
MZM output), `sampling.window_wraparound` (fiber), `sampling.samples_per_symbol`
(NRZ generator), `sampling.pattern_periodicity` (PRBS), `laser.offset_not_periodic`,
`filter.bandwidth_above_nyquist`, `ber.low_error_count`,
`ber.alignment_unreliable`; inconsistent grids raise `SamplingError`.

Known limitation of the aliasing heuristic: it measures energy in the outer
10 % of the band. Harmonics of a strictly periodic waveform that fold exactly
onto lower bins (e.g. a 0101… pattern at 4 samples/bit) are not detected.
Random-like patterns (PRBS) are detected reliably.

| Limitation | Cause | Diagnostic (Phase 1/2) |
|---|---|---|
| **Aliasing** | Content beyond ±fs/2 folds back. Nonlinear operations (e.g. `|A|²`, MZM `cos`) generate harmonics. | `sampling.aliasing_risk`: fraction of spectral energy in the outer 10 % of the band above a threshold (default 1e-6 of total). |
| **Window wrap-around** | Circular convolution: dispersion-broadened pulses or filter impulse responses longer than the guard interval wrap to the other window edge. | `sampling.window_wraparound`: compare accumulated dispersive spread `|β2| L · 2π B` (B = signal bandwidth) and filter impulse-response length with the window and with the pattern period. |
| **Finite frequency resolution** | `df = 1/T`. Narrow features (laser linewidth, ring resonances) below `df` are unresolved. | `sampling.resolution`: feature width < 5 df. |
| **Periodicity of patterns** | A PRBS of length `2^m − 1` must fit an integer number of times in the window, otherwise the wrap creates a spurious transition. | `sampling.pattern_periodicity`. |
| **Too few samples per symbol** | Filters/eye diagrams need several samples per symbol. | `sampling.samples_per_symbol` (< 4 → warning, < 2 → error). |
| **Inconsistent grids** | Combining signals with different dt/N/f_ref. | `SamplingError` (fatal). |
| **Round-off** | float64; FFT round-trip error ~1e-15 relative. | Tolerances in tests reflect this (≥ 1e-12). |
| **Statistical BER floor** | With `N_bits` counted bits and zero errors, BER < ~3/N_bits (95 % confidence) is all that can be stated. | BER analyzer reports N_bits, N_errors and the confidence bound. |

## 7. Data types

* Sampled optical fields: `complex128`; electrical: `float64`; bits: `uint8`.
* `complex64`/`float32` may be offered later as an explicit performance option
  after accuracy studies (ADR-0007); never silently.
