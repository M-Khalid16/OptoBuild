# Physics models

Status: **no physics model is implemented yet.** This document defines the
mandatory documentation template and records the *planned* Phase 2 models so
that their equations, units and conventions are agreed before code is written.
Everything in §3 is a specification, not a claim about existing code.

All formulas use the conventions of [numerical_conventions.md](numerical_conventions.md):
complex envelope `A` in sqrt(W), `E = Re{A exp(+i 2π f_ref t)}`, forward FT
kernel `exp(−i 2π f t)`, SI units.

## 1. Mandatory template

Every implemented model gets a section with:

1. **Equation(s)**
2. **Symbols** with **SI units**
3. **Assumptions**
4. **Range of applicability**
5. **Numerical implementation** (function name, discretization, complexity)
6. **Known limitations**
7. **References**
8. **Validation** (test file and analytical reference, tolerance and why)

A model is not "implemented" until all eight items exist.

## 2. Model index

| Model | Phase | Status | Module (planned) |
|---|---|---|---|
| CW laser (ideal) | 2 | planned | `physics/sources` |
| PRBS (LFSR) | 2 | planned | `physics/sources` (digital) |
| NRZ waveform | 2 | planned | `physics/modulation` |
| Mach-Zehnder modulator | 2 | planned | `physics/modulation` |
| Fiber: attenuation, delay, chromatic dispersion | 2 | planned | `physics/fiber` |
| PIN photodiode with shot/thermal/dark noise | 2 | planned | `physics/detection`, `physics/noise` |
| Electrical low-pass filters | 2 | planned | `physics/…` / `numerics` filters |
| Decision circuit, BER counting | 2 | planned | `analysis/ber` |
| Q-factor / Gaussian BER reference | 2 | planned | `analysis/qfactor` |
| Optical power, spectrum, eye diagram | 2 | planned | `analysis/…` |
| SSFM / NLSE | 4 | not started | `solvers/ssfm` |
| Atmospheric channel, turbulence | 5 | not started | `physics/atmospheric` |
| Coherent receiver / DSP | 6 | not started | — |
| Photonic circuit elements | 7 | not started | `physics/photonics` |
| Laser rate equations, fiber lasers | 8 | not started | `physics/laser` |
| Ultrafast / mode-locked cavities | 9 | not started | `solvers/cavity` |

## 3. Planned Phase 2 models (specification)

### 3.1 CW laser (ideal)

`A_p(t) = sqrt(P0) · exp(i φ0) · j_p`, `j` = unit Jones vector.
Symbols: `P0` [W] output power, `φ0` [rad] phase, `f_ref` [Hz] (or `λ0 = c/f_ref` [m]).
The laser frequency may be offset from the signal reference: `f_laser = f_ref + Δf`
gives `A ∝ exp(i 2π Δf t)`; Δf must satisfy |Δf| < fs/2 and should be a
multiple of df to avoid a window-edge phase discontinuity (diagnostic).
Assumptions: zero linewidth, no RIN (phase noise / RIN in Phase 8).
Validation: mean |A|² = P0 exactly; spectrum is a single bin with power P0.

### 3.2 PRBS

Fibonacci LFSR over GF(2); sequence length `2^m − 1`. Default polynomials per
ITU-T O.150 (e.g. PRBS7: x⁷+x⁶+1; PRBS15: x¹⁵+x¹⁴+1; PRBS23: x²³+x¹⁸+1;
PRBS31: x³¹+x²⁸+1). Deterministic given the initial state (non-zero).
Validation: period = 2^m − 1; exactly 2^(m−1) ones per period; comparison
with an independent bit-by-bit reference implementation.

### 3.3 NRZ waveform generator

`v(t) = V_0 + (V_1 − V_0) · Σ_k b_k · p(t − kT_b)`, `p` = unit rectangle of
width `T_b = 1/R_b`, optionally smoothed by a documented shaping filter
(e.g. Gaussian with specified 10–90 % rise time). Integer samples per symbol
required. Units: V. Validation: levels, transition timing, rise time vs.
analytic Gaussian-edge 10–90 % relation.

### 3.4 Mach-Zehnder modulator

Push-pull, dual-arm, chirp-free, with finite extinction ratio via arm
power imbalance:

```
A_out(t) = A_in(t) · sqrt(IL) · [ a1 · exp(+i Δφ/2) + a2 · exp(−i Δφ/2) ]
Δφ(t)    = π (V(t) + V_bias) / V_π  + φ_0
a1 = (1 + ε)/2,  a2 = (1 − ε)/2,     ε = 1/sqrt(ER)
```

Power transfer: `T = IL · [ cos²(Δφ/2) + ε² sin²(Δφ/2) ]`; maximum `IL`
(Δφ = 0), minimum `IL · ε²`, so the extinction ratio is exactly `ER`.
With `ε = 0`: `T = IL cos²(π (V + V_bias)/(2 V_π) + φ_0/2)` (ideal MZM).
The imbalanced form introduces residual chirp: `angle(a1 e^{iΔφ/2} + a2 e^{-iΔφ/2}) = arctan(ε tan(Δφ/2))`;
this is documented, not hidden. Symbols: `V_π` [V], `V_bias` [V], `IL` (linear power
transmission ≤ 1, displayed in dB), `ER` (linear ≥ 1, displayed in dB), `φ_0` [rad].
Validation: sampled transfer curve vs. the closed form; quadrature bias point
`T = IL(1+ε²)/2`; ER measured = ER specified.

### 3.5 Optical fiber (linear)

```
Ã(L, ω) = Ã(0, ω) · exp(−α L / 2) · exp( −i (β2/2 ω² + β3/6 ω³) L )
t0 → t0 + β1 L        (group delay, retarded frame)
```

Symbols: `L` [m], `α` [1/m] (from dB/km), `β1 = n_g/c` [s/m], `β2` [s²/m]
(from `D` [s/m²]: `β2 = −D λ²/(2π c)`), `β3` [s³/m] (from slope `S`).
Assumptions: linear, single mode, scalar (no PMD), frequency-independent loss,
no nonlinearity (SSFM in Phase 4).
Validation:
* attenuation `P(L) = P(0) exp(−α L)`;
* unchirped Gaussian `|A(0,T)|² = P0 exp(−T²/T0²)` broadens as
  `T1/T0 = sqrt(1 + (L/L_D)²)`, `L_D = T0²/|β2|`, peak power reduced by the same factor
  (Agrawal, *Nonlinear Fiber Optics*, ch. 3);
* energy conservation when α = 0; group-delay bookkeeping.
Diagnostics: window wrap-around (§6 of numerical conventions).

### 3.6 PIN photodiode

```
I(t) = R · P(t) + I_d + n_shot(t) + n_th(t)
one-sided PSDs:  G_shot = 2 q (R P̄ + I_d)  [A²/Hz]   (evaluated per sample with P(t) in Phase 2+)
                 G_th   = 4 k_B T_K / R_L      [A²/Hz]
```

Symbols: `R` [A/W], `I_d` [A], `q` [C], `k_B` [J/K], `T_K` [K], `R_L` [Ω].
Realized noise with per-sample variance `G fs / 2` (numerical conventions §4);
the receiver filter sets the effective noise bandwidth.
Assumptions: square-law detection summed over polarizations, flat responsivity,
Gaussian approximation of shot noise (valid for many photoelectrons per sample).
Detector bandwidth is modelled by the following electrical filter.
Validation: noiseless mean `I = R P + I_d`; noise variance after an ideal
filter of bandwidth B equals `(G_shot + G_th) B` within statistical tolerance
set from the chi-square distribution of the sample variance.

### 3.7 Electrical low-pass filters

Planned: Bessel-Thomson (order n, 3-dB bandwidth), Butterworth, Gaussian,
ideal rectangular, raised cosine. Implemented as frequency responses on the
grid (`numerics.fft.apply_transfer_function`); analog prototypes from
`scipy.signal` evaluated with `freqs`, normalized to the specified −3 dB
bandwidth. Validation: |H| at DC = 1 and at f_3dB = 1/√2; Gaussian impulse
response vs. analytic; group delay of Bessel flat within spec; noise-equivalent
bandwidth vs. analytic.

### 3.8 Decision circuit and BER

Sample at `t_s = t_0 + (k + φ_s) T_b`, compare with threshold `V_th`, count
mismatches against the reference sequence after deterministic delay alignment
(cross-correlation, reported). BER = `N_err / N_bits`; report N_bits, N_err
and the 95 % upper bound when N_err = 0 (≈ 3/N_bits).
Reference: Gaussian-noise OOK with levels `μ1, μ0`, std `σ1, σ0`:
`Q = (μ1 − μ0)/(σ1 + σ0)`, `BER = ½ erfc(Q/√2)` at optimum threshold
(Agrawal, *Fiber-Optic Communication Systems*, ch. 4).
Validation: deterministic injected-error count; Monte Carlo AWGN BER within
binomial confidence of the analytic value.

## 4. References

* G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press, 2019.
* G. P. Agrawal, *Fiber-Optic Communication Systems*, 5th ed., Wiley, 2021.
* B. E. A. Saleh, M. C. Teich, *Fundamentals of Photonics*, 3rd ed., Wiley, 2019.
* J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
* ITU-T Rec. O.150, *General requirements for instrumentation for performance measurements on digital transmission equipment*.
* ITU-T Rec. G.652, *Characteristics of a single-mode optical fibre and cable*.
* E. Tiesinga et al., "CODATA recommended values of the fundamental physical constants: 2018," *Rev. Mod. Phys.* 93, 025010 (2021).
