# Physics models

Status: **Phase 2, 4, 5, 6 and 6b models implemented and validated** (v0.8.0). Later-phase
models are listed in §2 as *not started*; nothing here describes code that
does not exist.

All formulas use the conventions of [numerical_conventions.md](numerical_conventions.md):
complex envelope `A` in sqrt(W), `E = Re{A exp(+i 2π f_ref t)}`, forward FT
kernel `exp(−i 2π f t)`, SI units. Equations are implemented once, in the
module named in each section; components only call these functions.

## 1. Mandatory template

Every implemented model has:

1. **Equation(s)**
2. **Symbols** with **SI units**
3. **Assumptions**
4. **Range of applicability**
5. **Numerical implementation** (function name, discretization)
6. **Known limitations**
7. **References**
8. **Validation** (test file, analytical reference, tolerance rationale)

## 2. Model index

| Model | Phase | Status | Implementation | Component |
|---|---|---|---|---|
| CW laser (ideal) | 2 | ✅ validated | `physics.sources` | `optobuild.source.cw_laser` |
| PRBS (maximal-length LFSR) | 2 | ✅ validated | `physics.prbs` | `optobuild.source.prbs` |
| NRZ waveform | 2 | ✅ validated | `physics.modulation` | `optobuild.modulator.nrz_generator` |
| Mach-Zehnder modulator | 2 | ✅ validated | `physics.modulation` | `optobuild.modulator.mzm` |
| Linear fiber: loss, delay, β2, β3 | 2 | ✅ validated | `physics.fiber` | `optobuild.channel.linear_fiber` |
| PIN photodiode, shot/thermal/dark noise | 2 | ✅ validated | `physics.detection`, `physics.noise` | `optobuild.detector.pin` |
| Electrical low-pass filters | 2 | ✅ validated | `numerics.filters` | `optobuild.electrical.lowpass_filter` |
| Decision circuit | 2 | ✅ validated | `analysis.decision` | `optobuild.electrical.decision` |
| BER counting, confidence, Gaussian theory | 2 | ✅ validated | `analysis.ber` | `optobuild.analyzer.ber` |
| Optical power, spectrum, eye data | 2 | ✅ validated | `analysis.power/spectrum/eye` | `optobuild.analyzer.*` |
| Optical pulse source (Gaussian, sech) | 4 | ✅ validated | `physics.sources` | `optobuild.source.optical_pulse` |
| Nonlinear fiber: NLSE (SPM, β2, β3, loss), SSFM | 4 | ✅ validated | `physics.nonlinear`, `solvers.ssfm` | `optobuild.channel.nonlinear_fiber` |
| FSO geometry, pointing, beam wander | 5 | ✅ validated | `physics.free_space` | `optobuild.channel.fso` |
| Atmospheric attenuation (Kim/Kruse, rain) | 5 | ✅ validated | `physics.atmospheric` | `optobuild.channel.fso` |
| Turbulence (Rytov, log-normal, Gamma-Gamma) | 5 | ✅ validated | `physics.turbulence` | `optobuild.channel.fso` |
| FSO channel gain, outage, link budget | 5 | ✅ validated | `physics.fso_channel`, `analysis.link_budget` | `optobuild.channel.fso`, `optobuild fso-budget` |
| Constellations, RC/RRC shaping | 6 | ✅ validated | `analysis.constellations`, `numerics.pulse_shaping` | `optobuild.dsp.symbol_mapper`, `optobuild.dsp.pulse_shaper` |
| IQ modulator, laser phase noise, ASE/OSNR | 6 | ✅ validated | `physics.modulation`, `physics.sources`, `physics.noise` | `optobuild.modulator.iq`, `optobuild.source.cw_laser`, `optobuild.amplifier.ase_noise_loader` |
| Coherent receiver (hybrid + balanced PDs) | 6 | ✅ validated | `physics.detection` | `optobuild.detector.coherent_receiver` |
| Coherent DSP, EVM/SNR/BER analysis | 6 | ✅ validated | `analysis.coherent_dsp` | `optobuild.dsp.coherent_dsp`, `optobuild.analyzer.coherent` |
| Dual polarization, MIMO equalizer, PMD | 6b | not started | — | — |
| Photonic circuit elements | 7 | not started | — | — |
| Laser rate equations, fiber lasers | 8 | not started | — | — |
| Ultrafast / mode-locked cavities | 9 | not started | — | — |

## 3. Phase 2 models

### 3.1 CW laser (ideal) — `physics.sources.cw_field`

1. `A(t) = sqrt(P0) · exp(i (φ0 + 2π Δf t))`, single polarization.
2. `P0` [W], `φ0` [rad], `Δf` [Hz] detuning from `f_ref = c/λ0`, `λ0` [m].
3. Zero linewidth, no RIN, no chirp.
4. `P0 ≥ 0`; `|Δf| < fs/2` (enforced; error otherwise).
5. Closed form sampled on the grid.
6. `Δf` not a multiple of `df` → phase jump at the periodic window edge
   (warning `laser.offset_not_periodic`). Phase noise/RIN: Phase 8.
7. Saleh & Teich, ch. 16.
8. `tests/validation/test_sources.py`: `mean|A|² = P0` to 1e-14 (round-off);
   single spectral line with power `P0`; dBm round trip.

### 3.2 PRBS — `physics.prbs.prbs`

1. `s[n] = s[n−k] ⊕ s[n−m]` (Fibonacci LFSR, taps m, k of `x^m + x^k + 1`).
2. Bits (dimensionless), order `m`, period `2^m − 1`.
3. Primitive trinomials: PRBS7 (7,6), 9 (9,5), 10 (10,7), 11 (11,9),
   15 (15,14), 20 (20,3), 23 (23,18), 31 (31,28).
4. Any `n_bits ≥ 1`; seed ∈ [1, 2^m − 1].
5. Vectorized recurrence in blocks of k bits.
6. ITU-T O.150 inversion/bit-order conventions are **not** reproduced — the
   sequences are maximal-length with the O.150 polynomials but not
   bit-identical to O.150 test instruments.
7. ITU-T O.150; S. W. Golomb, *Shift Register Sequences*.
8. `test_sources.py`: minimal period `2^m − 1` (checked against every prime
   divisor), balance `2^(m−1)` ones, bit-exact agreement with an independent
   bit-by-bit LFSR for all orders.

### 3.3 NRZ waveform — `physics.modulation.nrz_waveform`

1. `v(t) = V0 + (V1 − V0) Σ b_k p(t − kT_b)`, optionally convolved with a
   unit-area Gaussian `h(t)`; `H(f) = exp(−2π²σ²f²)`, `σ = t_r / (2 Φ⁻¹(0.9))`.
2. `V0, V1` [V], `T_b = 1/R_b` [s], `t_r` 10–90 % rise time [s], `sps` samples/bit.
3. Ideal driver; Gaussian edge shape; pattern periodic in the window.
4. `sps ≥ 2` (warning below 4).
5. `np.repeat` then frequency-domain Gaussian filter (circular).
6. Rectangular NRZ (`t_r = 0`) is not band-limited → aliasing warning.
7. Agrawal, *Fiber-Optic Communication Systems*, ch. 1.
8. `test_modulation.py`: exact levels/timing; measured 10–90 % rise time within
   1 % of `t_r` (interpolation error < 0.1 dt at 256 sps); 50 % crossing at the
   bit boundary.

### 3.4 Mach-Zehnder modulator — `physics.modulation.mzm_field_transfer`

1. `A_out = A_in sqrt(IL) [cos(Δφ/2) + i ε sin(Δφ/2)]`,
   `Δφ = π (V + V_bias)/V_π + φ0`, `ε = 1/sqrt(ER)`;
   `T = IL [cos²(Δφ/2) + ε² sin²(Δφ/2)]`.
2. `V_π, V_bias` [V], `φ0` [rad], `IL` ∈ (0, 1] linear, `ER ≥ 1` linear.
3. Push-pull, quasi-static (no electro-optic bandwidth), polarization-independent;
   finite ER from arm amplitude imbalance, which gives residual chirp
   `arctan(ε tan(Δφ/2))`.
4. Any drive voltage; the transfer is periodic in `2V_π`.
5. Pointwise closed form; output field = input field × transfer.
6. No drive-bandwidth limit (model it with a filter on the drive), no
   frequency-dependent V_π, no DC drift.
7. Agrawal, *FOCS*, sec. 3.4; Saleh & Teich, ch. 21.
8. `test_modulation.py`: sampled transfer and phase vs the closed form to
   1e-12 for several (IL, ER, bias, φ0); `cos²` identity for ε = 0; periodicity;
   `T_max/T_min = ER`; quadrature point `IL(1+1/ER)/2`; component delegates
   exactly to the physics function (`test_link_components.py`).

### 3.5 Linear fiber — `physics.fiber.propagate_linear`

1. `Ã(L,ω) = Ã(0,ω) e^{−αL/2} e^{−i(β2ω²/2 + β3ω³/6)L}`; `t0 → t0 + β1L`.
2. `L` [m]; `α` [1/m] (from dB/km); `β1 = n_g/c` [s/m]; `β2 = −Dλ²/(2πc)` [s²/m];
   `β3 = (λ/2πc)²(λ²S + 2λD)` [s³/m]; `D` [s/m²], `S` [s/m³] at the carrier λ.
3. Linear, scalar (no PMD), single mode, frequency-flat loss, dispersion to 3rd order.
4. Powers low enough that nonlinearity is negligible (not checked — Phase 4).
5. Exact multiplication in the frequency domain (`numerics.fft`).
6. Circular convolution: correct for patterns periodic in the window;
   dispersive spread > 25 % of the window → warning `sampling.window_wraparound`.
7. Agrawal, *Nonlinear Fiber Optics*, 6th ed., sec. 3.2–3.3 (signs converted).
8. `test_fiber.py`: `P(L)/P(0) = e^{−αL}` to 1e-12 and 0.2 dB/km × L exactly;
   energy conservation for α = 0; group delay in `t0`; complex Gaussian field
   vs the analytic solution `sqrt(P0)T0/sqrt(T0²+iβ2L)·exp(−T²/(2(T0²+iβ2L)))`
   to 1e-9 of the peak; RMS broadening `sqrt(1+(L/L_D)²)` to 1e-9; physical sign
   check (anomalous dispersion: higher frequency arrives earlier, delay
   `β2·2πΔf·L`); β3 group-delay curvature.

### 3.6 PIN photodiode — `physics.detection`, `physics.noise`

1. `I = R P(t) + I_d + n_shot + n_th`; one-sided PSDs `2q(RP(t)+I_d)` and
   `4k_B T/R_L`; per-sample variance `G fs/2`.
2. `R` [A/W], `I_d` [A], `T` [K], `R_L` [Ω], `q`, `k_B` exact SI constants.
3. Square law summed over polarizations; Gaussian shot noise using the
   instantaneous noiseless current; white noise over the simulated band;
   detector bandwidth modelled by the following filter.
4. Gaussian approximation needs many photoelectrons per sample (`I dt/q ≫ 1`).
5. Realized noise from the component's seeded generator (ADR-0008).
6. No APD gain, saturation, frequency-dependent responsivity, or 1/f noise.
7. Agrawal, *FOCS*, sec. 4.4; Saleh & Teich, sec. 19.5.
8. `test_detection.py`: noiseless `I = RP + I_d` to 1e-14; dual-polarization
   sum; shot and thermal variances and filtered variance `G·NEB` within 5
   standard deviations of the sample-variance estimator; reproducibility.

### 3.7 Electrical low-pass filters — `numerics.filters`

1. Gaussian `exp(−(ln2/2)(f/B)²)`; rectangular; Butterworth and Bessel-Thomson
   analog prototypes `H(i2πf)` (scipy.signal), all with −3 dB bandwidth `B`.
2. `B` [Hz], order `n`.
3. Linear time-invariant; applied circularly.
4. `B < fs/2` (warning otherwise).
5. `H(f_k)` on the grid; Nyquist bin made real for even N.
6. Causal filters delay the waveform (absorbed by decision timing).
7. Oppenheim & Willsky; Thomson (1949).
8. `test_filters.py`: `|H(0)| = 1`, `|H(B)| = 1/√2`, Hermitian symmetry;
   Butterworth magnitude; Gaussian impulse response vs analytic to 1e-9;
   NEB vs closed forms (Butterworth `B(π/2n)/sin(π/2n)`, Gaussian
   `B sqrt(π/(4 ln2))`); Bessel group-delay flatness < 1 % to 0.5B.

### 3.8 Decision circuit and BER — `analysis.decision`, `analysis.ber`

1. Decision samples `x[k·sps + offset]`, bit = `x > threshold`;
   `BER = N_err/N_bits` after circular alignment (FFT cross-correlation);
   Clopper–Pearson 95 % interval; Gaussian reference
   `p_k = ½ erfc(|m_k − th|/(√2 σ))`, `BER = ½ erfc(Q/√2)`.
2. Offset in samples, threshold in A or V, counts dimensionless.
3. Timing: max-variance (data-independent) or fixed phase; threshold: mean or fixed.
4. Alignment reliable for BER ≪ 0.5 (warning above 0.1).
5. Vectorized.
6. No clock jitter, no adaptive threshold; with `N_err = 0` only an upper
   bound can be stated (reported).
7. Agrawal, *FOCS*, sec. 4.5; Proakis & Salehi.
8. `test_ber.py`: injected errors counted exactly; alignment recovers circular
   shifts; confidence limits; Q reference values. `test_link_ber.py`: counted
   errors of the complete thermal-noise-limited link agree with the exact
   Gaussian expectation (from the noiseless run, including ISI) within
   5 Poisson standard deviations at BER ≈ 1.4e-3…9e-3 (observed: 45 vs 47.3,
   118 vs 123.5, 303 vs 297.1).

### 3.9 Measurements — `analysis.power`, `analysis.spectrum`, `analysis.eye`

* Power: `P_avg = mean Σ_p|A_p|²`, `P_peak = max`.
* Spectrum: `S_k = |X_k|²/T` summed over polarizations, `ν = f_ref + f_k`,
  `λ = c/ν`, power per rectangular RBW; Parseval `Σ S df = P_avg`. With a
  rectangular RBW, strong discrete lines (the carrier, and lines at multiples
  of the bit rate caused by e.g. the MZM's cos² nonlinearity) produce visible
  steps where they enter/leave the window; this is the correct response of a
  rectangular filter, not an artefact of the plot.
* Eye: traces of `n` bit periods at the simulated samples (no interpolation),
  centred on the max-variance instant; decision-directed Q and eye opening are
  labelled as such.
* Validation: `tests/unit/test_analysis.py`, `test_sources.py`,
  `tests/integration/test_optical_link.py` (TX spectrum total = TX power;
  RX/TX power ratio = `e^{−αL}`).

### 3.10 Nonlinear fiber (NLSE, SSFM) — `physics.nonlinear`, `solvers.ssfm`

1. `∂A/∂z = −α/2·A − i(β2/2·ω² + β3/6·ω³)A − iγ|A|²A` (our convention; Agrawal
   has `+iγ|A|²A`). Split step: `D(h/2) N(h) D(h/2)`,
   `N(h): A → A e^{−αh/2} exp(−iγ|A|² L_eff(h))`, `L_eff(h) = (1−e^{−αh})/α`.
2. `γ = 2π n2/(λ A_eff)` [1/(W m)] (≈ 1.3 /(W km) for SMF), `n2` [m²/W],
   `A_eff` [m²], `L_D = T0²/|β2|`, `L_NL = 1/(γP0)`, `N² = L_D/L_NL`.
3. Scalar field, instantaneous Kerr (SPM only), γ frequency-independent,
   β up to third order, frequency-flat loss.
4. Pulses ≳ 1 ps (no Raman/self-steepening); powers below SRS/SBS thresholds;
   step small enough that per-step nonlinear phase ≪ 1 rad.
5. Adaptive steps (peak phase per step ≤ `max_phase`, ≤ `max_step`) or fixed
   steps; each step = 2 FFT pairs of length N.
6. Splitting error O(h²); periodic window; aliasing if the spectrum
   broadens to ±fs/2 (warning `ssfm.spectral_truncation`); wrap-around for
   pulses (warning `sampling.window_wraparound`). Note that β3 derived from D
   is non-zero even for zero dispersion slope S.
7. Agrawal, *Nonlinear Fiber Optics*, 6th ed., ch. 2, 4, 5; Sinkin et al.,
   *J. Lightwave Technol.* 21, 61 (2003); Satsuma & Yajima, *Prog. Theor.
   Phys. Suppl.* 55, 284 (1974).
8. `tests/validation/test_ssfm.py`: γ = 0 equals the linear fiber (1e-12);
   pure SPM with loss exact (1e-12) for 1, 3, 50 steps; leading-edge red shift;
   fundamental soliton stationary with phase −γP0z/2 (1e-5); N = 2 complex field
   vs the Satsuma–Yajima solution at three distances (1e-5; observed ≤ 3.7e-6);
   N = 2 period; measured convergence order 2 ± 0.2; energy conservation for
   α = 0 (1e-12); step control. `tests/integration/test_nonlinear_components.py`.

### 3.11 Optical pulse source — `physics.sources.pulse_field`

Gaussian `sqrt(P0) exp(−(t−tc)²/(2T0²))` or sech `sqrt(P0) sech((t−tc)/T0)`,
centred in the window, unchirped. Energies `P0 T0 √π` and `2 P0 T0` are
verified numerically. Diagnostics: fewer than 3 samples per T0, pulse energy
at the window edges.

### 3.12 FSO geometry and pointing — `physics.free_space`

1. `w(L)² = w0² + (θL)²`, `θ_dl = λ/(πw0)`; collected fraction of a spot
   displaced by r: `h_p(r) = F_ncx2((a/σ)²; 2, (r/σ)²)`, `σ = w/2`
   (`= 1 − exp(−2a²/w²)` at r = 0); displacement Rice-distributed
   (static offset + Gaussian jitter/wander); beam wander
   `<r_c²> = 2.42 Cn² L³ w0^{−1/3}`.
2. w0, w, a, r [m]; θ [rad]; Cn² [m^−2/3].
3. Gaussian beam, paraxial, far field; aperture in the transverse plane;
   independent Gaussian jitter and wander.
4. Horizontal links, weak-turbulence wander formula (collimated beam, infinite outer scale).
5. Exact CDF (`scipy.special.chndtr`); means by 400-point Gauss–Legendre
   quadrature over the Rice density.
6. No angle-of-arrival, obscuration or focusing optics.
7. Farid & Hranilovic, JLT 25, 1702 (2007); Andrews & Phillips (2005) ch. 6;
   Kaushal & Kaddoum, IEEE COMST 19, 57 (2017).
8. `tests/validation/test_fso_physics.py`: Gaussian-beam formula (1e-12);
   r = 0 closed form (1e-12); offset fraction vs independent 2-D quadrature
   (1e-8); small/large-aperture limits; Farid–Hranilovic mean for a ≪ w (0.2 %).

### 3.13 Atmospheric attenuation — `physics.atmospheric`

1. `β = (3.912/V)(λ/550 nm)^{−q}` [1/m] with Kim or Kruse q(V);
   rain `1.076 R^{0.67}` dB/km (R in mm/h).
2. V [m] (display km), λ [m], R [m/s] (display mm/h).
3. Empirical fits; homogeneous path.
4. 0.5–2 µm; V from ~50 m to > 50 km; several dB/km uncertainty in dense fog.
5. Closed form.
6. No snow, no specific aerosol models, no absorption lines.
7. Kim et al., Proc. SPIE 4214 (2001); Kruse et al. (1962); Kaushal & Kaddoum (2017).
8. Koschmieder definition (2 % at 550 nm over V, exact); q regimes;
   V = 1 km at 1550 nm → 10.12 dB/km; dense fog wavelength-independent; rain formula.

### 3.14 Turbulence — `physics.turbulence`

1. `σ_R² = 1.23 Cn² k^{7/6} L^{11/6}`; log-normal `ln h ~ N(−s²/2, s²)`,
   `s² = ln(1+σ_I²)`; Gamma-Gamma with Al-Habash α, β and
   `σ_I² = 1/α + 1/β + 1/(αβ)`; aperture averaging `[1 + 1.062 kD²/(4L)]^{−7/6}`.
2. Cn² [m^−2/3], k [1/m], L [m], D [m]; h dimensionless, E[h] = 1.
3. Kolmogorov spectrum, constant Cn², plane wave, point receiver (GG).
4. Log-normal: σ_R² ≲ 1 (warning otherwise); GG: weak to strong.
5. GG sampling as product of unit-mean gammas; GG CDF by quantile
   quadrature (vectorized).
6. No inner/outer scale, no aperture averaging for GG, no temporal model.
7. Andrews & Phillips (2005) ch. 8–11; Al-Habash, Andrews & Phillips, Opt. Eng. 40, 1554 (2001).
8. Rytov scaling laws; aperture-averaging limits; GG pdf normalization,
   mean and second moment by independent quadrature (≤ 1e-7); CDF vs
   integrated pdf; weak-turbulence limit; samplers vs CDFs (5σ binomial).

### 3.15 FSO channel and link budget — `physics.fso_channel`, `analysis.link_budget`

1. `h = η_tx η_rx e^{−βL} h_p(r) h_t`; `P_out = P(h < h_th)`;
   margin `= P_rx,mean − S` [dB].
2. η linear; β [1/m]; powers [W]/[dBm].
3. Quasi-static channel (one state per trial); power-only (IM/DD).
4. Horizontal links; window ≪ coherence time (warning above 100 µs).
5. Outage: exact Rice survival without turbulence; otherwise quadrature over r.
6. See ADR-0015.
7. As above.
8. Mean gain and outage vs 3×10⁵ Monte Carlo samples for four channel
   configurations (5σ); component trial gains vs analytic mean/outage over
   3000 engine trials (`tests/integration/test_fso_component.py`).

### 3.16 Constellations and pulse shaping — `analysis.constellations`, `numerics.pulse_shaping`

1. Square M-QAM from two Gray PAM axes, unit E_s; exact AWGN
   `SER = 1 − (1 − P_L)²`, `P_L = 2(1 − 1/√M) Q(√(3 SNR/(M−1)))`; BPSK/QPSK BER.
   RC `H(f)` (raised cosine, roll-off β), RRC `√H`; TX `sps·H`, RX matched `H`.
2. SNR = E_s/N0; R_s [Bd]; β ∈ [0, 1].
3. Equiprobable symbols; ideal linear filters; periodic symbol pattern.
4. `(1+β)R_s/2 ≤ fs/2`.
5. Frequency-domain filters on the grid (β = 0 edge bin = ½).
6. No Gray BER formula for M ≥ 16 (count bits instead).
7. Proakis & Salehi, *Digital Communications*, 5th ed., sec. 4.3, 9.2.
8. `tests/validation/test_constellations.py`, `test_pulse_shaping.py`: Gray
   neighbours, unit energy, SER/BER vs Monte Carlo (5σ); exact zero ISI for
   β ∈ {0, 0.1, 0.5, 1}, sps ∈ {2, 4, 8}; folded-spectrum flatness.

### 3.17 IQ modulator, phase noise, ASE — `physics.modulation`, `physics.sources`, `physics.noise`

1. `A_out/A_in = √IL/2 [T(v_I) + i T(v_Q)]`, `T` = MZM at null
   (`−sin(πv/2V_π)` for infinite ER); Wiener phase with increments
   `N(0, 2πΔν dt)`; ASE per pol `N = P/(2 OSNR B_ref)`, complex white noise
   variance `N fs`; `SNR = 2 B_ref OSNR/(p R_s)`.
2. V [V], Δν [Hz], N [W/Hz], B_ref [Hz].
3. Ideal couplers and 90° phase; Lorentzian laser line; white ASE.
4. Linear regime for |v| ≪ V_π (the sine is part of the model).
5. Closed forms; seeded Gaussian draws.
6. No modulator bandwidth, bias drift, I/Q skew; phase noise not periodic in the window.
7. Seimetz (2009) ch. 4; Agrawal *FOCS* sec. 3.5; Essiambre et al., JLT 28, 662 (2010).
8. `test_coherent_physics.py`: closed form, linear regime, leakage;
   phase-increment variance (5σ); ASE power and OSNR definition.

### 3.18 Coherent receiver — `physics.detection.coherent_detection`

1. Hybrid outputs `(E_s ± E_lo)/2`, `(E_s ± iE_lo)/2`; balanced currents
   `i_I + i i_Q = R E_s·conj(E_lo)`; per-diode shot + thermal noise.
2. R [A/W], P [W], T [K], R_L [Ω].
3. Ideal hybrid, identical diodes, aligned single-pol LO.
4. Gaussian shot-noise regime.
5. Four square-law detections reusing `photocurrent` / `pin_noise`.
6. Hybrid phase error ε and quadrature gain g (v1.1, sec. 3.21); no finite
   CMRR, no bandwidth limit.
7. Kikuchi, JLT 34, 157 (2016).
8. Noiseless identity (1e-15); per-quadrature variance
   `(2qR(P_s+P_lo)/2 + 8kT/R_L)·fs/2` (5σ); end-to-end
   `1/SNR = 1/SNR_ase + 1/SNR_rx` within 0.12 dB.

### 3.19 Coherent DSP and analysis — `analysis.coherent_dsp`

1. CD compensation `exp(+iβ2Lω²/2)`; Oerder–Meyr timing
   `τ/T = −arg Σ|y_n|² e^{−i2πn/sps}/(2π)`; 4th-power FOE; BPS (B = 32, 2N+1 = 33);
   data-aided SNR with LS gain; EVM = 1/√SNR.
2. D·L [s/m] (display ps/nm); τ [s]; Δf [Hz].
3. Square-QAM/BPSK symmetry; |Δf| < R_s/8; sps ≥ 3 for Oerder–Meyr.
4. Laser linewidth × T ≲ 1e-4 (QPSK), smaller for 16-QAM.
5. Two-pass FOE around the matched filter (ADR-0016).
6. Single-pol DSP has no adaptive equalizer (dual-pol: sec. 3.21); β3 not compensated;
   rotation ambiguity resolved with the reference.
7. Oerder & Meyr, IEEE Trans. Commun. 36, 605 (1988); Pfau et al., JLT 27, 989 (2009).
8. `test_coherent_dsp.py`: FOE within 1e-6 R_s; BPS < 0.2 dB vs ideal phase;
   phase-noise tracking without slips; fractional timing; EVM = 1/√SNR.
   `test_coherent_link.py`: measured SNR = OSNR formula (±0.12 dB) at 10/15/20 dB;
   QPSK BER and 16-QAM SER within Poisson bands of the exact values;
   80 km CD compensation with 1 GHz LO offset; receiver noise; phase noise.

### 3.20 Polarization optics and PMD — `physics.polarization`

1. `U(θ, φ) = [[cos θ, −sin θ e^{−iφ}], [sin θ e^{iφ}, cos θ]]`; waveplate
   `R(−θ) diag(e^{−iδ/2}, e^{iδ/2}) R(θ)`; DGD `D(ω) = diag(e^{−iωτ/2}, e^{iωτ/2})`;
   PMD `J(ω) = D_N U_N … D_1 U_1` with Haar-random `U_k`, `E[τ²] = Σ τ_k²`,
   Maxwellian mean `⟨τ⟩ = √(8/3π) √E[τ²]`; DGD by eigenanalysis
   `τ = |arg(ρ₁/ρ₂)|/Δω` of `J(ω+Δω) J(ω)^H`.
2. θ, φ, δ [rad]; τ [s]; ω [rad/s] baseband.
3. Lossless (no PDL); frequency-independent coupling and section DGD; static.
4. Maxwellian only for many sections (finite N: random flight; sec. 8).
5. Per-bin 2×2 product in the frequency domain (one FFT pair per polarization).
6. No PDL, no temporal SOP drift, no higher-order PMD beyond the section model.
7. Foschini & Poole, JLT 9, 1439 (1991); Gordon & Kogelnik, PNAS 97, 4541
   (2000); Heffner, IEEE PTL 4, 1066 (1992).
8. `tests/validation/test_polarization.py`: SU(2) properties, λ/2 and λ/4
   plates; Haar isotropy (Stokes moments, 5σ); DGD pulse centroids ±τ/2
   (exact); aligned/crossed/orthogonal section sums; `E[τ²] = N τ_s²` over
   3000 fibres (5σ); mean DGD vs an independent 3-D random-flight simulation.

### 3.21 Dual-polarization receiver DSP and hybrid impairments — `analysis.coherent_dsp`, `physics.detection`

1. Hybrid with phase error ε, Q gain g: `i_Q = gR Im(z e^{−iε})`; skew: Q
   delayed by τ (`delay_samples`). GSOP: `I' = I/√P_I`,
   `Q' = (Q − ⟨IQ⟩I/P_I)/√P_Q'`. 2×2 butterfly
   `z_p[k] = Σ_q Σ_m W_pqm y_q[2k+m−h]`, update `W_p ← W_p − μ e_p z_p y*`,
   `e_p = |z_p|² − R` (CMA: `R = E|s|⁴/E|s|²`; RDE: nearest ring). FOE on
   both outputs (summed 4th-power spectra).
2. ε [rad]; g [1]; τ [s]; μ [1] for unit-power input.
3. Unitary channel (for the orthogonal y initialization), static over the
   window, i.i.d. symbols (blind statistics), T/2 sampling.
4. Equalizer span 15 taps at T/2 (≈ ±3.5 symbols) for DGD + residual ISI;
   |Δf| < R_s/8.
5. Offline block training (6 passes, 2 x-only, last at μ/4), then frozen taps;
   two-pass FOE around the matched filter (ADR-0017).
6. Blind DP-64QAM convergence unreliable; PRBS whose recurrence spans < 3
   symbols (e.g. PRBS15 with 64-QAM) misleads blind equalization (diagnostic
   `dsp.prbs_order_too_low`); no training/pilots, no PDL.
7. Godard, IEEE Trans. Commun. 28, 1867 (1980); Kikuchi, JLT 34, 157 (2016);
   Fatadin et al., IEEE PTL 20, 1733 (2008).
8. `tests/validation/test_dual_pol_dsp.py`: CMA inverts rotations incl. the
   equal-mixing state and 1.5-symbol DGD (SNR within 5σ + 0.1 dB of Es/N0,
   distinct sources), RDE for 16-QAM, GSOP exactness, joint FOE.
   `test_dp_coherent_link.py`: SNR = `2 B_ref OSNR/(2 R_s)` with random SOP and
   with 80 km CD + PMD + LO offset; DP-QPSK BER Poisson band; DP-16QAM;
   GSOP + deskew restore the unimpaired SNR (±0.01 dB).

## 4. References

* G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press, 2019.
* G. P. Agrawal, *Fiber-Optic Communication Systems*, 5th ed., Wiley, 2021.
* B. E. A. Saleh, M. C. Teich, *Fundamentals of Photonics*, 3rd ed., Wiley, 2019.
* D. N. Godard, "Self-recovering equalization and carrier tracking in two-dimensional data communication systems," *IEEE Trans. Commun.* 28, 1867 (1980).
* G. J. Foschini, C. D. Poole, "Statistical theory of polarization dispersion in single mode fibers," *J. Lightwave Technol.* 9, 1439 (1991).
* J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
* A. V. Oppenheim, A. S. Willsky, *Signals and Systems*, 2nd ed., Prentice Hall, 1997.
* W. E. Thomson, "Delay networks having maximally flat frequency characteristics," *Proc. IEE* 96 (1949).
* S. W. Golomb, *Shift Register Sequences*, rev. ed., Aegean Park Press, 1982.
* ITU-T Rec. O.150; ITU-T Rec. G.652.
* E. Tiesinga et al., "CODATA recommended values of the fundamental physical constants: 2018," *Rev. Mod. Phys.* 93, 025010 (2021).
