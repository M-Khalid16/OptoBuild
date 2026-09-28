# ADR-0017: Dual-polarization coherent systems, adaptive MIMO equalization and PMD

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 6b extends the single-polarization coherent chain (ADR-0016) to
polarization-multiplexed transmission. This needs polarization optics
(PBS/PBC, rotations, PMD), a polarization-diverse receiver and blind 2×2
adaptive equalization, plus receiver impairments (hybrid phase/amplitude
imbalance, I/Q skew) and their compensation.

## Decision
1. **Jones calculus in `physics.polarization`.** Fields `(2, N)`; lossless
   elements are SU(2) matrices, frequency-dependent ones `(2, 2, N)` applied
   per FFT bin. PMD uses the waveplate model with Haar-random coupling; equal
   section DGDs give the target mean DGD in the Maxwellian limit. DGD is
   measured by Jones-matrix eigenanalysis.
2. **Composition from existing components instead of monolithic DP blocks.**
   The DP transmitter is laser → PBS(45°) → two IQ modulators → PBC; the
   receiver is PBS on signal and LO → two single-pol `CoherentReceiver`s →
   `DualPolCoherentDSP` (four electrical inputs). A scalar signal is one fully
   polarized state; its azimuth matters only at a PBS (parameter there).
   Symbols carry both tributaries as `(2, n)`; the `PulseShaper` selects one.
3. **y tributary = time-reversed pattern.** A cyclic shift of the x pattern
   (an obvious choice) makes the tributaries indistinguishable to cyclic
   alignment when an odd-length PRBS is repeated for QPSK; the reversed PRBS
   is a different m-sequence.
4. **DSP order (dual pol):** per pol deskew → GSOP → CD compensation → matched
   filter; joint Oerder–Meyr timing; 2 samples/symbol; 2×2 T/2-spaced
   butterfly with CMA (QPSK/BPSK) or CMA then RDE (QAM); FOE on both outputs;
   second pass after derotating the full-rate waveform (re-centres the matched
   filter; 64-QAM fails without it); BPS per pol.
5. **Equalizer schedule (defaults):** 15 taps, μ = 2.5×10⁻⁴ on unit-power
   input, 6 passes over the block: 2 adapting x only, then y initialized
   orthogonally (`W_yy[m] = W_xx*[−m]`, `W_yx[m] = −W_xy*[−m]`), then both;
   last pass μ/4; output computed with frozen taps (offline training for a
   channel static over the window). Chosen from measurements: μ = 10⁻³ lets
   16-QAM taps drift to the filter edges; one x-only pass leaves the x taps too
   noisy for a reliable y initialization (1 failure in 12 seeds at μ = 5×10⁻⁴;
   0 in 12 with the defaults).
6. **Analyzer:** chooses the output-to-tributary assignment by the larger
   normalized correlation; pools signal and noise powers over both
   polarizations; records per-tributary SNR/BER.

## Alternatives considered
* *Constrained equalizer (y always derived from x):* avoids the singularity
  but is exact only for unitary channels without frequency offset.
* *Data-aided (training) equalizer:* robust, but needs a frame/pilot model;
  deferred (Phase 10 or a later coherent extension).
* *Monolithic DP transmitter/receiver components:* fewer graph nodes but
  duplicate validated physics and hide the polarization optics.

## Consequences
* Validated (`tests/validation/test_dp_coherent_link.py`): pooled and
  per-tributary SNR equal `2 B_ref OSNR / (2 R_s)` within 5σ + 0.1 dB with a
  random SOP, and with 80 km CD, random PMD (15 ps mean DGD) and 1 GHz LO
  offset; DP-QPSK BER within Poisson bands; DP-16QAM; GSOP + deskew exactly
  restore the unimpaired SNR.
* **Known limitations:** blind DP-64QAM convergence is unreliable in the full
  demo (2 of 4 seeds with a PRBS23 segment), and PRBS15 with 64-QAM fails even
  in the ideal case because the PRBS recurrence spans only 2.5 symbols
  (non-i.i.d. higher-order statistics; diagnosed as
  `dsp.prbs_order_too_low` when the span is < 3 symbols, an empirical
  threshold). No PDL, no time-varying SOP, no polarization-dependent
  modulator imperfections.
