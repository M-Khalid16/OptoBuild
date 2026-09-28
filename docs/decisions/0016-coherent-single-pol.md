# ADR-0016: Single-polarization coherent transmission and receiver DSP

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 6 adds coherent optical communication. A complete dual-polarization
system also needs polarization beam splitters/combiners, a 2x2 adaptive
(CMA/LMS) equalizer and PMD, each with its own validation. Doing everything
at once would weaken validation of the basic chain.

## Decision
1. **Phase 6a (this ADR): single polarization.** Symbols (`SymbolSequence`),
   Gray constellations (BPSK/QPSK/16-QAM/64-QAM), RC/RRC shaping, IQ
   modulator (two nulled child MZMs from `mzm_field_transfer`), ideal
   amplifier with specified output OSNR, intradyne receiver (90° hybrid +
   balanced PIN pairs with per-diode noise), DSP, analyzer.
   **Phase 6b:** dual polarization, adaptive MIMO equalization, PMD.
2. **Signal chain types:** drives are two real ELECTRICAL signals (I, Q), the
   receiver outputs two photocurrents; the DSP emits SYMBOLS. No complex
   electrical signal type is introduced.
3. **DSP order:** CD compensation → coarse FOE (pass 1) → frequency shift of the
   full-rate waveform → RRC matched filter → Oerder–Meyr fractional timing →
   decimation → residual FOE → BPS → normalization. The two-pass FOE keeps the
   matched filter centred on the signal spectrum (a 1 GHz LO offset otherwise
   costs 0.16 dB at 32 GBd); fractional timing is required because CD
   compensation with an LO offset leaves a fractional group delay (2.5 dB
   penalty with integer-sample timing, measured).
4. **Analysis conventions:** alignment and the rotation ambiguity are resolved
   against the transmitted symbols (standard in simulation; pilots/differential
   coding in practice). SNR is estimated data-aided with a least-squares gain,
   `SNR = |c|² E|s|² / E|r − c s|²`, EVM = 1/√SNR (unbiased by the noise in
   power normalization).
5. **OSNR convention:** `OSNR = P / (N_ase,total B_ref)` with ASE counted in both
   polarizations, B_ref = 12.5 GHz; single-pol signals receive the
   co-polarized half; `SNR = 2 B_ref OSNR / (p R_s)`.

## Alternatives considered
* *Ideal "symbols-to-field" modulator*: hides modulator nonlinearity and
  leakage; the nested-MZM model reuses validated physics.
* *Data-aided phase recovery*: simpler but unrealistic; BPS is blind and its
  penalty is validated (< 0.2 dB vs ideal phase).
* *Complex ELECTRICAL signal type*: convenient, but I/Q as two real signals
  matches hardware ports and needs no new port kind.

## Consequences
* Measured SNR equals the OSNR formula within 0.12 dB (5σ) back-to-back and
  over 80 km with CD compensation and LO offset.
* Demos include realistic impairments whose penalties are attributable
  (receiver noise, carrier leakage, sine nonlinearity, BPS).
