# Signal model

Status: **implemented in Phase 1** for `OpticalSignal`, `ElectricalSignal`
and `DigitalSequence` (`optobuild.signals`), and `SymbolSequence` (Phase 6;
shape `(n,)` or `(2, n)` for dual polarization, Phase 6b). `NoiseRepresentation` (tracked noise),
`ComplexEnvelope` as a separate base class and `PulseTrain` are specified
here but **not yet implemented**; they
are added when the first component needs them. Decisions: ADR-0001
(representation), ADR-0002 (FFT convention), ADR-0003 (units).

## 1. General rules

1. Signals are **immutable** value objects (frozen dataclasses whose NumPy
   arrays are set read-only). A component never mutates its input; fan-out of
   a signal to several consumers is therefore safe without copying.
2. **Time is the last array axis** of every sampled array.
3. All stored quantities are **SI** (s, Hz, W, V, A, m). Presentation units
   (dBm, nm, ps) are produced only by `core.units` at interfaces.
4. Every sampled signal carries a `TimeGrid` (`numerics.grid.TimeGrid`):
   `n_samples` N, sampling interval `dt`, start time `t0`.
5. Every signal carries a `metadata` mapping (read-only) for provenance:
   producing component name/type/version, and model-specific annotations
   (e.g. `symbol_rate`, `samples_per_symbol`, reference bit sequence id).
   Metadata never influences physics silently; components read it only via
   documented keys, defined in `optobuild.signals.metadata`: `bit_rate`
   [bit/s], `samples_per_bit`, `pattern` (ADR-0010).

## 2. Signal types

| Type | Kind | Sampled | Payload | Units |
|---|---|---|---|---|
| `ComplexEnvelope` (abstract base) | — | yes | complex array `(..., N)` + reference frequency | payload-specific |
| `OpticalSignal` | `OPTICAL` | yes | Jones field `(n_pol, N)`, `n_pol in {1, 2}` | sqrt(W) |
| `ElectricalSignal` | `ELECTRICAL` | yes | real array `(N,)` + quantity (`VOLTAGE`/`CURRENT`) | V or A |
| `DigitalSequence` | `DIGITAL` | no | `uint8` bits (0/1), `bit_rate` | bits, Hz |
| `SymbolSequence` | `SYMBOLS` | no | complex or integer symbols, `symbol_rate`, alphabet description | —, Hz |
| `PulseTrain` (Phase 9) | `OPTICAL` | yes | `OpticalSignal` + repetition rate / window semantics | sqrt(W) |
| `NoiseRepresentation` | attached | yes | see §5 | W/Hz or A^2/Hz |

`PulseTrain` is not a separate port kind: ultrafast propagation uses the same
`OpticalSignal` with metadata describing the pulse window. This keeps all
optical components interoperable.

## 3. Optical signals (complex-envelope representation)

The optical carrier is **not** sampled. The real field is represented as

```
E_p(t) = Re{ A_p(t) * exp(+i 2 pi f_ref t) }      p = polarization index
```

with a normalization such that

```
P(t) = sum_p |A_p(t)|^2            instantaneous (cycle-averaged) power [W]
```

i.e. `A` is in sqrt(W). This is the standard fiber-optics normalization.

| Quantity | Symbol | Representation |
|---|---|---|
| time | t | `grid.time()` = `t0 + n dt` [s] |
| sampling interval | dt | `grid.dt` [s] |
| sampling rate | fs | `grid.sample_rate` = 1/dt [Hz]; equals the simulated optical bandwidth |
| carrier (reference) frequency | f_ref | `center_frequency` [Hz], a float field of the signal |
| wavelength | lambda | derived: `c / f_ref` (vacuum wavelength) [m]; never stored separately |
| baseband frequency | f | `grid.frequency()`; absolute frequency `nu = f_ref + f` |
| complex envelope | A_p(t) | `field`, complex128, shape `(n_pol, N)` |
| power | P(t) | derived: `sum_p |A_p|^2` [W]; average power = mean over window |
| phase | phi_p(t) | derived: `angle(A_p)` [rad], relative to the reference carrier |
| polarization | — | Jones vector along axis 0: `n_pol = 1` (scalar, assumed single polarization state) or `2` (x, y) |
| noise | — | realized inside `field` and/or described by `NoiseRepresentation` (§5) |
| metadata | — | read-only mapping |

Key consequences:

* A CW laser at exactly `f_ref` with power P is `A = sqrt(P) exp(i phi0)`,
  constant in time; its spectrum is a single DC bin.
* A positive baseband frequency is a *higher* optical frequency, i.e. a
  *shorter* wavelength: `lambda(f) = c / (f_ref + f)`.
* Signals combined in one component (e.g. couplers) must have identical
  `TimeGrid` N and dt and identical `f_ref`; otherwise a `SamplingError` is
  raised with instructions (explicit resampling / frequency-shift component).
  Automatic resampling is never performed.
* WDM (later): one `OpticalSignal` whose `f_ref` is the band centre and whose
  sampling rate covers all channels; each channel is frequency-shifted onto
  the common grid by an explicit multiplexer component.
* Scalar (`n_pol = 1`) signals are the Phase 2 default: one fully polarized
  state whose orientation matters only where a component resolves
  polarization (a PBS takes it as a parameter). Polarization-transforming
  components (controller, PMD) require `n_pol = 2`; a PBC builds such a
  field from two scalar signals (Phase 6b, ADR-0017).

### Time reference and delay

Propagation uses a **retarded time frame** moving at the group velocity at
`f_ref`. A pure group delay `tau_g = beta_1 L` is recorded by advancing the
grid start time (`t0 -> t0 + tau_g`) instead of circularly shifting samples,
which keeps the pulse/pattern inside the window. Absolute arrival time is
`t0 + n dt`. Any *relative* delay between frequency components (dispersion)
acts on the samples and is subject to window wrap-around
(see [numerical_conventions.md](numerical_conventions.md) §6).

## 4. Electrical signals

```
x(t_n)   real-valued, units V or A  (field `quantity`)
```

* Real-valued baseband; the spectrum is Hermitian and only `f >= 0` is shown
  in one-sided plots (see numerical conventions §5 for the one-sided PSD).
* Conversion optical -> electrical happens only in detector components.
* RF/IQ complex baseband electrical signals (Phase 6) will use
  `ComplexEnvelope` with an electrical carrier; not needed before Phase 6.

## 5. Noise representation

Two complementary representations are supported by design:

1. **Realized (Monte Carlo) noise** — random samples added to `field` /
   samples using the component's seeded generator. Required for direct BER
   counting. This is the Phase 2 default.
2. **Tracked (semi-analytic) noise** — a `NoiseRepresentation` attached to the
   signal, holding a noise PSD on the signal's frequency grid (e.g. ASE in
   W/Hz per polarization) *not* added to the samples. Used later for OSNR and
   Gaussian-approximation BER estimates.

A noise source is either realized or tracked, never both; the representation
records which sources have been realized to prevent double counting.
Complex white noise with two-sided PSD `N0` [W/Hz] over the simulated
bandwidth `fs` has per-sample variance `sigma^2 = N0 * fs`, split equally
between real and imaginary parts.

## 6. Digital and symbol sequences

* `DigitalSequence`: bits as `uint8` in {0, 1}, `bit_rate` [bit/s]; metadata
  records the generator (e.g. PRBS order, seed/state) so BER analyzers can
  reconstruct or receive the reference pattern.
* `SymbolSequence`: symbols plus `symbol_rate` [Bd]; the mapping
  (e.g. OOK {0,1}, PAM-4 levels, QPSK constellation) is described in metadata.
* The **reference sequence** for BER counting travels as an explicit graph
  connection (transmitter bits -> BER analyzer reference port), not as hidden
  global state.

## 7. Validation obligations (Phase 1)

* Round-trip: power computed from `field` equals the requested CW power.
* Immutability: attempts to write into signal arrays raise.
* Construction rejects: wrong shapes, `n_pol` not in {1,2}, non-finite
  values, `f_ref <= 0`, grid/array length mismatch, bits outside {0,1}.
