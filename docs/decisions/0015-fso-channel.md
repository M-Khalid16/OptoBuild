# ADR-0015: Free-space optical channel model

* Status: Accepted
* Date: 2026-09-27

## Context
Phase 5 adds horizontal FSO links: geometric spreading, apertures, pointing,
fog/haze/rain, turbulence, beam wander, outage and link margin. The model
must fit the existing waveform simulator and the Monte Carlo framework.

## Decision
1. **Power-gain channel**: the channel multiplies the optical envelope by
   `sqrt(h)`, `h = η_tx η_rx e^{−βL} h_p(r) h_t`. Phase distortion by turbulence
   is not modelled (valid for intensity modulation / direct detection;
   coherent FSO would need phase-screen propagation later).
2. **Quasi-static fading**: one channel state per run / Monte Carlo trial
   (fading coherence ~1-10 ms ≫ simulation windows of ns-µs); a diagnostic
   flags windows longer than 100 µs. Statistics over channel states come
   from trials (ADR-0012).
3. **One model object** (`physics.fso_channel.FSOChannelModel`) owns the
   analytic statistics (mean gain, outage) *and* the sampler, so analysis,
   the component and Monte Carlo share the same equations.
4. **Exact geometry**: collected power of a displaced Gaussian spot in a
   circular aperture via the non-central χ² CDF (instead of the
   Farid–Hranilovic approximation, which is used only as a cross-check);
   displacement = static offset + Gaussian (jitter + beam wander) ⇒ Rice.
5. **Turbulence**: log-normal (weak) and Gamma-Gamma (plane wave, point
   receiver) with Rytov variance for constant Cn²; aperture averaging only
   for the log-normal model.
6. **Outage**: exact Rice survival probability without turbulence; otherwise
   Gauss-Legendre quadrature over the displacement with the turbulence CDF.

## Alternatives considered
* *Split-step phase-screen propagation*: needed for coherent/adaptive-optics
  studies and short-term beam statistics; much more expensive; later phase.
* *Fast fading within a window*: requires a temporal turbulence model; not
  needed for the bit rates simulated here.

## Consequences
* FSO links reuse the IM/DD receiver, BER analyzer and Monte Carlo tools.
* The model's limits (constant Cn², plane-wave Gamma-Gamma parameters, no
  inner/outer scale, no angle-of-arrival or obscuration) are documented and
  surfaced as diagnostics where detectable.
