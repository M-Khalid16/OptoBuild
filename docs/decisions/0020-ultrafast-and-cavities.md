# ADR-0020: Ultrafast propagation (GNLSE/RK4IP) and round-trip cavity iteration

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 9 covers ultrafast pulse propagation (supercontinuum, Raman
self-frequency shift, optical shocks) and mode-locked lasers. The Phase 4
SSFM (ADR-0014) handles only β2, β3 and instantaneous Kerr, with second-order
splitting. Mode-locked lasers need the steady state of a cavity round trip,
which the feed-forward graph executor cannot express as a graph cycle.

## Decision
1. **Generalized NLSE in the frequency domain**: Taylor dispersion to β10,
   loss, Kerr with self-steepening factor (1 + ω/ω0), Blow–Wood Raman
   response applied with its exact spectrum (no sampling of h_R). Our
   envelope convention makes the equation the complex conjugate of the usual
   form (documented in physics.ultrafast).
2. **RK4IP** (Hult 2007) with step-doubling local-error control and local
   extrapolation; fixed steps optional. The Phase 4 SSFM stays for
   telecom-scale propagation (the two agree in the NLSE limit; tested).
3. **Cavities are iterated by a solver, not by graph cycles.** A cavity is an
   ordered list of element callables (gain, filter, fiber, absorber, output
   coupler); `solvers.cavity.run_cavity` repeats round trips until the
   intensity reproduces itself (tolerance, patience) or a limit is reached,
   reporting non-convergence. A component (ModeLockedFiberLaser) builds the
   element list from its parameters and emits the output pulse of the last
   round trip. Graph-level feedback remains out of scope: it would need
   fixed-point semantics for every component and convergence control across
   the graph.
4. **Lumped cavity elements** (energy-saturated gain, fast saturable absorber,
   Gaussian filter/gate, output coupler) in `physics.cavity`; the Gaussian
   gate+filter cavity has an exact fixed point used to validate the iteration.
5. **Pulse metrology** in `analysis.pulses` (FWHM, rms width, TBP, intensity
   autocorrelation with deconvolution) and an Autocorrelator component.

## Alternatives considered
* *Adaptive SSFM with Raman as an ODE*: lower order; RK4IP is the reference
  method in the supercontinuum literature.
* *Cyclic graphs with an iterative executor*: general but heavy (cache
  invalidation, per-edge convergence, deterministic ordering); the round-trip
  solver covers the Phase 9 use cases.
* *Noise-seeded self-starting only*: realistic but slow and stochastic in
  outcome; a weak Gaussian seed plus seeded noise converges reproducibly.

## Consequences
* Validated: fundamental and Satsuma–Yajima solitons, exact dispersionless
  self-steepening intensity solution, photon-number conservation with Raman
  and shock, Gordon's self-frequency shift (2 %), fourth-order convergence,
  exact linear propagation; exact Gaussian fixed point and steady energy of
  the gate+filter cavity from noise; pulse metrics vs closed forms.
* The mode-locked soliton laser is checked for plausibility only (single
  pulse, TBP within 30 % of the sech² limit, soliton number ≈ 1): there is no
  exact solution for the lumped soliton laser.
* Limitations: scalar field (no vector/birefringence effects), frequency-
  independent γ apart from (1 + ω/ω0), no noise (no shot-noise-seeded
  supercontinuum coherence studies), lumped cavity elements (no distributed
  gain dynamics), single circulating window (no pulse-train interactions or
  harmonic mode locking), cavities only inside components.
