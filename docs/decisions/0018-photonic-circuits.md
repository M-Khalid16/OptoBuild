# ADR-0018: Photonic circuits — S-matrix solver and signal-flow device components

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 7 adds photonic integrated circuits: waveguides, couplers, MZIs,
microrings, Bragg gratings, phase shifters and filters, analysed over
wavelength (transmission, resonances, Q, FSR, insertion loss, group delay).
Such circuits contain feedback (rings, cavities) and reflections, which the
feed-forward time-domain executor (ADR-0005/0011) cannot represent as a
graph of unidirectional signals.

## Decision
1. **Two complementary paths.**
   * *Frequency-domain circuit analysis* (`solvers.circuit`): netlists of
     elements with bidirectional ports and S-matrices on a frequency grid,
     solved exactly by eliminating internal waves,
     `S_ext = S_ee + S_ei (I − G S_ii)⁻¹ G S_ie`. Feedback and reflections are
     handled without iteration. Circuits are built in Python (not yet stored
     in project files).
   * *Signal-flow device components* (`components.photonic`): rings, MZI and
     Bragg grating apply their closed-form transfer at the absolute frequency
     of every FFT bin (`nu = f_ref + f`) to the optical envelope, so they fit
     existing links. Assumptions: unidirectional excitation (a grating's
     reflection is an output port, never re-injected upstream),
     polarization-independent devices, periodic window (diagnostic when a
     resonance spans < 4 frequency bins).
2. **Device physics once, in `physics.integrated_optics`** (closed forms) and
   `physics.multilayer` (thin-film TMM); the solver is generic and contains no
   device equations.
3. **Conventions:** forward propagation `exp(−iβz)` (consistent with
   `E = Re{A e^{+iωt}}`); coupler `[[t, −iκ], [−iκ, t]]`; waveguide dispersion
   linear in frequency about λ₀ from `n_eff` and `n_g`; group delay
   `τ = −d arg H / dω`.
4. **Resonance analysis** (`analysis.resonances`) works on any sampled
   response: prominence-based peak finding, half-depth FWHM with
   interpolation, Q = x/FWHM, FSR, extinction, group delay.

## Alternatives considered
* *Time-domain (delay-line) ring models in the executor:* would need an
  iterative/cyclic executor (planned for Phase 9 cavities) and sample-level
  delays; the frequency-domain transfer is exact for LTI devices.
* *Transfer (ABCD/T) matrices only:* natural for cascades but awkward for
  ring/branching topologies; S-matrices cover both.
* *External library (SAX, Photontorch, gdsfactory):* heavy dependencies (JAX,
  PyTorch) against the dependency policy; the formulation is small.

## Consequences
* Validation: circuit solver = closed forms to 1e-12 (all-pass and add-drop
  rings, MZI), reciprocity and unitarity; CMT = TMM to (dn/n)² scale for
  uniform and π-shifted gratings (the latter assembled in the solver); FSR,
  FWHM, Q, extinction and group delay against exact expressions.
* Limitations: no mode solver (n_eff, n_g, κ are inputs), no wavelength
  dependence of coupling, no thermal/electro-optic models (phase shift is a
  parameter), no back-reflection feedback into upstream time-domain
  components, dense solve per frequency (tens–hundreds of ports).
