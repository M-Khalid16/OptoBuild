# ADR-0019: Laser and amplifier models — rate equations, two-level EDFA, fixed-step integration

* Status: Accepted
* Date: 2026-09-28

## Context
Phase 8 adds laser modelling: CW, semiconductor and fiber lasers. Earlier
phases used an ideal CW laser (with an optional Lorentzian linewidth) and an
amplifier specified by its output OSNR; Phase 6 deferred deriving the OSNR
from a physical amplifier model to this phase.

## Decision
1. **Semiconductor laser: single-mode rate equations** (carrier density,
   photon density, phase) with gain compression, spontaneous coupling and the
   α factor; Langevin forces (Henry) for RIN and phase noise.
   *Integration:* a small generic fixed-step RK4 (`solvers.ode`) whose
   right-hand side is plain arithmetic, so one realization runs on Python
   floats (fast without compiled extensions) and ensembles run on NumPy
   arrays; noise is added after each step (Euler–Maruyama, Ito). Fixed steps
   keep the output on the simulation grid; the default step is τ_p/4.
2. **Output-frequency reference of a DML:** the signal's carrier c/λ is the
   emission frequency at the bias current; the envelope phase carries the
   transient and adiabatic chirp relative to it (avoids an arbitrary phase
   ramp from the gain-compression frequency offset).
3. **EDFA: two-level Giles–Desurvire model**, co-propagating pump and
   signals, integrated in z (RK4); ASE density at the signal integrated along
   the fiber but not fed back into the populations (small-ASE regime); gain
   quasi-static at the average input power (erbium lifetime ~10 ms). The
   amplifier's noise figure therefore follows from the inversion instead of
   being specified.
4. **Fiber laser: steady-state ring** (EDF + output coupler + passive loss)
   solved by bisection on the round-trip condition; with zero background loss
   the Saleh–Jopson relation gives closed forms used for validation.
5. **Layering:** equations in `physics.semiconductor_laser` / `physics.edfa`;
   integration and root finding in `solvers.laser_dynamics` / `solvers.edfa`.

## Alternatives considered
* *scipy `solve_ivp` (adaptive)*: output interpolation off-grid, no SDE
  support, and per-call overhead; the problem is mildly stiff at most for the
  chosen step bound.
* *Numba-compiled integrator*: forbidden without a benchmark (CLAUDE.md);
  pure-Python floats give ~8 µs per RK4 step, adequate for 10⁴–10⁵ steps.
* *Giles model with ASE self-saturation and counter-propagating beams*:
  needs a two-point boundary-value solver (shooting/relaxation); deferred.

## Consequences
* Validated: threshold, L-I slope, turn-on delay, relaxation frequency and
  damping (closed forms); chirp identity; RK4 fourth-order convergence;
  simulated RIN/FM-noise spectra vs an independent linear-response calculation
  (whose f → 0 limit is Henry's linewidth); EDFA z-integration vs the
  Saleh–Jopson solution; full-inversion NF = 2 − 1/G; ring-laser output,
  threshold and slope vs closed forms.
* Limitations: single longitudinal mode (no side modes, mode hopping, MSR),
  no thermal effects or parasitics, linear gain; EDFA without ASE
  self-saturation, backward pumping, ESA or spectrally resolved ASE (white
  over the simulated band); ring laser CW steady state only (no relaxation
  dynamics or mode locking — Phase 9).
