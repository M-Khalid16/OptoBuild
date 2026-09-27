# ADR-0010: Per-source sampling grids and bit-timing metadata

* Status: Accepted
* Date: 2026-09-27

## Context
Phase 2 needs every signal path of a link on one sampling grid, and the
receiver (decision circuit, eye analyzer) needs the bit rate and samples per
bit. Commercial tools use global "layout parameters" (bit rate, sequence
length, samples per bit).

## Decision
1. **No global simulation parameters yet.** Each source defines its own grid:
   the NRZ generator derives it from the bit sequence (`N = n_bits · sps`,
   `dt = 1/(R_b · sps)`); the CW laser has explicit `n_samples` and
   `sample_rate`. Components that combine signals (the MZM) require identical
   grids and raise `SamplingError` with a hint otherwise — never resample
   implicitly. Demo/project builders compute all source grids from one set of
   variables.
2. **Bit timing travels as documented signal metadata**
   (`optobuild.signals.metadata`: `bit_rate`, `samples_per_bit`, `pattern`).
   The NRZ generator sets it, the MZM/fiber/PIN/filter propagate it, and the
   decision circuit and eye analyzer require it (`SignalTypeError` with a
   hint if missing).
3. The BER analyzer receives the transmitted bits through an explicit
   graph connection (`prbs.out -> ber.reference`), never via hidden state.

## Alternatives considered
* *Global parameters in the RunContext*: fewer duplicated values, but adds
  implicit coupling between components and to the project schema; revisit
  when the GUI (Phase 3) needs a "layout parameters" panel.
* *Bit rate as a decision-circuit parameter*: duplicates information and can
  silently disagree with the transmitter.

## Consequences
* Users must keep laser `n_samples`/`sample_rate` consistent with the data
  path; mistakes are caught at run time with a clear message.
* Timing metadata keys are part of the component contract and are tested.
