# ADR-0003: Physical unit strategy

* Status: Accepted
* Date: 2026-09-27

## Context
Optical engineering mixes nm, THz, ps/(nm·km), dB/km, dBm, … (FR-004).
Hidden unit conversions are a major error source; a unit-aware array library
costs performance in inner loops.

## Decision
* **All internal values are plain SI floats/arrays** (s, Hz, m, W, V, A, K, Ω,
  rad; dimensionless ratios linear, never dB).
* **Conversion only at boundaries** (user input, project files for
  display-unit values, reports, GUI) through `optobuild.core.units`:
  explicit functions (`dbm_to_watt`, `db_per_km_to_per_m`,
  `ps_per_nm_km_to_s_per_m2`, `wavelength_to_frequency`, …) and a
  whitelisted unit table for parsing `(value, unit)` pairs.
* User text such as `"1550 nm"` is parsed only into `(number, unit-symbol)` by
  a strict tokenizer and looked up in the table — never `eval`, never
  arithmetic on formatted strings.
* `ParameterSpec.unit` records the SI unit of the stored value;
  `display_unit` is presentation-only.
* Project files store SI values (with the unit symbol alongside for
  readability and validation).

## Alternatives considered
* *pint / astropy.units throughout*: strong checking but overhead on large
  arrays and friction with SciPy; may still be used in tests or at the GUI
  boundary later.
* *Storing display units*: makes physics code unit-dependent.

## Consequences
* Physics functions document SI units for every argument and never see dB.
* Unit mistakes are caught by boundary validation and by tests of `core.units`.
