# Regression tests

Regression tests protect behaviour that has **already been validated** against
an analytical or independent reference (see `tests/validation`). A reference
data file may only be added here together with:

1. the validation test that established its correctness,
2. the generating script and its fixed seed,
3. the package version / commit that produced it.

No regression references exist yet (Phase 0 has no physics models).
