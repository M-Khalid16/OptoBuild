"""Layer 3 - reusable physical models as pure functions.

Every physical equation lives here exactly once. Implemented (Phase 2):
``prbs`` (maximal-length sequences), ``sources`` (CW laser), ``modulation``
(NRZ drive, Mach-Zehnder modulator), ``fiber`` (linear propagation),
``noise`` (shot/thermal PSDs), ``detection`` (PIN photodiode).
Functions operate on SI-valued arrays and never on component or GUI objects.
"""
