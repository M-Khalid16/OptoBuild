"""Layer 3 - reusable physical models as pure functions.

Every physical equation lives here exactly once (fiber, modulation, detection,
noise, sources; later atmospheric, laser, photonics). Functions operate on
SI-valued arrays or signals and never on component or GUI objects.
"""
