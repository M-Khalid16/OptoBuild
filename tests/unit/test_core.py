from __future__ import annotations

import math

import pytest

from optobuild.core import constants as c
from optobuild.core import errors
from optobuild.core.diagnostics import Diagnostic, Severity, has_errors


def test_exact_si_constants() -> None:
    assert c.SPEED_OF_LIGHT == 299_792_458.0
    assert c.PLANCK_CONSTANT == 6.62607015e-34
    assert c.ELEMENTARY_CHARGE == 1.602176634e-19
    assert c.BOLTZMANN_CONSTANT == 1.380649e-23
    assert c.REDUCED_PLANCK_CONSTANT == pytest.approx(c.PLANCK_CONSTANT / (2 * math.pi))


def test_error_hint_is_part_of_message() -> None:
    err = errors.InvalidParameterError("V_pi must be > 0 V, got -1.", hint="Use a positive V_pi.")
    assert "V_pi must be > 0 V" in str(err)
    assert "Hint: Use a positive V_pi." in str(err)
    assert err.hint == "Use a positive V_pi."


@pytest.mark.parametrize(
    ("cls", "bases"),
    [
        (errors.PortTypeMismatchError, (errors.InvalidGraphError, TypeError)),
        (errors.SimulationCycleError, (errors.InvalidGraphError,)),
        (errors.InvalidParameterError, (ValueError,)),
        (errors.SamplingError, (ValueError,)),
        (errors.NumericalStabilityError, (ArithmeticError,)),
        (errors.ProjectFormatError, (ValueError,)),
    ],
)
def test_error_hierarchy(cls: type, bases: tuple[type, ...]) -> None:
    assert issubclass(cls, errors.OptoBuildError)
    for base in bases:
        assert issubclass(cls, base)


def test_diagnostics() -> None:
    warn = Diagnostic(
        Severity.WARNING,
        "sampling.aliasing_risk",
        "1% energy near Nyquist.",
        hint="Increase samples per symbol.",
        source="fiber",
    )
    assert "WARNING sampling.aliasing_risk: [fiber]" in str(warn)
    assert not has_errors([warn])
    assert has_errors([warn, Diagnostic(Severity.ERROR, "x", "y")])
