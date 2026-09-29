"""Global simulation layout (ADR-0011).

A layout fixes the time grid shared by every source of a bit-oriented
simulation, like the "global parameters" of commercial link simulators:

    sample_rate = bit_rate * samples_per_bit          [Hz]
    n_samples   = n_bits   * samples_per_bit
    dt          = 1 / sample_rate                     [s]
    T           = n_bits / bit_rate                   [s]

Sources opt in with ``timing_source = "layout"``; the engine passes the layout
to every component through ``RunContext.layout``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from optobuild.core.errors import InvalidParameterError
from optobuild.numerics.grid import TimeGrid


@dataclass(frozen=True)
class SimulationLayout:
    """Bit rate [bit/s], number of bits and samples per bit shared by all sources."""

    bit_rate: float
    n_bits: int
    samples_per_bit: int

    def __post_init__(self) -> None:
        problems = []
        if not (
            isinstance(self.bit_rate, (int, float))
            and not isinstance(self.bit_rate, bool)
            and math.isfinite(self.bit_rate)
            and self.bit_rate > 0
        ):
            problems.append(f"bit_rate must be a finite number > 0 bit/s, got {self.bit_rate!r}")
        for name, minimum in (("n_bits", 1), ("samples_per_bit", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                problems.append(f"{name} must be an integer >= {minimum}, got {value!r}")
        if problems:
            raise InvalidParameterError(
                "Invalid simulation layout: " + "; ".join(problems) + ".",
                hint="Example: bit_rate=10e9, n_bits=2047, samples_per_bit=16.",
            )
        object.__setattr__(self, "bit_rate", float(self.bit_rate))

    @property
    def sample_rate(self) -> float:
        """Sampling rate R_b * sps [Hz]."""
        return self.bit_rate * self.samples_per_bit

    @property
    def n_samples(self) -> int:
        """Number of samples n_bits * sps."""
        return self.n_bits * self.samples_per_bit

    def grid(self) -> TimeGrid:
        """The shared time grid (t0 = 0)."""
        return TimeGrid(self.n_samples, 1.0 / self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form used in project files."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> SimulationLayout:
        """Inverse of :meth:`to_dict` (strict keys)."""
        if not isinstance(data, dict) or set(data) != {"bit_rate", "n_bits", "samples_per_bit"}:
            raise InvalidParameterError(
                f"A layout needs exactly the keys bit_rate, n_bits, samples_per_bit; got {data!r}."
            )
        return cls(**data)


__all__ = ["SimulationLayout"]
