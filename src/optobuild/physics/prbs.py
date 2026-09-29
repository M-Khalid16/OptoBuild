"""Pseudo-random binary sequences from maximal-length LFSRs.

Model
-----
For a primitive trinomial ``x^m + x^k + 1`` (m > k) a Fibonacci LFSR with
feedback taps at stages k and m produces the binary recurrence

    s[n] = s[n - k] XOR s[n - m]                (arithmetic over GF(2))

whose characteristic polynomial ``x^m + x^(m-k) + 1`` is the reciprocal of the
trinomial and therefore also primitive. Every non-zero seed of m bits thus
produces a maximal-length sequence of period ``2^m - 1`` containing
``2^(m-1)`` ones and ``2^(m-1) - 1`` zeros per period. The recurrence is
evaluated in vectorized blocks of k bits.

Polynomials follow ITU-T Rec. O.150 (PRBS7, 9, 11, 15, 23, 31) and common
practice (PRBS10, 20). Limitations: O.150 additionally specifies, for some
orders, output inversion and a particular register bit order; those
conventions are *not* reproduced, so bit sequences are maximal-length with the
right polynomial but not bit-identical to an O.150 test set.

Units: bits are dimensionless 0/1 (uint8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from optobuild.core.errors import InvalidParameterError

PRBS_POLYNOMIALS: dict[int, tuple[int, int]] = {
    7: (7, 6),
    9: (9, 5),
    10: (10, 7),
    11: (11, 9),
    15: (15, 14),
    20: (20, 3),
    23: (23, 18),
    31: (31, 28),
}
"""order m -> (m, k) of the trinomial x^m + x^k + 1."""


def prbs_period(order: int) -> int:
    """Period 2^m - 1 of a maximal-length sequence of order m."""
    return 2**order - 1


def prbs(order: int, n_bits: int, seed: int | None = None) -> NDArray[np.uint8]:
    """Generate ``n_bits`` bits of PRBS``order``.

    Parameters
    ----------
    order:
        Register length m; must be a key of :data:`PRBS_POLYNOMIALS`.
    n_bits:
        Number of output bits (>= 1).
    seed:
        Initial register content as an integer in [1, 2^m - 1]; bit i of the
        integer is the i-th output bit. ``None`` means all ones.

    Returns
    -------
    ndarray of uint8, shape (n_bits,).
    """
    if order not in PRBS_POLYNOMIALS:
        raise InvalidParameterError(
            f"Unsupported PRBS order {order}.", hint=f"Use one of {sorted(PRBS_POLYNOMIALS)}."
        )
    if n_bits < 1:
        raise InvalidParameterError(f"n_bits must be >= 1, got {n_bits}.")
    m, k = PRBS_POLYNOMIALS[order]
    state = prbs_period(m) if seed is None else int(seed)
    if not 1 <= state <= prbs_period(m):
        raise InvalidParameterError(
            f"PRBS seed must be in [1, 2^{m} - 1], got {seed}.",
            hint="The all-zero register state is not allowed.",
        )
    total = max(n_bits, m)
    s = np.empty(total, dtype=np.uint8)
    s[:m] = [(state >> i) & 1 for i in range(m)]
    lag_a, lag_b = m, k  # s[n] = s[n - m] ^ s[n - k]
    block = lag_b  # s[n : n+block] depends only on already-computed values
    n = m
    while n < total:
        e = min(n + block, total)
        s[n:e] = s[n - lag_a : e - lag_a] ^ s[n - lag_b : e - lag_b]
        n = e
    return s[:n_bits].copy()


__all__ = ["PRBS_POLYNOMIALS", "prbs", "prbs_period"]
