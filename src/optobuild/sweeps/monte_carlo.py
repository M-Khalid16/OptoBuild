"""Monte Carlo accumulation of bit errors over independent noise realizations.

Trial k runs the project with every stochastic component's generator seeded by
``(root seed, component name, k)`` (ADR-0008, ADR-0012); deterministic parts
are computed once and reused from the cache. Error counts of the trials are
summed:

    N_bits = sum_k N_bits,k,   N_err = sum_k N_err,k,   BER = N_err / N_bits

with an exact Clopper-Pearson 95 % interval on the totals.

Statistical caveat: every trial transmits the *same* bit pattern (the PRBS is
deterministic), so pattern-dependent effects (ISI) are sampled only for that
pattern; the noise realizations are independent. For a pattern that exercises
all relevant bit neighbourhoods (PRBS of order >= memory of the channel in
bits) this is the usual practice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from optobuild.analysis.ber import clopper_pearson
from optobuild.engine.cache import ResultCache
from optobuild.engine.context import CancellationToken
from optobuild.engine.executor import FeedForwardExecutor
from optobuild.persistence.project import Project, run_project


@dataclass(frozen=True)
class MonteCarloBER:
    """Accumulated error counts."""

    n_trials: int
    n_bits: int
    n_errors: int
    ber_lower_95: float
    ber_upper_95: float
    errors_per_trial: tuple[int, ...]
    first_trial: int

    @property
    def ber(self) -> float:
        """Accumulated BER N_err / N_bits."""
        return self.n_errors / self.n_bits


def monte_carlo_ber(
    project: Project,
    n_trials: int,
    *,
    ber_node: str = "ber",
    first_trial: int = 0,
    seed: int | None = None,
    executor: FeedForwardExecutor | None = None,
    cancel: CancellationToken | None = None,
    on_trial: Callable[[int, int, int], None] | None = None,
) -> MonteCarloBER:
    """Run trials ``first_trial ... first_trial + n_trials - 1`` and sum the BER counts.

    ``ber_node`` must record ``n_bits`` and ``n_errors`` (the BER analyzer does).
    ``on_trial(trial, n_errors, n_bits)`` is called after each trial.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if ber_node not in project.graph:
        raise KeyError(f"The project has no node '{ber_node}'.")
    executor = FeedForwardExecutor(ResultCache()) if executor is None else executor
    per_trial: list[int] = []
    total_bits = 0
    for k in range(first_trial, first_trial + n_trials):
        res = run_project(project, seed=seed, executor=executor, trial=k, cancel=cancel)
        n_err = int(res.result(ber_node, "n_errors"))
        n_bits = int(res.result(ber_node, "n_bits"))
        per_trial.append(n_err)
        total_bits += n_bits
        if on_trial is not None:
            on_trial(k, n_err, n_bits)
    total_err = sum(per_trial)
    lo, hi = clopper_pearson(total_err, total_bits)
    return MonteCarloBER(n_trials, total_bits, total_err, lo, hi, tuple(per_trial), first_trial)


__all__ = ["MonteCarloBER", "monte_carlo_ber"]
