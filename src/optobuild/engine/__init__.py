"""Layer 7 - execution of simulation graphs.

Feed-forward DAG executor, run context, caching and invalidation, progress
reporting and cancellation. Iterative/cavity execution will be a separate
executor (ADR-0005).
"""

from optobuild.engine.cache import ResultCache
from optobuild.engine.context import CancellationToken, ExecutionContext
from optobuild.engine.executor import (
    FeedForwardExecutor,
    NodeResult,
    ProgressEvent,
    SimulationResult,
)

__all__ = [
    "CancellationToken",
    "ExecutionContext",
    "FeedForwardExecutor",
    "NodeResult",
    "ProgressEvent",
    "ResultCache",
    "SimulationResult",
]
