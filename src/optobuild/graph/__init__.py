"""Layer 6 - the simulation graph model.

Component instances, typed connections, connection validation and topology
(ordering, cycle detection). Contains no execution logic (ADR-0005).
"""

from optobuild.graph.model import Connection, Endpoint, SimulationGraph

__all__ = ["Connection", "Endpoint", "SimulationGraph"]
