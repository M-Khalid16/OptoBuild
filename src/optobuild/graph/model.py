"""The simulation graph model (ADR-0005): nodes, typed connections, topology.

The graph holds component instances and directed connections
``(source node, output port) -> (target node, input port)``. It validates
every connection when it is made and the whole graph on demand; it contains
no execution logic.

Connection rules (ADR-0004):

* both ports must exist and carry the same :class:`SignalKind`;
* an input port accepts exactly one connection;
* an OPTICAL output may feed at most one non-tap input (plus any number of
  tap inputs) - physical power splitting requires a splitter component;
* ELECTRICAL/DIGITAL/SYMBOLS outputs may fan out freely.

Cycles are allowed to be *built* (future iterative executor) but are
rejected by :meth:`SimulationGraph.topological_order`.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from optobuild.components.base import Component
from optobuild.components.spec import PortSpec
from optobuild.core.diagnostics import Diagnostic, Severity
from optobuild.core.errors import (
    InvalidGraphError,
    PortTypeMismatchError,
    SimulationCycleError,
)
from optobuild.signals.kinds import SignalKind


@dataclass(frozen=True)
class Endpoint:
    """A port on a named node."""

    node: str
    port: str

    def __str__(self) -> str:
        return f"{self.node}.{self.port}"


@dataclass(frozen=True)
class Connection:
    """Directed connection from an output port to an input port."""

    source: Endpoint
    target: Endpoint

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"


def _find_port(ports: tuple[PortSpec, ...], name: str, node: str, direction: str) -> PortSpec:
    for p in ports:
        if p.name == name:
            return p
    raise InvalidGraphError(
        f"Component '{node}' has no {direction} port '{name}'.",
        hint=f"Available {direction} ports: {[p.name for p in ports]}.",
    )


class SimulationGraph:
    """Directed graph of component instances."""

    def __init__(self) -> None:
        self._nodes: dict[str, Component] = {}
        self._connections: list[Connection] = []

    # --- nodes -----------------------------------------------------------------------------
    def add(self, component: Component) -> Component:
        """Add a component; its name must be unique in the graph."""
        if not isinstance(component, Component):
            raise InvalidGraphError(f"{component!r} is not a Component instance.")
        if component.name in self._nodes:
            raise InvalidGraphError(
                f"A component named '{component.name}' already exists in the graph.",
                hint="Component names must be unique; choose another name.",
            )
        self._nodes[component.name] = component
        return component

    def node(self, name: str) -> Component:
        """Component instance called ``name``."""
        try:
            return self._nodes[name]
        except KeyError:
            raise InvalidGraphError(
                f"No component named '{name}' in the graph.",
                hint=f"Existing components: {list(self._nodes)}.",
            ) from None

    @property
    def nodes(self) -> tuple[Component, ...]:
        """Components in insertion order."""
        return tuple(self._nodes.values())

    @property
    def node_names(self) -> tuple[str, ...]:
        """Component names in insertion order."""
        return tuple(self._nodes)

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Component]:
        return iter(self._nodes.values())

    def remove(self, name: str) -> None:
        """Remove a component and every connection touching it."""
        self.node(name)
        del self._nodes[name]
        self._connections = [
            c for c in self._connections if name not in (c.source.node, c.target.node)
        ]

    def replace(self, component: Component) -> None:
        """Replace the component with the same name, keeping valid connections.

        Raises if an existing connection is invalid for the new instance (for
        example because its parameter-dependent ports changed); in that case
        the graph is left unchanged.
        """
        old = self.node(component.name)
        self._nodes[component.name] = component
        try:
            for conn in self._connections:
                if component.name in (conn.source.node, conn.target.node):
                    self._check_ports(conn.source, conn.target)
        except InvalidGraphError:
            self._nodes[component.name] = old
            raise

    def set_parameters(self, name: str, **changes: Any) -> Component:
        """Change parameters of node ``name`` (validated); returns the new instance."""
        new = self.node(name).with_parameters(**changes)
        self.replace(new)
        return new

    # --- connections -----------------------------------------------------------------------
    @property
    def connections(self) -> tuple[Connection, ...]:
        """Connections in creation order."""
        return tuple(self._connections)

    def _check_ports(self, source: Endpoint, target: Endpoint) -> tuple[PortSpec, PortSpec]:
        src = _find_port(self.node(source.node).outputs(), source.port, source.node, "output")
        dst = _find_port(self.node(target.node).inputs(), target.port, target.node, "input")
        if src.kind is not dst.kind:
            raise PortTypeMismatchError(
                f"Cannot connect {source} ({src.kind.value}) to {target} ({dst.kind.value}).",
                hint="Ports must carry the same signal kind; insert a conversion component "
                "(e.g. a photodiode for optical -> electrical).",
            )
        return src, dst

    def connect(
        self, source_node: str, source_port: str, target_node: str, target_port: str
    ) -> Connection:
        """Connect ``source_node.source_port`` to ``target_node.target_port``."""
        source = Endpoint(source_node, source_port)
        target = Endpoint(target_node, target_port)
        src, dst = self._check_ports(source, target)
        for c in self._connections:
            if c.target == target:
                raise InvalidGraphError(
                    f"Input {target} is already connected to {c.source}.",
                    hint="An input accepts one connection; use a combiner/adder for several.",
                )
        if src.kind is SignalKind.OPTICAL and not dst.tap:
            for c in self._connections:
                if c.source == source and not self._target_port(c).tap:
                    raise InvalidGraphError(
                        f"Optical output {source} already feeds {c.target}; connecting it to "
                        f"{target} as well would duplicate optical power.",
                        hint="Insert an optical splitter, or connect an analyzer (tap) input.",
                    )
        conn = Connection(source, target)
        self._connections.append(conn)
        return conn

    def disconnect(self, connection: Connection) -> None:
        """Remove a connection."""
        try:
            self._connections.remove(connection)
        except ValueError:
            raise InvalidGraphError(f"Connection {connection} does not exist.") from None

    def _target_port(self, conn: Connection) -> PortSpec:
        return _find_port(
            self.node(conn.target.node).inputs(), conn.target.port, conn.target.node, "input"
        )

    def incoming(self, name: str) -> Mapping[str, Connection]:
        """Connections into node ``name`` keyed by input port name."""
        self.node(name)
        return {c.target.port: c for c in self._connections if c.target.node == name}

    def outgoing(self, name: str) -> tuple[Connection, ...]:
        """Connections out of node ``name``."""
        self.node(name)
        return tuple(c for c in self._connections if c.source.node == name)

    def downstream(self, name: str) -> set[str]:
        """All nodes reachable from ``name`` (excluding ``name`` unless in a cycle)."""
        seen: set[str] = set()
        stack = [c.target.node for c in self.outgoing(name)]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(c.target.node for c in self.outgoing(n))
        return seen

    # --- validation and topology -----------------------------------------------------------
    def find_cycle(self) -> list[str] | None:
        """One directed cycle as a list of node names, or ``None`` if acyclic."""
        succ = {n: [c.target.node for c in self.outgoing(n)] for n in self._nodes}
        color = dict.fromkeys(self._nodes, 0)  # 0 new, 1 on stack, 2 done
        for start in self._nodes:
            if color[start]:
                continue
            path: list[str] = []
            stack: list[tuple[str, Iterator[str]]] = [(start, iter(succ[start]))]
            color[start] = 1
            path.append(start)
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    stack.pop()
                    path.pop()
                    color[node] = 2
                elif color[nxt] == 1:
                    return path[path.index(nxt) :] + [nxt]
                elif color[nxt] == 0:
                    color[nxt] = 1
                    path.append(nxt)
                    stack.append((nxt, iter(succ[nxt])))
        return None

    def topological_order(self) -> list[str]:
        """Deterministic execution order (Kahn's algorithm, ties by insertion order)."""
        index = {n: i for i, n in enumerate(self._nodes)}
        indegree = dict.fromkeys(self._nodes, 0)
        for c in self._connections:
            indegree[c.target.node] += 1
        ready = [(index[n], n) for n, d in indegree.items() if d == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            _, n = heapq.heappop(ready)
            order.append(n)
            for c in self.outgoing(n):
                indegree[c.target.node] -= 1
                if indegree[c.target.node] == 0:
                    heapq.heappush(ready, (index[c.target.node], c.target.node))
        if len(order) != len(self._nodes):
            cycle = self.find_cycle() or []
            raise SimulationCycleError(
                f"The graph contains a cycle: {' -> '.join(cycle)}.",
                hint="Feed-forward simulation needs an acyclic graph; feedback/cavity "
                "systems require the iterative executor (not yet available).",
            )
        return order

    def validate(self) -> list[Diagnostic]:
        """Whole-graph diagnostics: unconnected inputs, cycles, component warnings."""
        diags: list[Diagnostic] = []
        if not self._nodes:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "graph.empty",
                    "The graph has no components.",
                    hint="Add at least one source component.",
                )
            )
        for comp in self._nodes.values():
            connected = self.incoming(comp.name)
            for port in comp.inputs():
                if port.name not in connected and not port.optional:
                    diags.append(
                        Diagnostic(
                            Severity.ERROR,
                            "graph.unconnected_input",
                            f"Required input '{port.name}' ({port.kind.value}) is not connected.",
                            hint="Connect a compatible output to this input.",
                            source=comp.name,
                        )
                    )
            diags.extend(comp.diagnostics)
        cycle = self.find_cycle()
        if cycle:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "graph.cycle",
                    f"Cycle: {' -> '.join(cycle)}.",
                    hint="Remove the feedback connection; cavities need the iterative executor.",
                )
            )
        return diags


__all__ = ["Connection", "Endpoint", "SimulationGraph"]
