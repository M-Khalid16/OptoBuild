"""Frequency-domain S-matrix circuit solver for linear photonic circuits (ADR-0018).

Every element has P bidirectional ports; at each optical frequency its
scattering matrix S (P x P) maps incoming wave amplitudes a to outgoing b:
b = S a (amplitudes in sqrt(W), our e^{+i w t} convention). A circuit is a
set of elements, internal connections (port pairs) and external ports.

Solution (sub-network connection; e.g. Filipsson, IEEE Trans. MTT 29, 1081
(1981); the same formulation as SAX/Photontorch): stack all element ports,
S = blockdiag(S_k), split into external (e) and internal (i) ports. An
internal connection p <-> q means a_p = b_q, a_q = b_p, i.e. a_i = G b_i with a
symmetric permutation G. Eliminating the internal waves,

    S_ext = S_ee + S_ei (I - G S_ii)^-1 G S_ie                            (1)

which is exact for any topology including feedback loops (rings, cavities),
provided I - G S_ii is invertible (it is for passive circuits with any loss;
exactly lossless resonances on the frequency grid are singular in principle
and flagged).

Numerics: dense solve per frequency, O(n_freq n_i^3); intended for circuits
of tens to hundreds of ports. Reciprocity and energy conservation of the
result are checked in tests, not enforced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optobuild.core.errors import InvalidParameterError, NumericalStabilityError


@dataclass(frozen=True)
class Element:
    """A circuit element: port names and S-matrices, shape (n_freq, P, P)."""

    name: str
    ports: tuple[str, ...]
    smatrix: NDArray[np.complex128]

    def __post_init__(self) -> None:
        s = np.asarray(self.smatrix, dtype=complex)
        if s.ndim == 2:
            s = s[None, :, :]
        p = len(self.ports)
        if s.ndim != 3 or s.shape[1:] != (p, p):
            raise InvalidParameterError(
                f"Element '{self.name}': S-matrix shape {s.shape} does not match {p} ports.",
                hint="Give S as (n_freq, P, P) with P = number of ports.",
            )
        if len(set(self.ports)) != p:
            raise InvalidParameterError(f"Element '{self.name}' has duplicate port names.")
        object.__setattr__(self, "smatrix", s)


@dataclass
class Circuit:
    """Netlist: elements, internal connections and named external ports."""

    elements: dict[str, Element] = field(default_factory=dict)
    connections: list[tuple[str, str]] = field(default_factory=list)
    external: dict[str, str] = field(default_factory=dict)

    def add(self, element: Element) -> Element:
        if element.name in self.elements:
            raise InvalidParameterError(f"Duplicate element name '{element.name}'.")
        self.elements[element.name] = element
        return element

    def _check_port(self, ref: str) -> None:
        name, _, port = ref.partition(".")
        el = self.elements.get(name)
        if el is None or port not in el.ports:
            raise InvalidParameterError(
                f"Unknown port '{ref}'.", hint="Refer to ports as 'element.port'."
            )
        used = {p for c in self.connections for p in c} | set(self.external.values())
        if ref in used:
            raise InvalidParameterError(f"Port '{ref}' is already connected or exposed.")

    def connect(self, a: str, b: str) -> None:
        """Join two element ports ('element.port')."""
        self._check_port(a)
        self._check_port(b)
        if a == b:
            raise InvalidParameterError(f"Cannot connect port '{a}' to itself.")
        self.connections.append((a, b))

    def expose(self, name: str, ref: str) -> None:
        """Make an element port an external circuit port called ``name``."""
        if name in self.external:
            raise InvalidParameterError(f"Duplicate external port '{name}'.")
        self._check_port(ref)
        self.external[name] = ref

    def smatrix(self) -> tuple[NDArray[np.complex128], tuple[str, ...]]:
        """External S-matrix (n_freq, E, E) by eq. (1) and the external port names."""
        refs: list[str] = []
        blocks = []
        n_freq = None
        for el in self.elements.values():
            refs += [f"{el.name}.{p}" for p in el.ports]
            blocks.append(el.smatrix)
            n = el.smatrix.shape[0]
            if n_freq is None or n_freq == 1:
                n_freq = n
            elif n not in (1, n_freq):
                raise InvalidParameterError(
                    f"Element '{el.name}' has {n} frequencies, others {n_freq}."
                )
        if not blocks:
            raise InvalidParameterError("The circuit has no elements.")
        index = {r: k for k, r in enumerate(refs)}
        total = len(refs)
        s = np.zeros((n_freq, total, total), dtype=complex)
        k = 0
        for b in blocks:
            p = b.shape[1]
            s[:, k : k + p, k : k + p] = b
            k += p
        ext = [index[r] for r in self.external.values()]
        # unconnected, unexposed ports are terminated without reflection (a = 0): dropping
        # them from both sets is exact (their outgoing waves leave the circuit)
        internal = [index[r] for pair in self.connections for r in pair]
        g = np.zeros((len(internal), len(internal)))
        for j in range(0, len(internal), 2):
            g[j, j + 1] = g[j + 1, j] = 1.0
        s_ee = s[:, ext][:, :, ext]
        if not internal:
            return s_ee, tuple(self.external)
        s_ei = s[:, ext][:, :, internal]
        s_ie = s[:, internal][:, :, ext]
        s_ii = s[:, internal][:, :, internal]
        lhs = np.eye(len(internal))[None] - g[None] @ s_ii
        cond = np.linalg.cond(lhs)
        if np.any(~np.isfinite(cond)) or np.any(cond > 1e12):
            raise NumericalStabilityError(
                "Circuit equations are singular (a lossless resonance exactly on the grid?).",
                hint="Add a small propagation loss or shift the frequency grid.",
            )
        x = np.linalg.solve(lhs, g[None] @ s_ie)
        return s_ee + s_ei @ x, tuple(self.external)


# -- element builders (S-matrices from transfer functions) ------------------------------------


def two_port(name: str, transmission: ArrayLike, reflection: ArrayLike | None = None) -> Element:
    """Reciprocal two-port (ports 'a', 'b'): S = [[r, t], [t, r]]."""
    t = np.atleast_1d(np.asarray(transmission, dtype=complex))
    r = np.zeros_like(t) if reflection is None else np.broadcast_to(reflection, t.shape)
    s = np.zeros((t.size, 2, 2), dtype=complex)
    s[:, 0, 1] = s[:, 1, 0] = t
    s[:, 0, 0] = s[:, 1, 1] = r
    return Element(name, ("a", "b"), s)


def four_port_coupler(name: str, matrix: ArrayLike, n_freq: int = 1) -> Element:
    """Directional coupler (ports 'in1', 'in2' -> 'out1', 'out2') from a 2x2 transfer matrix C:
    b_out = C a_in and, reciprocally, b_in = C^T a_out."""
    c = np.asarray(matrix, dtype=complex)
    s = np.zeros((4, 4), dtype=complex)
    s[2:, :2] = c
    s[:2, 2:] = c.T
    return Element(name, ("in1", "in2", "out1", "out2"), np.repeat(s[None], n_freq, axis=0))


def element_from_blocks(name: str, ports: Sequence[str], smatrix: ArrayLike) -> Element:
    """Generic element from an explicit S-matrix."""
    return Element(name, tuple(ports), np.asarray(smatrix, dtype=complex))


__all__ = ["Circuit", "Element", "element_from_blocks", "four_port_coupler", "two_port"]
