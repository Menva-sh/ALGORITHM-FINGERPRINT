"""Shared adjacency-list graph structure used by every candidate algorithm."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Edge = tuple[int, int] | tuple[int, int, float]


@dataclass
class Graph:
    """Graph on nodes 0..n-1 with non-negative edge weights.

    adj[u]  = list of (v, w) for edges u -> v.
    radj[v] = list of (u, w) for edges u -> v (same object as adj when undirected).
    weighted=False means every weight is exactly 1.0.
    coords[u] = (x, y) when the graph has coordinates, else coords is None.
    heuristic_scale = min over edges of w / euclid(u, v); multiplying the Euclidean
    distance by it gives an admissible, consistent A* heuristic for ANY weights.
    """

    n: int
    adj: list[list[tuple[int, float]]]
    radj: list[list[tuple[int, float]]]
    directed: bool
    weighted: bool
    coords: list[tuple[float, float]] | None = None
    heuristic_scale: float = field(default=0.0)
    _m: int = field(default=-1, repr=False)

    @classmethod
    def from_edges(
        cls,
        n: int,
        edges: Iterable[Edge],
        directed: bool = False,
        weighted: bool = True,
        coords: Sequence[tuple[float, float]] | None = None,
    ) -> "Graph":
        """Build a graph. Duplicate edges keep the smallest weight; self-loops are dropped."""
        best: dict[tuple[int, int], float] = {}
        for e in edges:
            u, v = int(e[0]), int(e[1])
            w = float(e[2]) if len(e) == 3 else 1.0
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"edge ({u}, {v}) out of range for n={n}")
            if w < 0 or math.isnan(w):
                raise ValueError(f"edge ({u}, {v}) has invalid weight {w}")
            if not weighted and w != 1.0:
                raise ValueError("unweighted graph must have all weights == 1")
            if u == v:
                continue
            key = (u, v) if directed or u < v else (v, u)
            if key not in best or w < best[key]:
                best[key] = w

        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        radj: list[list[tuple[int, float]]] = [[] for _ in range(n)] if directed else adj
        for (u, v), w in best.items():
            adj[u].append((v, w))
            if directed:
                radj[v].append((u, w))
            else:
                adj[v].append((u, w))

        c = [(float(x), float(y)) for x, y in coords] if coords is not None else None
        if c is not None and len(c) != n:
            raise ValueError("coords must have one entry per node")
        g = cls(n=n, adj=adj, radj=radj, directed=directed, weighted=weighted, coords=c,
                _m=len(best))
        g.heuristic_scale = g._compute_heuristic_scale()
        return g

    @property
    def m(self) -> int:
        """Number of edges (an undirected edge counts once); O(1), stored at build time."""
        if self._m < 0:
            total = sum(len(a) for a in self.adj)
            self._m = total if self.directed else total // 2
        return self._m

    @property
    def has_coordinates(self) -> bool:
        return self.coords is not None

    def edges(self) -> list[tuple[int, int, float]]:
        """All edges as (u, v, w); undirected edges listed once with u < v."""
        return [(u, v, w) for u in range(self.n) for v, w in self.adj[u]
                if self.directed or u < v]

    def component_labels(self) -> list[int]:
        """Weakly connected component id per node, numbered in order of their smallest node."""
        label = [-1] * self.n
        comp = 0
        for start in range(self.n):
            if label[start] >= 0:
                continue
            label[start] = comp
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for nbrs in (self.adj[u], self.radj[u]):
                    for v, _ in nbrs:
                        if label[v] < 0:
                            label[v] = comp
                            queue.append(v)
            comp += 1
        return label

    def _compute_heuristic_scale(self) -> float:
        """min w / euclid over edges with positive length (inf if there are none)."""
        if self.coords is None:
            return 0.0
        scale = math.inf
        for u, v, w in self.edges():
            (x1, y1), (x2, y2) = self.coords[u], self.coords[v]
            d = math.hypot(x1 - x2, y1 - y2)
            if d > 0:
                scale = min(scale, w / d)
        return 0.0 if scale == math.inf else scale
