"""Candidate single-pair shortest-path algorithms, all on the shared Graph structure.

Each algorithm exists in two versions with IDENTICAL search logic:

  ALGORITHMS[name](graph, s, t)       -> (cost, path, op_counts)   counting version
  TIMED_ALGORITHMS[name](graph, s, t) -> (cost, path)              timing version

Counters inside the search loop slow it down (~1.7x measured for Dijkstra), which would
inflate algorithm time relative to feature time. So the benchmark times the counter-free
version and takes op counts from a separate run of the counting version.
Tests check that both versions return the same cost and path.

  cost      shortest-path cost (math.inf if t is unreachable from s)
  path      list of nodes s..t ([] if unreachable, [s] if s == t)
  op_counts {"nodes_popped": ..., "edges_relaxed": ...}
              nodes_popped  = every pop from the priority queue / BFS queue,
                              including stale heap entries (they cost real work)
              edges_relaxed = every edge examined from a popped node
"""
from __future__ import annotations

import math
from collections import deque
from heapq import heappop, heappush
from typing import Callable

from src.graph import Graph

OpCounts = dict[str, int]
Result = tuple[float, list[int], OpCounts]
TimedResult = tuple[float, list[int]]
INF = math.inf


def _ops(popped: int, relaxed: int) -> OpCounts:
    return {"nodes_popped": popped, "edges_relaxed": relaxed}


def _walk_back(pred: dict[int, int | None], node: int) -> list[int]:
    """Follow predecessors from node back to the root; returns root..node."""
    path = []
    cur: int | None = node
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return path


def _join(pred_f: dict[int, int | None], pred_b: dict[int, int | None], meet: int) -> list[int]:
    """Path s..meet..t from forward and backward predecessor maps."""
    return _walk_back(pred_f, meet) + _walk_back(pred_b, meet)[::-1][1:]


def same_cost(a: float, b: float, rel_tol: float) -> bool:
    """True if two path costs agree (relative tolerance; inf only equals inf)."""
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=rel_tol)


def _check_astar(g: Graph) -> None:
    if g.coords is None:
        raise ValueError("A* needs node coordinates")


def _check_unweighted(g: Graph) -> None:
    if g.weighted:
        raise ValueError("BFS variants are only valid on unweighted graphs")


# ------------------------------------------------------------------ Dijkstra

def dijkstra(g: Graph, s: int, t: int) -> Result:
    """Dijkstra with a binary heap, lazy deletion, early exit at t (counting)."""
    popped = relaxed = 0
    dist = {s: 0.0}
    pred: dict[int, int | None] = {s: None}
    heap = [(0.0, s)]
    adj = g.adj
    while heap:
        d, u = heappop(heap)
        popped += 1
        if d > dist[u]:
            continue
        if u == t:
            return d, _walk_back(pred, t), _ops(popped, relaxed)
        for v, w in adj[u]:
            relaxed += 1
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                heappush(heap, (nd, v))
    return INF, [], _ops(popped, relaxed)


def dijkstra_timed(g: Graph, s: int, t: int) -> TimedResult:
    """Counter-free copy of dijkstra()."""
    dist = {s: 0.0}
    pred: dict[int, int | None] = {s: None}
    heap = [(0.0, s)]
    adj = g.adj
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        if u == t:
            return d, _walk_back(pred, t)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                heappush(heap, (nd, v))
    return INF, []


# ------------------------------------------------------------------ Bidirectional Dijkstra

def bidirectional_dijkstra(g: Graph, s: int, t: int) -> Result:
    """Bidirectional Dijkstra; expands the side with the smaller heap (counting).

    Stops when top_forward + top_backward >= best meeting cost mu.
    """
    popped = relaxed = 0
    if s == t:
        return 0.0, [s], _ops(popped, relaxed)
    dist: list[dict[int, float]] = [{s: 0.0}, {t: 0.0}]
    pred: list[dict[int, int | None]] = [{s: None}, {t: None}]
    heaps: list[list[tuple[float, int]]] = [[(0.0, s)], [(0.0, t)]]
    adjs = (g.adj, g.radj)
    mu, meet = INF, -1
    while heaps[0] and heaps[1]:
        if heaps[0][0][0] + heaps[1][0][0] >= mu:
            break
        side = 0 if len(heaps[0]) <= len(heaps[1]) else 1
        d, u = heappop(heaps[side])
        popped += 1
        dist_side, dist_other = dist[side], dist[1 - side]
        if d > dist_side[u]:
            continue
        for v, w in adjs[side][u]:
            relaxed += 1
            nd = d + w
            if nd < dist_side.get(v, INF):
                dist_side[v] = nd
                pred[side][v] = u
                heappush(heaps[side], (nd, v))
            if v in dist_other:
                total = dist_side[v] + dist_other[v]
                if total < mu:
                    mu, meet = total, v
    if meet < 0:
        return INF, [], _ops(popped, relaxed)
    return mu, _join(pred[0], pred[1], meet), _ops(popped, relaxed)


def bidirectional_dijkstra_timed(g: Graph, s: int, t: int) -> TimedResult:
    """Counter-free copy of bidirectional_dijkstra()."""
    if s == t:
        return 0.0, [s]
    dist: list[dict[int, float]] = [{s: 0.0}, {t: 0.0}]
    pred: list[dict[int, int | None]] = [{s: None}, {t: None}]
    heaps: list[list[tuple[float, int]]] = [[(0.0, s)], [(0.0, t)]]
    adjs = (g.adj, g.radj)
    mu, meet = INF, -1
    while heaps[0] and heaps[1]:
        if heaps[0][0][0] + heaps[1][0][0] >= mu:
            break
        side = 0 if len(heaps[0]) <= len(heaps[1]) else 1
        d, u = heappop(heaps[side])
        dist_side, dist_other = dist[side], dist[1 - side]
        if d > dist_side[u]:
            continue
        for v, w in adjs[side][u]:
            nd = d + w
            if nd < dist_side.get(v, INF):
                dist_side[v] = nd
                pred[side][v] = u
                heappush(heaps[side], (nd, v))
            if v in dist_other:
                total = dist_side[v] + dist_other[v]
                if total < mu:
                    mu, meet = total, v
    if meet < 0:
        return INF, []
    return mu, _join(pred[0], pred[1], meet)


# ------------------------------------------------------------------ A*

def astar(g: Graph, s: int, t: int) -> Result:
    """A* with h(v) = heuristic_scale * euclid(v, t) (counting).

    heuristic_scale (computed once when the graph is built) makes h consistent for any
    non-negative weights, so the first time t is popped its cost is optimal.
    Heap entries are (f, -g, node): ties on f go to the node FURTHER from s (larger g),
    the standard choice that avoids expanding whole plateaus of equal f.
    """
    _check_astar(g)
    popped = relaxed = 0
    coords, scale = g.coords, g.heuristic_scale
    tx, ty = coords[t]
    sx, sy = coords[s]
    dist = {s: 0.0}
    pred: dict[int, int | None] = {s: None}
    heap = [(scale * math.hypot(sx - tx, sy - ty), -0.0, s)]
    adj = g.adj
    while heap:
        _, neg_d, u = heappop(heap)
        popped += 1
        d = -neg_d
        if d > dist[u]:
            continue
        if u == t:
            return d, _walk_back(pred, t), _ops(popped, relaxed)
        for v, w in adj[u]:
            relaxed += 1
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                vx, vy = coords[v]
                heappush(heap, (nd + scale * math.hypot(vx - tx, vy - ty), -nd, v))
    return INF, [], _ops(popped, relaxed)


def astar_timed(g: Graph, s: int, t: int) -> TimedResult:
    """Counter-free copy of astar()."""
    _check_astar(g)
    coords, scale = g.coords, g.heuristic_scale
    tx, ty = coords[t]
    sx, sy = coords[s]
    dist = {s: 0.0}
    pred: dict[int, int | None] = {s: None}
    heap = [(scale * math.hypot(sx - tx, sy - ty), -0.0, s)]
    adj = g.adj
    while heap:
        _, neg_d, u = heappop(heap)
        d = -neg_d
        if d > dist[u]:
            continue
        if u == t:
            return d, _walk_back(pred, t)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                pred[v] = u
                vx, vy = coords[v]
                heappush(heap, (nd + scale * math.hypot(vx - tx, vy - ty), -nd, v))
    return INF, []


# ------------------------------------------------------------------ BFS

def bfs(g: Graph, s: int, t: int) -> Result:
    """Breadth-first search; unweighted graphs only. Stops when t is discovered (counting)."""
    _check_unweighted(g)
    popped = relaxed = 0
    if s == t:
        return 0.0, [s], _ops(popped, relaxed)
    pred: dict[int, int | None] = {s: None}
    queue = deque([s])
    adj = g.adj
    while queue:
        u = queue.popleft()
        popped += 1
        for v, _ in adj[u]:
            relaxed += 1
            if v not in pred:
                pred[v] = u
                if v == t:
                    path = _walk_back(pred, t)
                    return float(len(path) - 1), path, _ops(popped, relaxed)
                queue.append(v)
    return INF, [], _ops(popped, relaxed)


def bfs_timed(g: Graph, s: int, t: int) -> TimedResult:
    """Counter-free copy of bfs()."""
    _check_unweighted(g)
    if s == t:
        return 0.0, [s]
    pred: dict[int, int | None] = {s: None}
    queue = deque([s])
    adj = g.adj
    while queue:
        u = queue.popleft()
        for v, _ in adj[u]:
            if v not in pred:
                pred[v] = u
                if v == t:
                    path = _walk_back(pred, t)
                    return float(len(path) - 1), path
                queue.append(v)
    return INF, []


# ------------------------------------------------------------------ Bidirectional BFS

def bidirectional_bfs(g: Graph, s: int, t: int) -> Result:
    """Bidirectional BFS; unweighted graphs only (counting).

    Expands one whole level of the smaller frontier at a time and, if any meeting is
    found in that level, returns the best one (minimum over the level keeps it exact).
    """
    _check_unweighted(g)
    popped = relaxed = 0
    if s == t:
        return 0.0, [s], _ops(popped, relaxed)
    dist: list[dict[int, int]] = [{s: 0}, {t: 0}]
    pred: list[dict[int, int | None]] = [{s: None}, {t: None}]
    frontiers: list[list[int]] = [[s], [t]]
    adjs = (g.adj, g.radj)
    while frontiers[0] and frontiers[1]:
        side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
        dist_side, dist_other, pred_side = dist[side], dist[1 - side], pred[side]
        best, meet = INF, -1
        nxt: list[int] = []
        for u in frontiers[side]:
            popped += 1
            du = dist_side[u]
            for v, _ in adjs[side][u]:
                relaxed += 1
                if v not in dist_side:
                    dist_side[v] = du + 1
                    pred_side[v] = u
                    nxt.append(v)
                if v in dist_other:
                    total = dist_side[v] + dist_other[v]
                    if total < best:
                        best, meet = total, v
        if meet >= 0:
            return float(best), _join(pred[0], pred[1], meet), _ops(popped, relaxed)
        frontiers[side] = nxt
    return INF, [], _ops(popped, relaxed)


def bidirectional_bfs_timed(g: Graph, s: int, t: int) -> TimedResult:
    """Counter-free copy of bidirectional_bfs()."""
    _check_unweighted(g)
    if s == t:
        return 0.0, [s]
    dist: list[dict[int, int]] = [{s: 0}, {t: 0}]
    pred: list[dict[int, int | None]] = [{s: None}, {t: None}]
    frontiers: list[list[int]] = [[s], [t]]
    adjs = (g.adj, g.radj)
    while frontiers[0] and frontiers[1]:
        side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
        dist_side, dist_other, pred_side = dist[side], dist[1 - side], pred[side]
        best, meet = INF, -1
        nxt: list[int] = []
        for u in frontiers[side]:
            du = dist_side[u]
            for v, _ in adjs[side][u]:
                if v not in dist_side:
                    dist_side[v] = du + 1
                    pred_side[v] = u
                    nxt.append(v)
                if v in dist_other:
                    total = dist_side[v] + dist_other[v]
                    if total < best:
                        best, meet = total, v
        if meet >= 0:
            return float(best), _join(pred[0], pred[1], meet)
        frontiers[side] = nxt
    return INF, []


# ------------------------------------------------------------------ registry

ALGORITHMS: dict[str, Callable[[Graph, int, int], Result]] = {
    "dijkstra": dijkstra,
    "bidirectional_dijkstra": bidirectional_dijkstra,
    "astar": astar,
    "bfs": bfs,
    "bidirectional_bfs": bidirectional_bfs,
}

TIMED_ALGORITHMS: dict[str, Callable[[Graph, int, int], TimedResult]] = {
    "dijkstra": dijkstra_timed,
    "bidirectional_dijkstra": bidirectional_dijkstra_timed,
    "astar": astar_timed,
    "bfs": bfs_timed,
    "bidirectional_bfs": bidirectional_bfs_timed,
}


def applicable_algorithms(g: Graph) -> list[str]:
    """Names of candidate algorithms that are valid on this graph."""
    names = ["dijkstra", "bidirectional_dijkstra"]
    if g.has_coordinates:
        names.append("astar")
    if not g.weighted:
        names += ["bfs", "bidirectional_bfs"]
    return names
