"""Stage 1: every applicable algorithm must return the networkx shortest-path cost."""
from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from src.algorithms import ALGORITHMS, TIMED_ALGORITHMS, applicable_algorithms, same_cost
from src.config import load_config
from src.graph import Graph

N_RANDOM_GRAPHS = 200
QUERIES_PER_GRAPH = 5
REL_TOL = load_config()["benchmark"]["cost_rel_tolerance"]


def random_graph(rng: random.Random, weight_factor: float = 1.0) -> Graph:
    """Small random graph mixing directed/undirected, weighted/unit, with/without coords."""
    n = rng.randint(2, 60)
    p = rng.uniform(0.02, 0.3)
    directed = rng.random() < 0.3
    weighted = rng.random() < 0.6
    with_coords = rng.random() < 0.5
    coords = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)] if with_coords else None
    weight_mode = rng.choice(["uniform", "with_zeros", "euclid_scaled", "heavy"])
    edges = []
    for u in range(n):
        for v in range(n):
            if u == v or (not directed and v < u) or rng.random() >= p:
                continue
            if not weighted:
                w = 1.0
            elif weight_mode == "with_zeros":
                w = rng.choice([0.0, rng.uniform(0, 5)])
            elif weight_mode == "euclid_scaled" and coords:
                w = math.dist(coords[u], coords[v]) * rng.uniform(1.0, 2.0)
            elif weight_mode == "heavy":
                w = rng.paretovariate(1.5)
            else:
                w = rng.uniform(0.1, 100.0)
            edges.append((u, v, w * weight_factor if weighted else w))
    return Graph.from_edges(n, edges, directed=directed, weighted=weighted, coords=coords)


def to_networkx(g: Graph) -> nx.Graph:
    nxg = nx.DiGraph() if g.directed else nx.Graph()
    nxg.add_nodes_from(range(g.n))
    nxg.add_weighted_edges_from(g.edges())
    return nxg


def nx_cost(nxg: nx.Graph, s: int, t: int) -> float:
    try:
        return nx.dijkstra_path_length(nxg, s, t, weight="weight")
    except nx.NetworkXNoPath:
        return math.inf


def assert_valid_path(g: Graph, s: int, t: int, cost: float, path: list[int]) -> None:
    """Path must start at s, end at t, use real edges, and sum to cost."""
    if cost == math.inf:
        assert path == []
        return
    assert path[0] == s and path[-1] == t
    total = 0.0
    for u, v in zip(path, path[1:]):
        weights = [w for x, w in g.adj[u] if x == v]
        assert weights, f"edge {u}->{v} not in graph"
        total += weights[0]
    assert same_cost(total, cost, REL_TOL)


def _check_random_graphs(seed: int, weight_factor: float) -> dict[str, int]:
    rng = random.Random(seed)
    checked = {name: 0 for name in ALGORITHMS}
    for _ in range(N_RANDOM_GRAPHS):
        g = random_graph(rng, weight_factor)
        nxg = to_networkx(g)
        for _ in range(QUERIES_PER_GRAPH):
            s, t = rng.randrange(g.n), rng.randrange(g.n)
            expected = nx_cost(nxg, s, t)
            for name in applicable_algorithms(g):
                cost, path, ops = ALGORITHMS[name](g, s, t)
                assert same_cost(cost, expected, REL_TOL), (name, s, t, cost, expected)
                assert_valid_path(g, s, t, cost, path)
                assert ops["nodes_popped"] >= 0 and ops["edges_relaxed"] >= 0
                checked[name] += 1
    return checked


def test_matches_networkx_on_random_graphs() -> None:
    checked = _check_random_graphs(load_config()["seed"], weight_factor=1.0)
    assert all(c > 0 for c in checked.values()), checked


def test_matches_networkx_with_huge_weights() -> None:
    """Large costs: summation order differs between algorithms, so tolerance must be relative."""
    _check_random_graphs(load_config()["seed"] + 1, weight_factor=1e8)


def test_timed_versions_match_counting_versions() -> None:
    """Timing and counting versions share logic, so cost AND path must be identical."""
    rng = random.Random(load_config()["seed"] + 2)
    for _ in range(N_RANDOM_GRAPHS):
        g = random_graph(rng)
        for _ in range(QUERIES_PER_GRAPH):
            s, t = rng.randrange(g.n), rng.randrange(g.n)
            for name in applicable_algorithms(g):
                cost, path, _ = ALGORITHMS[name](g, s, t)
                assert TIMED_ALGORITHMS[name](g, s, t) == (cost, path), name


def _line_graph(weighted: bool, coords: bool) -> Graph:
    """0-1-2 connected, node 3 isolated."""
    w = 2.5 if weighted else 1.0
    xy = [(0, 0), (1, 0), (2, 0), (5, 5)] if coords else None
    return Graph.from_edges(4, [(0, 1, w), (1, 2, w)], weighted=weighted, coords=xy)


@pytest.mark.parametrize("registry", [ALGORITHMS, TIMED_ALGORITHMS])
@pytest.mark.parametrize("weighted", [True, False])
def test_unreachable_target(weighted: bool, registry: dict) -> None:
    g = _line_graph(weighted, coords=True)
    for name in applicable_algorithms(g):
        cost, path = registry[name](g, 0, 3)[:2]
        assert cost == math.inf and path == [], name


@pytest.mark.parametrize("registry", [ALGORITHMS, TIMED_ALGORITHMS])
@pytest.mark.parametrize("weighted", [True, False])
def test_source_equals_target(weighted: bool, registry: dict) -> None:
    g = _line_graph(weighted, coords=True)
    for name in applicable_algorithms(g):
        cost, path = registry[name](g, 1, 1)[:2]
        assert cost == 0.0 and path == [1], name


def test_directed_unreachable_backwards() -> None:
    g = Graph.from_edges(3, [(0, 1, 1.0), (1, 2, 1.0)], directed=True, weighted=False)
    for name in applicable_algorithms(g):
        assert ALGORITHMS[name](g, 0, 2)[0] == 2.0, name
        assert ALGORITHMS[name](g, 2, 0)[0] == math.inf, name


def test_astar_tie_breaking_beats_dijkstra_on_grid() -> None:
    """With ties broken toward larger g, A* must pop fewer nodes than Dijkstra corner-to-corner."""
    nxg = nx.grid_2d_graph(30, 30)
    idx = {v: i for i, v in enumerate(nxg.nodes)}
    g = Graph.from_edges(len(idx), [(idx[u], idx[v]) for u, v in nxg.edges],
                         weighted=False, coords=list(nxg.nodes))
    s, t = idx[(0, 0)], idx[(29, 29)]
    a = ALGORITHMS["astar"](g, s, t)[2]["nodes_popped"]
    d = ALGORITHMS["dijkstra"](g, s, t)[2]["nodes_popped"]
    assert a < d


def test_same_cost() -> None:
    assert same_cost(2e8 + 0.3, 2e8 + 0.3 + 3e-8, 1e-9)
    assert not same_cost(1.0, 1.1, 1e-9)
    assert same_cost(math.inf, math.inf, 1e-9)
    assert not same_cost(math.inf, 1e300, 1e-9)
    assert same_cost(0.0, 0.0, 1e-9)


def test_applicable_algorithms_masks() -> None:
    weighted_plain = Graph.from_edges(2, [(0, 1, 3.0)])
    unweighted_xy = Graph.from_edges(2, [(0, 1)], weighted=False, coords=[(0, 0), (1, 1)])
    assert applicable_algorithms(weighted_plain) == ["dijkstra", "bidirectional_dijkstra"]
    assert set(applicable_algorithms(unweighted_xy)) == set(ALGORITHMS)


def test_registry_matches_config() -> None:
    candidates = set(load_config()["algorithms"]["candidates"])
    assert set(ALGORITHMS) == candidates == set(TIMED_ALGORITHMS)


@pytest.mark.parametrize("registry", [ALGORITHMS, TIMED_ALGORITHMS])
def test_inapplicable_algorithms_refuse(registry: dict) -> None:
    g = Graph.from_edges(2, [(0, 1, 3.0)])
    for name in ["astar", "bfs", "bidirectional_bfs"]:
        with pytest.raises(ValueError):
            registry[name](g, 0, 1)
