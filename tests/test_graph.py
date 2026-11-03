"""Stage 1: Graph construction rules."""
import math

import pytest

from src.graph import Graph


def test_undirected_counts_and_symmetry() -> None:
    g = Graph.from_edges(3, [(0, 1, 2.0), (1, 2, 3.0)])
    assert g.m == 2
    assert (0, 2.0) in g.adj[1] and (1, 2.0) in g.adj[0]
    assert g.radj is g.adj


def test_directed_reverse_adjacency() -> None:
    g = Graph.from_edges(2, [(0, 1, 4.0)], directed=True)
    assert g.adj[0] == [(1, 4.0)] and g.adj[1] == []
    assert g.radj[1] == [(0, 4.0)] and g.radj[0] == []


def test_duplicates_keep_min_and_self_loops_dropped() -> None:
    g = Graph.from_edges(2, [(0, 1, 5.0), (1, 0, 2.0), (0, 0, 1.0)])
    assert g.m == 1 and g.adj[0] == [(1, 2.0)]


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError):
        Graph.from_edges(2, [(0, 1, -1.0)])
    with pytest.raises(ValueError):
        Graph.from_edges(2, [(0, 2, 1.0)])
    with pytest.raises(ValueError):
        Graph.from_edges(2, [(0, 1, 3.0)], weighted=False)


def test_heuristic_scale_is_min_weight_over_length() -> None:
    coords = [(0, 0), (3, 4), (3, 0)]  # lengths 5 and 4
    g = Graph.from_edges(3, [(0, 1, 10.0), (1, 2, 2.0)], coords=coords)
    assert g.heuristic_scale == pytest.approx(min(10 / 5, 2 / 4))
    assert Graph.from_edges(2, [(0, 1, 1.0)]).heuristic_scale == 0.0
    assert not math.isinf(g.heuristic_scale)
