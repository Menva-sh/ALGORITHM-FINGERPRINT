"""User inputs: every supported format becomes a valid graph; unsupported problems are rejected."""
from __future__ import annotations

import io
import json

import networkx as nx
import pytest

from src.algorithms import ALGORITHMS, applicable_algorithms
from src.config import PROJECT_ROOT
from src.inputs import (SAMPLE_MAZE, UnsupportedInput, parse_file, parse_maze, parse_text)


def test_edge_list_with_commas_and_names() -> None:
    p = parse_text("# roads\nA,B,4\nB,C,2\nA,C,9\n")
    assert p.fmt == "edge list" and p.labels == ["A", "B", "C"] and p.graph.weighted
    assert ALGORITHMS["dijkstra"](p.graph, 0, 2)[0] == 6.0


def test_unweighted_edge_list_enables_bfs() -> None:
    p = parse_text("1 2\n2 3\n3 4\n")
    assert not p.graph.weighted and "bfs" in applicable_algorithms(p.graph)


def test_distance_matrix_with_header_and_index() -> None:
    p = parse_text(",P,Q,R,S\nP,0,3,0,7\nQ,3,0,2,0\nR,0,2,0,1\nS,7,0,1,0\n")
    assert p.fmt == "distance matrix" and p.labels == ["P", "Q", "R", "S"]
    assert p.graph.m == 4 and ALGORITHMS["dijkstra"](p.graph, 0, 3)[0] == 6.0


def test_asymmetric_matrix_noted() -> None:
    p = parse_text("0 1 0 0\n2 0 1 0\n0 1 0 1\n0 0 1 0\n")
    assert p.notes and p.graph.edges()[0] == (0, 1, 1.0)


def test_coordinates_for_edge_list() -> None:
    p = parse_text("a b 1\nb c 1.5\n", "a 0 0\nb 1 0\nc 2 0\n")
    assert p.graph.has_coordinates and "astar" in applicable_algorithms(p.graph)
    with pytest.raises(UnsupportedInput):
        parse_text("a b 1\nb c 1\n", "a 0 0\n")


@pytest.mark.parametrize("fmt", ["graphml", "gml", "json"])
def test_structured_formats(fmt: str) -> None:
    g = nx.DiGraph()
    g.add_edge("x", "y", weight=2.0)
    g.add_edge("y", "z", weight=3.0)
    if fmt == "graphml":
        buf = io.BytesIO()
        nx.write_graphml(g, buf)
        data = buf.getvalue()
    elif fmt == "gml":
        data = "\n".join(nx.generate_gml(g)).encode()
    else:
        data = json.dumps(nx.node_link_data(g, edges="links")).encode()
    p = parse_file(f"g.{fmt}", data)
    assert p.graph.n == 3 and p.graph.m == 2 and p.graph.weighted
    assert any("both directions" in n for n in p.notes)  # directed input made undirected


@pytest.mark.parametrize("text,msg", [
    ("A B -3\n", "negative"), ("A B x\n", "not a number"), ("", "empty"),
    ("A\n", "two node names"), ("A A 1\n", "two nodes"),
])
def test_unsupported_inputs_are_explained(text: str, msg: str) -> None:
    with pytest.raises(UnsupportedInput, match=msg):
        parse_text(text)


def test_maze() -> None:
    p = parse_maze(SAMPLE_MAZE)
    assert p.fmt == "maze" and p.graph.has_coordinates and not p.graph.weighted
    cost = ALGORITHMS["bfs"](p.graph, p.source, p.target)[0]
    assert cost == ALGORITHMS["astar"](p.graph, p.source, p.target)[0] > 0
    with pytest.raises(UnsupportedInput, match="S"):
        parse_maze("#####\n#   #\n#####")
    with pytest.raises(UnsupportedInput, match="no way"):
        parse_maze("#####\n#S#T#\n#####")


def test_binary_file_rejected() -> None:
    with pytest.raises(UnsupportedInput, match="not a text file"):
        parse_file("x.txt", b"\xff\xfe\x00\x80")


def test_app_runs_in_maze_mode() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "demo.py"), default_timeout=300)
    at.session_state["input_mode"] = "Draw a maze"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("your maze" in m.value for m in at.markdown)
