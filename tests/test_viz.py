"""Ready-made example inputs and the SVG graph picture."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.config import PROJECT_ROOT, load_config
from src.inputs import parse_file
from src.viz import graph_svg, layout

CFG = load_config()
EX = PROJECT_ROOT / "app" / "examples"
ROAD = PROJECT_ROOT / CFG["paths"]["real_graphs"] / "road_piedmont"


def ready_parsed(kind: str):
    if kind == "weighted":
        return parse_file("weighted_network.csv", (EX / "weighted_network.csv").read_bytes())
    if kind == "unweighted":
        return parse_file("unweighted_network.txt", (EX / "unweighted_network.txt").read_bytes())
    return parse_file("road_piedmont.edges", ROAD.with_suffix(".edges").read_bytes(),
                      ROAD.with_suffix(".coords").read_text(encoding="utf-8"))


def ready_graph(kind: str):
    return ready_parsed(kind).graph


@pytest.mark.parametrize("kind, weighted, coords", [("weighted", True, False),
                                                    ("unweighted", False, False),
                                                    ("coordinates", True, True)])
def test_ready_made_inputs_have_the_advertised_kind(kind: str, weighted: bool, coords: bool) -> None:
    g = ready_graph(kind)
    assert (g.weighted, g.has_coordinates) == (weighted, coords)
    assert g.n >= CFG["generation"]["sizes"][0]  # inside the size range the selector was trained on


@pytest.mark.parametrize("kind", ["weighted", "unweighted", "coordinates"])
def test_picture_is_valid_svg_with_path_and_endpoints(kind: str) -> None:
    g = ready_graph(kind)
    pos = layout(g)
    assert len(pos) == g.n and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pos)
    svg = graph_svg(g, pos, 0, g.n - 1, path=[0, g.n - 1])
    assert ET.fromstring(svg).tag.endswith("svg")  # raises if the markup is broken
    assert svg.count("<circle") >= 2 and "<polyline" in svg


def test_layout_is_deterministic() -> None:
    g = ready_graph("unweighted")
    assert layout(g) == layout(g)


def test_neighbourhood_keeps_the_path_and_stays_small() -> None:
    from src.algorithms import ALGORITHMS
    from src.viz import CAP, neighbourhood

    g = ready_graph("weighted")
    _, path, _ = ALGORITHMS["dijkstra"](g, 0, g.n - 1)
    nodes = neighbourhood(g, path)
    assert nodes[:len(path)] == path and len(set(nodes)) == len(nodes)
    assert len(nodes) <= max(CAP, len(path) + 12)
    assert neighbourhood(g, path) == nodes  # deterministic
    # every extra node touches the picture: it is adjacent to something already chosen
    chosen = set(nodes)
    assert all(any(v in chosen for v, _ in g.adj[u]) for u in nodes[len(path):])


@pytest.mark.parametrize("kind, has_weights, has_coords", [("weighted", True, False),
                                                           ("unweighted", False, False),
                                                           ("coordinates", True, True)])
def test_path_view_draws_nodes_edges_weights_and_coordinates(kind: str, has_weights: bool,
                                                             has_coords: bool) -> None:
    from src.algorithms import ALGORITHMS
    from src.viz import path_view

    parsed = ready_parsed(kind)
    g = parsed.graph
    _, path, _ = ALGORITHMS["dijkstra"](g, 0, g.n - 1)
    svg, shown, numbered = path_view(g, path, parsed.labels)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f".//{ns}circle")) == shown                 # one circle per node
    assert len(root.findall(f".//{ns}line")) >= len(path) - 1            # edges, path included
    import re

    # distances are the grey or blue numbers on edges (node ids are black or white)
    weights = [t.text for t in root.iter(f"{ns}text")
               if t.get("fill") in ("#5A5A55", "#0072B2") and re.fullmatch(r"[\d.,]+", t.text or "")]
    assert bool(weights) == has_weights
    assert not has_weights or len(weights) >= len(path) - 1   # at least every path edge is labelled
    assert ("Start" in "".join(root.itertext())) and ("End" in "".join(root.itertext()))
    assert ("(" in "".join(t.itertext() if False else (t.text or "") for t in root.iter(f"{ns}title"))
            ) == has_coords
    assert 'stroke="#0072B2"' in svg or "stroke='#0072B2'" in svg  # the path is highlighted


def test_long_ids_are_numbered_and_far_stragglers_dropped() -> None:
    from src.algorithms import ALGORITHMS
    from src.viz import neighbourhood, path_view

    parsed = ready_parsed("coordinates")  # OpenStreetMap ids have 8 digits
    g = parsed.graph
    _, path, _ = ALGORITHMS["astar"](g, 0, g.n - 1)
    svg, shown, numbered = path_view(g, path, parsed.labels)
    assert numbered and shown <= len(neighbourhood(g, path))
    ns = "{http://www.w3.org/2000/svg}"
    written = "".join("".join(t.itertext()) for t in ET.fromstring(svg).iter(f"{ns}text"))
    assert "5301" not in written and "5306" not in written   # no 8-digit ids written on the picture
    assert "node 53" in svg                                  # but hovering still names them
