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


def ready_graph(kind: str):
    if kind == "weighted":
        return parse_file("weighted_network.csv", (EX / "weighted_network.csv").read_bytes()).graph
    if kind == "unweighted":
        return parse_file("unweighted_network.txt", (EX / "unweighted_network.txt").read_bytes()).graph
    return parse_file("road_piedmont.edges", ROAD.with_suffix(".edges").read_bytes(),
                      ROAD.with_suffix(".coords").read_text(encoding="utf-8")).graph


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
