"""Stage 10: demo logic, plain-language helpers, and a headless run of every app page."""
from __future__ import annotations

import pandas as pd
import pytest

from src.algorithms import applicable_algorithms
from src.config import PROJECT_ROOT, load_config
from src.demo_logic import (NAMES, confidence_label, feature_label, nearest_graph, reason,
                            recommend, run_all, train_selector, why_items)
from src.fingerprint import extract
from src.generators import InstanceSpec, build_instance
from src.real_graphs import parse_edge_list
from tests.test_models_evaluate import synthetic_dataset

CFG = load_config()


def test_recommend_and_run_all() -> None:
    model = train_selector(synthetic_dataset(40, 3), CFG)
    g = build_instance(InstanceSpec("grid_2d", 100, (("obstacle_fraction", 0.0),), "unit", 0),
                       CFG).graph
    rec = recommend(model, extract(g, tiers=[0], cfg=CFG).features, g, CFG)
    assert rec["choice"] in applicable_algorithms(g)
    assert 0 <= rec["confidence"] <= 1
    assert sum(rec["probs"].values()) == pytest.approx(1.0)
    assert set(rec["shap"].index) == set(CFG["features"]["tiers"][0])
    runs = run_all(g, 0, g.n - 1, CFG)
    assert set(runs["algorithm"]) == set(applicable_algorithms(g))
    assert runs["median_us"].is_monotonic_increasing and runs["path_cost"].nunique() == 1


def test_parse_edge_list_text() -> None:
    g, ids = parse_edge_list("x y 2\ny z 3\n", "x 0 0\ny 1 0\nz 2 0\n")
    assert g.n == 3 and g.weighted and g.has_coordinates
    with pytest.raises(ValueError):
        parse_edge_list("# only a comment\n")


def test_plain_language_helpers() -> None:
    unweighted = {"is_weighted": 0.0, "has_coordinates": 0.0, "n_nodes": 1000.0}
    for algo in NAMES:
        assert reason(algo, unweighted).endswith(".")
    assert "unweighted" in reason("bidirectional_bfs", unweighted)
    assert "coordinates" in reason("astar", {"is_weighted": 1.0, "has_coordinates": 1.0})
    assert feature_label("n_nodes", 1000) == "Graph size (1,000 nodes)"
    assert feature_label("has_coordinates", 0) == "No map coordinates"
    items = why_items(pd.Series({"is_weighted": 0.4, "has_coordinates": 0.2, "n_nodes": -0.1}),
                      unweighted)
    assert items == [("Graph is unweighted", 1.0), ("No map coordinates", 0.5)]
    assert confidence_label(0.9)[0] == "Confident pick" and confidence_label(0.3)[1] == "warn"


def test_nearest_graph() -> None:
    graphs = pd.DataFrame({"f_a": [1.0, 100.0, 10.0], "f_b": [0.0, 1.0, 0.0]})
    assert nearest_graph({"a": 9.0, "b": 0.0}, graphs, ["a", "b"]) == 2


@pytest.mark.skipif(not (PROJECT_ROOT / CFG["paths"]["dataset"]).exists(),
                    reason="needs data/dataset.csv")
@pytest.mark.parametrize("page", [None, "views/algorithm_map.py", "views/road_network.py"])
def test_app_pages_run_without_errors(page: str | None) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "demo.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page:
        at.switch_page(page).run()
        assert not at.exception, [e.value for e in at.exception]
    assert any("<h1>" in m.value for m in at.markdown)
