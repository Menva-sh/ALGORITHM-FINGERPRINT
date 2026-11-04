"""Stage 10: demo logic and a headless run of every page of the Streamlit app."""
from __future__ import annotations

import pytest

from src.algorithms import applicable_algorithms
from src.config import PROJECT_ROOT, load_config
from src.demo_logic import recommend, run_all, train_selector
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


@pytest.mark.skipif(not (PROJECT_ROOT / CFG["paths"]["dataset"]).exists(),
                    reason="needs data/dataset.csv")
@pytest.mark.parametrize("page", ["Try a graph", "Algorithm map", "Road network"])
def test_app_pages_run_without_errors(page: str) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "demo.py"), default_timeout=300)
    at.run()
    if page != "Try a graph":
        at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and at.title[0].value
