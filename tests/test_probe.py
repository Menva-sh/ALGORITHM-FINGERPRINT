"""Stage 7: probe features, per-query cost accounting and the probing experiment."""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src.algorithms import dijkstra
from src.config import load_config
from src.experiments import probing as px
from src.generators import InstanceSpec, build_instance, save_instance
from src.graph import Graph
from src.probe import _slope, probe_features, probe_graph, timed_probe
from tests.test_algorithms import random_graph
from tests.test_models_evaluate import small_cfg, synthetic_dataset

CFG = load_config()
NAMES = CFG["features"]["probe"]["names"]


def path_graph(n: int) -> Graph:
    return Graph.from_edges(n, [(i, i + 1) for i in range(n - 1)], weighted=False)


# ------------------------------------------------------------------ probe features

def test_probe_on_path_graph() -> None:
    f = probe_features(path_graph(5), 0, 4, 3)
    assert f["probe_settled"] == 3 and f["probe_found_target"] == 0.0
    assert f["probe_radius"] == 2.0
    assert f["probe_branch_mean"] == 1.0 and f["probe_branch_var"] == 0.0
    assert f["probe_queue_mean"] == 1.0 and f["probe_growth"] == 0.0
    full = probe_features(path_graph(5), 0, 4, 100)
    assert full["probe_found_target"] == 1.0 and full["probe_settled"] == 5


def test_probe_star_branching_and_source_is_target() -> None:
    star = Graph.from_edges(6, [(0, i) for i in range(1, 6)], weighted=False)
    f = probe_features(star, 0, 3, 1)
    assert f["probe_branch_mean"] == 5.0 and f["probe_queue_final"] == 5.0
    same = probe_features(star, 2, 2, 10)
    assert same["probe_found_target"] == 1.0 and same["probe_settled"] == 1
    assert set(same) == set(NAMES)


def test_unlimited_probe_agrees_with_dijkstra() -> None:
    """With a huge budget the probe is Dijkstra: found <=> reachable, radius == cost."""
    rng = random.Random(3)
    for _ in range(100):
        g = random_graph(rng)
        s, t = rng.randrange(g.n), rng.randrange(g.n)
        cost = dijkstra(g, s, t)[0]
        f = probe_features(g, s, t, g.n + 1)
        assert f["probe_found_target"] == float(cost < math.inf)
        if cost < math.inf:
            assert f["probe_radius"] == pytest.approx(cost)


def test_slope_and_timing() -> None:
    assert _slope([1, 2, 3, 4]) == pytest.approx(1.0)
    assert _slope([5]) == 0.0
    feats, ns = timed_probe(path_graph(50), 0, 49, 16, 3)
    assert ns > 0 and feats["probe_settled"] == 16


def test_probe_graph_rows(tmp_path) -> None:
    spec = InstanceSpec("grid_2d", 100, (("obstacle_fraction", 0.0),), "unit", 0)
    inst = build_instance(spec, CFG)
    rows = probe_graph(save_instance(inst, tmp_path), [4, 8], 1)
    assert len(rows) == 2 * len(inst.queries)
    assert all(r["probe_settled"] <= r["k"] for r in rows)


# ------------------------------------------------------------------ experiment

def fake_probes(df: pd.DataFrame, k_values: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df.iterrows():
        for k in k_values:
            rows.append({"graph_id": r["graph_id"], "query_idx": r["query_idx"], "k": k,
                         **{n: float(rng.random()) for n in NAMES}, "probe_ns": 1000.0 * k})
    return pd.DataFrame(rows)


def probe_cfg() -> dict:
    cfg = small_cfg()
    cfg["features"] = copy.deepcopy(cfg["features"])
    cfg["features"]["probe"]["k_values"] = [16, 64]
    cfg["experiments"]["probing"] = {"static_sets": [[0]], "models": ["plain_tree"]}
    return cfg


def test_merge_probes_keeps_order_and_detects_missing() -> None:
    df = synthetic_dataset(10, 3)
    probes = fake_probes(df, [16])
    merged = px.merge_probes(df, probes, 16, CFG)
    assert merged["graph_id"].tolist() == df["graph_id"].tolist()
    assert (merged["cost_probe_ns"] == 16_000.0).all()
    with pytest.raises(ValueError):
        px.merge_probes(df, probes.iloc[1:], 16, CFG)


def test_probe_cost_is_charged_per_query_not_amortised() -> None:
    df = px.merge_probes(synthetic_dataset(30, 4), fake_probes(synthetic_dataset(30, 4), [64]),
                         64, CFG)
    res = ev.evaluate(df, small_cfg(), [0], model_names=["plain_tree"], cv=False,
                      extra_features=px.probe_feature_names(CFG), extra_cost_col="cost_probe_ns")
    t, test = res["holdout"], res["_test_index"]
    probe_s = df.iloc[test]["cost_probe_ns"].sum() / 1e9
    tier0_s = df.iloc[test]["cost_tier0_ns"].sum() / 1e9
    assert t.loc["plain_tree", "feature_s"] == pytest.approx(tier0_s + probe_s)
    am = t.loc["plain_tree", "net_amortised_s"] - t.loc["plain_tree", "search_s"] \
        - t.loc["plain_tree", "inference_s"]
    assert am == pytest.approx(tier0_s / 4 + probe_s)


def test_probing_run_and_figures(tmp_path) -> None:
    df = synthetic_dataset(40, 3)
    cfg = probe_cfg()
    summary, groups = px.run(df, fake_probes(df, [16, 64]), cfg)
    assert len(summary) == 1 * 3 * 1
    static = summary[summary["k"] == 0].iloc[0]
    assert static["probe_ms"] == 0.0 and static["accuracy_gain"] == 0.0
    probed = summary[summary["k"] == 64].iloc[0]
    assert probed["probe_ms"] == pytest.approx(0.064)
    assert probed["overhead_ms"] > static["overhead_ms"]
    px.plot_tradeoff(summary, "plain_tree", tmp_path / "a.png")
    px.plot_by_weights(groups, "T0", "plain_tree", tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "b.png").stat().st_size > 0
