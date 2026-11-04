"""Stage 9: config profiles, shifted train/test evaluation, real-graph loading."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import src.benchmark as bm
from src.config import apply_profile, load_config
from src.evaluate import train_test_evaluate
from src.experiments import labels_ablation as lx
from src.experiments import real_graphs as rx
from src.experiments import size_shift as sx
from src.generators import InstanceSpec, build_instance, load_instance, save_instance
from src.real_graphs import build_all, read_edge_list
from tests.test_models_evaluate import small_cfg, synthetic_dataset

CFG = load_config()


def renamed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.assign(graph_id=prefix + df["graph_id"])


def shift_cfg() -> dict:
    cfg = small_cfg()
    cfg["experiments"] = copy.deepcopy(cfg["experiments"])
    cfg["experiments"]["size_shift"].update(
        small_max=100, large_min=1000, feature_sets=[{"tiers": [0]}], models=["plain_tree"])
    cfg["experiments"]["labels"]["models"] = ["plain_tree"]
    return cfg


# ------------------------------------------------------------------ profiles

def test_profiles_merge_and_reject_unknown() -> None:
    large = load_config(profile="large")
    assert large["generation"]["sizes"] == [20000]
    assert large["generation"]["families"] == CFG["generation"]["families"]  # untouched
    assert large["benchmark"]["repeats"] == CFG["benchmark"]["repeats"]
    assert large["paths"]["dataset"] != CFG["paths"]["dataset"]
    with pytest.raises(KeyError):
        apply_profile(CFG, "nope")


def test_benchmark_respects_feature_tiers(tmp_path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["benchmark"].update(feature_tiers=[0, 1], warmup_runs=1, repeats=2)
    cfg["generation"]["queries_per_graph"] = 2
    inst = build_instance(InstanceSpec("erdos_renyi", 100, (("avg_degree", 6),), "unit", 0), cfg)
    df = bm.benchmark_graph(save_instance(inst, tmp_path), cfg)
    assert "cost_tier1_ns" in df and "cost_tier2_ns" not in df and "f_diameter_exact" not in df


# ------------------------------------------------------------------ shifted evaluation

def test_train_test_evaluate() -> None:
    tr, te = synthetic_dataset(40, 3), renamed(synthetic_dataset(20, 3, seed=1), "x")
    res = train_test_evaluate(tr, te, small_cfg(), [0], ["plain_tree"])
    t = res["holdout"]
    assert t.loc["oracle", "regret_mean"] == 0 and t.loc["plain_tree", "accuracy"] >= 0.9
    assert len(res["_choices"]) == len(te)
    with pytest.raises(ValueError):
        train_test_evaluate(tr, tr, small_cfg(), [0], ["plain_tree"])


def test_size_shift_run_and_summary() -> None:
    df = synthetic_dataset(80, 3)
    df_large = renamed(synthetic_dataset(40, 3, seed=2), "xl")
    table = sx.run(df, shift_cfg(), df_large=df_large)
    assert set(table["scenario"]) == {"large->large", "small->large", "xl->xl", "main->xl"}
    sub = table[table["scenario"] == "small->large"]
    assert (sub["test_n_min"] >= 1000).all()
    ref = table[table["scenario"] == "large->large"]
    assert sub["test_queries"].iloc[0] == ref["test_queries"].iloc[0]  # same test graphs
    summary = sx.shift_summary(table)
    assert set(summary["shift"]) == {"small->large", "main->xl"}
    assert np.allclose(summary["regret_change"],
                       summary["regret_shifted"] - summary["regret_reference"])


def test_labels_ablation_with_second_machine() -> None:
    df = synthetic_dataset(60, 3)
    rng = np.random.default_rng(0)
    other = df.copy()
    for a in ["dijkstra", "bidirectional_dijkstra", "astar", "bfs", "bidirectional_bfs"]:
        other[f"time_{a}"] = other[f"time_{a}"] * rng.uniform(0.8, 1.2, len(df))
    table = lx.run(df, shift_cfg(), other)
    assert {"time labels", "op-count labels"} <= set(table["scenario"])
    assert any(s.startswith("other machine") for s in table["scenario"])
    assert (table[table["method"] == "oracle"]["regret_mean"] == 0).all()


# ------------------------------------------------------------------ real graphs

def test_read_edge_list(tmp_path) -> None:
    (tmp_path / "g.edges").write_text("# comment\n% other\na b 2.5\nb c 1.0\nb a 4.0\n")
    (tmp_path / "g.coords").write_text("a 0 0\nb 1 0\nc 2 0\n")
    g, ids = read_edge_list(tmp_path / "g.edges", tmp_path / "g.coords")
    assert g.n == 3 and g.m == 2 and g.weighted and g.has_coordinates
    assert (ids["b"], 2.5) in g.adj[ids["a"]]  # duplicate kept the smaller weight
    (tmp_path / "u.edges").write_text("1 2 1\n2 3 1\n")
    assert not read_edge_list(tmp_path / "u.edges")[0].weighted
    (tmp_path / "bad.coords").write_text("a 0 0\n")
    with pytest.raises(ValueError):
        read_edge_list(tmp_path / "g.edges", tmp_path / "bad.coords")


def test_build_all_real_instances(tmp_path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "ring.edges").write_text("\n".join(f"{i} {(i + 1) % 30}" for i in range(30)))
    cfg = load_config(profile="real")
    cfg["generation"]["queries_per_graph"] = 5
    index = build_all(cfg, src, out)
    assert len(index) == 1 and index.loc[0, "family"] == "real"
    g, queries, meta = load_instance(out / f"{index.loc[0, 'graph_id']}.npz")
    assert g.n == 30 and len(queries) == 5 and not meta["weighted"]
    assert (out / "index.csv").exists()


def test_real_graphs_experiment_on_synthetic_stand_in() -> None:
    df = synthetic_dataset(60, 3)
    real = renamed(synthetic_dataset(5, 4, seed=3), "real")
    summary, graphs = rx.run(df, real, shift_cfg())
    assert len(summary) == 1 and graphs["graph_id"].nunique() == 5
    assert np.isclose(summary["regret_change"].iloc[0],
                      summary["regret_real"].iloc[0] - summary["regret_synthetic"].iloc[0])
