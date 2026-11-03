"""Stage 5: masked selection, group split, baselines, metrics and the full evaluation."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import evaluate as ev
from src.config import load_config
from src.models import (PlainTreeSelector, build_models, class_scores, feature_columns,
                        feature_cost_ns, inference_ns, masked_choice, select)

CFG = load_config()
CANDIDATES = CFG["algorithms"]["candidates"]
TIER_NAMES = CFG["features"]["tiers"]


def synthetic_dataset(n_graphs: int = 60, queries: int = 5, seed: int = 0) -> pd.DataFrame:
    """Dataset where the fastest algorithm is a known function of the graph flags/size."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_graphs):
        weighted, coords = bool(rng.integers(2)), bool(rng.integers(2))
        n = int(rng.choice([100, 1000]))
        app = {"dijkstra": True, "bidirectional_dijkstra": True, "astar": coords,
               "bfs": not weighted, "bidirectional_bfs": not weighted}
        if weighted:
            winner = "astar" if coords else "bidirectional_dijkstra"
        else:
            winner = "bfs" if n == 100 else "bidirectional_bfs"
        feats = {name: float(rng.random()) for t in [0, 1] for name in TIER_NAMES[t]}
        feats.update(n_nodes=float(n), is_weighted=float(weighted), has_coordinates=float(coords))
        for q in range(queries):
            row = {"graph_id": f"g{g}", "family": "fam", "n_target": n,
                   "weights": "uniform" if weighted else "unit", "query_idx": q,
                   **{f"f_{k}": v for k, v in feats.items()},
                   "cost_tier0_ns": 1_000.0, "cost_tier1_ns": 50_000.0,
                   "cost_tier2_ns": 9e9, "cost_tier3_ns": 9e9}
            for a in CANDIDATES:
                row[f"app_{a}"] = app[a]
                row[f"time_{a}"] = (100_000.0 if a == winner else 200_000.0 * (1 + rng.random())) \
                    if app[a] else np.nan
            row["best_time"] = row["best_ops"] = winner
            rows.append(row)
    return pd.DataFrame(rows)


def small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["model"]["cv_folds"] = 3
    cfg["model"]["inference_timing_samples"] = 10
    cfg["model"]["rf"]["n_estimators"] = 20
    return cfg


# ------------------------------------------------------------------ selection

def test_masked_choice_never_picks_inapplicable() -> None:
    scores = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [-np.inf, -np.inf, -np.inf]])
    mask = np.array([[True, False, True], [False, True, True], [False, True, True]])
    assert masked_choice(scores, mask).tolist() == [0, 1, 1]  # last row: first applicable


def test_class_scores_maps_classes_and_unseen_get_minus_inf() -> None:
    X = np.zeros((4, 1))
    model = DummyClassifier(strategy="prior").fit(X, ["bfs", "bfs", "astar", "bfs"])
    s = class_scores(model, X, CANDIDATES)
    assert s[0, CANDIDATES.index("bfs")] == pytest.approx(0.75)
    assert s[0, CANDIDATES.index("astar")] == pytest.approx(0.25)
    assert np.isneginf(s[0, CANDIDATES.index("dijkstra")])


@pytest.mark.parametrize("name", ["plain_tree", "random_forest", "svm", "mlp"])
def test_models_respect_mask(name: str) -> None:
    df = synthetic_dataset(30, 3)
    cols = feature_columns(df, CFG, [0, 1])
    X, y = df[cols].to_numpy(), df["best_time"].to_numpy()
    model = build_models(small_cfg())[name].fit(X, y)
    mask = np.zeros((len(df), len(CANDIDATES)), dtype=bool)
    mask[:, CANDIDATES.index("dijkstra")] = True  # only Dijkstra allowed
    assert (select(model, X, mask, CANDIDATES) == CANDIDATES.index("dijkstra")).all()


def test_inference_time_positive() -> None:
    df = synthetic_dataset(10, 2)
    X = df[feature_columns(df, CFG, [0])].to_numpy()
    model = DummyClassifier().fit(X, df["best_time"])
    mask = ev.applicability_mask(df, CANDIDATES)
    assert inference_ns(model, X, mask, CANDIDATES, 5, 0) > 0


def test_feature_columns_and_cost() -> None:
    df = synthetic_dataset(2, 1)
    assert len(feature_columns(df, CFG, [0, 1])) == len(TIER_NAMES[0]) + len(TIER_NAMES[1])
    with pytest.raises(KeyError):
        feature_columns(df, CFG, [2])
    assert (feature_cost_ns(df, [0, 1]) == 51_000.0).all()


# ------------------------------------------------------------------ split and baselines

def test_split_by_graph_keeps_graphs_whole() -> None:
    df = synthetic_dataset(50, 4)
    train, test = ev.split_by_graph(df, 0.2, 0)
    tr, te = set(df.iloc[train]["graph_id"]), set(df.iloc[test]["graph_id"])
    assert not tr & te and len(tr) + len(te) == 50 and len(te) == 10


def test_best_fixed_must_be_always_applicable() -> None:
    T = np.array([[5.0, 4.0, 1.0], [5.0, 4.0, np.nan]])
    assert ev.best_fixed_algorithm(T, ["a", "b", "c"]) == 1


def test_choice_metrics_oracle_and_inapplicable() -> None:
    T = np.array([[100.0, 200.0], [300.0, 150.0]])
    z = np.zeros(2)
    m = ev.choice_metrics(np.array([0, 1]), T, np.array([0, 1]), z + 10, z + 1, 5.0, 0.05)
    assert m["accuracy"] == 1.0 and m["regret_mean"] == 0.0
    assert m["net_s"] == pytest.approx((20 + 10 + 250) / 1e9)
    assert m["net_amortised_s"] == pytest.approx((2 + 10 + 250) / 1e9)
    m2 = ev.choice_metrics(np.array([1, 0]), T, np.array([0, 1]), z, z, 0.0, 0.05)
    assert m2["accuracy"] == 0.0 and m2["regret_mean"] == pytest.approx(1.0)  # 200/100, 300/150
    with pytest.raises(ValueError):
        ev.choice_metrics(np.array([0, 0]), np.array([[np.nan, 1.0]] * 2), np.zeros(2, int),
                          z, z, 0.0, 0.05)


# ------------------------------------------------------------------ full evaluation

def test_evaluate_end_to_end_on_learnable_data() -> None:
    df = synthetic_dataset(80, 4)
    res = ev.evaluate(df, small_cfg())
    t = res["holdout"]
    assert t.loc["oracle", "regret_mean"] == 0.0 and t.loc["oracle", "accuracy"] == 1.0
    assert t.loc["random_forest", "accuracy"] >= 0.95
    assert t.loc["best_fixed", "saved_vs_best_fixed"] == pytest.approx(0.0)
    assert t.loc["run_all", "net_s"] >= t.loc["oracle", "net_s"]
    for name in ["plain_tree", "random_forest", "svm", "mlp"]:
        row = t.loc[name]
        assert row["net_s"] == pytest.approx(row["search_s"] + row["feature_s"] + row["inference_s"])
        assert row["feature_s"] > 0 and row["inference_s"] > 0
    assert t.attrs["best_fixed"] in {"dijkstra", "bidirectional_dijkstra"}
    assert set(res["cv"].index) == {"plain_tree", "random_forest", "svm", "mlp"}
    assert t.loc["plain_tree", "accuracy"] >= 0.95
    assert {"weights", "n_target", "family"} <= set(res["by_group"]["group_by"])


def test_evaluate_is_deterministic() -> None:
    df = synthetic_dataset(40, 3)
    a = ev.evaluate(df, small_cfg())["holdout"]
    b = ev.evaluate(df, small_cfg())["holdout"]
    cols = ["accuracy", "regret_mean", "search_s", "feature_s"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


# ------------------------------------------------------------------ plain-Python tree

def _fitted_plain_tree(df: pd.DataFrame) -> tuple[PlainTreeSelector, np.ndarray]:
    X = df[feature_columns(df, CFG, [0, 1])].to_numpy(dtype=float)
    model = PlainTreeSelector(max_depth=4, candidates=tuple(CANDIDATES), random_state=0)
    return model.fit(X, df["best_time"].to_numpy()), X


def test_plain_tree_choose_matches_masked_select() -> None:
    df = synthetic_dataset(40, 3)
    model, X = _fitted_plain_tree(df)
    rng = np.random.default_rng(1)
    mask = ev.applicability_mask(df, CANDIDATES) & (rng.random((len(df), len(CANDIDATES))) > 0.3)
    mask[:, CANDIDATES.index("dijkstra")] |= ~mask.any(axis=1)  # keep >= 1 applicable
    batch = select(model, X, mask, CANDIDATES)
    single = [model.choose(x, m) for x, m in zip(X.tolist(), mask.tolist())]
    assert batch.tolist() == single
    assert all(mask[i, c] for i, c in enumerate(single))


def test_plain_tree_matches_sklearn_tree() -> None:
    from sklearn.tree import DecisionTreeClassifier
    df = synthetic_dataset(40, 3)
    model, X = _fitted_plain_tree(df)
    ref = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, df["best_time"])
    np.testing.assert_allclose(model.predict_proba(X), ref.predict_proba(X))
    assert (model.predict_proba(X).sum(axis=1) == pytest.approx(1.0))


def test_inference_uses_plain_python_path() -> None:
    df = synthetic_dataset(20, 2)
    model, X = _fitted_plain_tree(df)
    calls = []
    original = model.choose
    model.choose = lambda x, m: calls.append(type(x)) or original(x, m)
    inference_ns(model, X, ev.applicability_mask(df, CANDIDATES), CANDIDATES, 5, 0)
    assert calls and all(t is list for t in calls)


def test_evaluate_accepts_other_tier_sets() -> None:
    df = synthetic_dataset(40, 3)
    t0 = ev.evaluate(df, small_cfg(), tiers=[0])["holdout"]
    t01 = ev.evaluate(df, small_cfg(), tiers=[0, 1])["holdout"]
    assert t01.loc["plain_tree", "feature_s"] > t0.loc["plain_tree", "feature_s"]
