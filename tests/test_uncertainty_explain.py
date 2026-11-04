"""Stage 8: calibration, conformal sets, abstention, and explainability."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src import explain as ex
from src.experiments import uncertainty as ux
from src.models import (abstaining_cost, build_models, calibrated, class_scores,
                        conformal_sets, conformal_threshold, expected_calibration_error,
                        feature_columns, masked_probs)
from tests.test_models_evaluate import CANDIDATES, small_cfg, synthetic_dataset


# ------------------------------------------------------------------ probabilities and ECE

def test_masked_probs() -> None:
    scores = np.array([[0.2, 0.6, 0.2], [-np.inf, 0.0, -np.inf], [0.5, 0.5, 0.0]])
    mask = np.array([[True, False, True], [True, True, False], [True, True, True]])
    p = masked_probs(scores, mask)
    assert p.sum(axis=1) == pytest.approx(np.ones(3))
    assert p[0].tolist() == pytest.approx([0.5, 0.0, 0.5])
    assert p[1].tolist() == pytest.approx([0.5, 0.5, 0.0])  # no mass -> uniform over applicable


def test_expected_calibration_error() -> None:
    conf = np.array([0.95] * 100)
    assert expected_calibration_error(conf, np.array([1] * 95 + [0] * 5), 10) == pytest.approx(0.0)
    assert expected_calibration_error(conf, np.array([1] * 50 + [0] * 50), 10) == pytest.approx(0.45)


# ------------------------------------------------------------------ conformal

def test_conformal_threshold_finite_sample_quantile() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    # scores 0.1..0.4; level = ceil(5 * 0.5) / 4 = 0.75 -> 'higher' quantile = 0.3
    assert conformal_threshold(probs, np.zeros(4, int), 0.5) == pytest.approx(0.3)


def test_conformal_sets_respect_mask_and_are_never_empty() -> None:
    probs = np.array([[0.1, 0.1, 0.8], [0.4, 0.6, 0.0]])
    mask = np.array([[True, True, False], [True, True, False]])
    sets = conformal_sets(probs, mask, threshold=0.5)
    assert sets.any(axis=1).all()
    assert sets[1].tolist() == [False, True, False]  # 1 - 0.4 > 0.5, 1 - 0.6 <= 0.5


def test_conformal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=6000)
    labels = np.array([rng.choice(4, p=p) for p in probs])
    mask = np.ones_like(probs, dtype=bool)
    for cov in [0.8, 0.9]:
        thr = conformal_threshold(probs[:3000], labels[:3000], cov)
        sets = conformal_sets(probs[3000:], mask[3000:], thr)
        assert sets[np.arange(3000), labels[3000:]].mean() >= cov - 0.03


def test_abstaining_cost_policies() -> None:
    T = np.array([[10.0, 30.0, 50.0], [10.0, 30.0, 50.0]])
    probs = np.array([[0.2, 0.7, 0.1], [0.2, 0.7, 0.1]])
    sets = np.array([[False, True, False], [True, True, False]])  # row 1 unsure
    assert abstaining_cost(sets, probs, T, 2, "none").tolist() == [30.0, 30.0]
    assert abstaining_cost(sets, probs, T, 2, "fallback").tolist() == [30.0, 50.0]
    assert abstaining_cost(sets, probs, T, 2, "top2").tolist() == [30.0, 20.0]  # 2*min(30,10)
    with pytest.raises(ValueError):
        abstaining_cost(sets, probs, T, 2, "other")


def test_calibrated_model_gives_probabilities() -> None:
    df = synthetic_dataset(40, 3)
    X = df[feature_columns(df, small_cfg(), [0])].to_numpy()
    model = calibrated(build_models(small_cfg())["plain_tree"], X, df["best_time"].to_numpy(),
                       df["graph_id"].to_numpy(), small_cfg())
    p = masked_probs(class_scores(model, X, CANDIDATES), np.ones((len(X), 5), dtype=bool))
    assert p.sum(axis=1) == pytest.approx(np.ones(len(X)))


def unc_cfg() -> dict:
    cfg = small_cfg()
    cfg["experiments"] = copy.deepcopy(cfg["experiments"])
    cfg["experiments"]["uncertainty"]["models"] = ["plain_tree"]
    return cfg


def test_uncertainty_run_and_figures(tmp_path) -> None:
    res = ux.run(synthetic_dataset(80, 4), unc_cfg())
    a = res["abstention"]
    assert set(a["policy"]) == set(ux.POLICIES)
    assert len(a) == len(unc_cfg()["model"]["conformal_coverage"]) * len(ux.POLICIES)
    assert a["empirical_coverage"].between(0, 1).all() and (a["mean_set_size"] >= 1).all()
    assert set(res["calibration"]["variant"]) == {"raw", "calibrated"}
    ux.plot_reliability(res["reliability"], tmp_path / "r.png")
    ux.plot_abstention(a, "plain_tree", tmp_path / "a.png")
    assert (tmp_path / "r.png").stat().st_size > 0 and (tmp_path / "a.png").stat().st_size > 0


def test_split_train_calibration_by_graph() -> None:
    df = synthetic_dataset(40, 3)
    proper, cal = ux.split_train_calibration(df, np.arange(len(df)), 0.25, 0)
    assert not set(df.iloc[proper]["graph_id"]) & set(df.iloc[cal]["graph_id"])
    assert len(proper) + len(cal) == len(df)


# ------------------------------------------------------------------ explainability

def test_correlated_groups() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(500)
    X = np.column_stack([a, 2 * a + 0.001 * rng.random(500), rng.random(500), np.ones(500)])
    groups = ex.correlated_groups(X, 0.8)
    assert [0, 1] in groups and [2] in groups and [3] in groups
    assert ex.correlated_groups(X[:, :1], 0.8) == [[0]]


def test_grouped_permutation_ranks_informative_feature_first() -> None:
    rng = np.random.default_rng(0)
    n = 600
    X = np.column_stack([rng.integers(0, 2, n), rng.random(n)])
    y = np.where(X[:, 0] == 1, "bfs", "dijkstra")
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    T = np.full((n, len(CANDIDATES)), 200.0)
    label_idx = np.array([CANDIDATES.index(a) for a in y])
    T[np.arange(n), label_idx] = 100.0
    mask = np.ones_like(T, dtype=bool)
    perm = ex.grouped_permutation_importance(model, X, mask, label_idx, T, [[0], [1]],
                                             ["signal", "noise"], CANDIDATES, 3, 0)
    assert perm.iloc[0]["group"] == "signal" and perm.iloc[0]["accuracy_drop"] > 0.3
    assert abs(perm.set_index("group").loc["noise", "accuracy_drop"]) < 0.02


def test_rule_list_and_binary_conditions() -> None:
    X = np.array([[0, 100], [0, 5000], [1, 100], [1, 5000]] * 10, dtype=float)
    y = np.array(["bfs", "bidirectional_bfs", "dijkstra", "bidirectional_dijkstra"] * 10)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    rules = ex.rule_list(tree, ["is_weighted", "n_nodes"])
    assert len(rules) == tree.get_n_leaves() == 4
    assert any("is_weighted = 0" in r and "THEN bfs" in r for r in rules)
    assert all("100%" in r for r in rules)


def test_textbook_checks() -> None:
    df = pd.DataFrame({"weights": ["unit", "unit", "euclidean", "uniform"],
                       "n_target": [100, 5000, 100, 5000],
                       "best_time": ["bfs", "astar", "astar", "bidirectional_dijkstra"]})
    t = ex.textbook_checks(df, ["bfs", "bfs", "dijkstra", "bidirectional_dijkstra"]).set_index("advice")
    row = t.loc["Unweighted graphs: a BFS variant is fastest"]
    assert row["queries"] == 2 and row["holds_in_labels"] == 0.5 and row["rules_follow"] == 1.0


def test_explain_tier_set_end_to_end(tmp_path) -> None:
    cfg = small_cfg()
    cfg["experiments"] = copy.deepcopy(cfg["experiments"])
    cfg["experiments"]["explain"]["shap_samples"] = 50
    res = ex.explain_tier_set(synthetic_dataset(60, 3), cfg, [0])
    assert set(res["shap"].index) == set(cfg["features"]["tiers"][0])
    assert 0 <= res["distill"]["fidelity"] <= 1 and res["rules"]
    ex.plot_shap(res["shap"], "t", tmp_path / "s.png")
    ex.plot_permutation(res["permutation"], "t", tmp_path / "p.png")
    assert (tmp_path / "s.png").stat().st_size > 0
