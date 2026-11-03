"""Stage 6: tier-set experiment, Pareto frontier and per-size payoff."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src import evaluate as ev
from src.experiments import tiers as tx
from tests.test_models_evaluate import small_cfg, synthetic_dataset


def tier_cfg() -> dict:
    cfg = small_cfg()
    cfg["experiments"] = copy.deepcopy(cfg["experiments"])
    cfg["experiments"]["tiers"]["tier_sets"] = [[0], [0, 1]]
    return cfg


def test_tier_name() -> None:
    assert tx.tier_name([0]) == "T0" and tx.tier_name([0, 1, 2]) == "T0+1+2"


def test_pareto_mask() -> None:
    cost = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    loss = np.array([0.5, 0.3, 0.3, 0.6, 0.5])
    # point 2 dominated by 1 (same loss, higher cost); 3 dominated by 1; duplicates 0 and 4 kept
    assert tx.pareto_mask(cost, loss).tolist() == [True, True, False, False, True]


def test_evaluate_can_restrict_models_and_skip_cv() -> None:
    res = ev.evaluate(synthetic_dataset(30, 3), small_cfg(), [0], model_names=["plain_tree"],
                      cv=False)
    assert res["cv"] is None
    assert "plain_tree" in res["holdout"].index and "svm" not in res["holdout"].index
    assert len(res["_test_index"]) == len(res["_choices"])


def test_run_summary_and_sizes() -> None:
    df = synthetic_dataset(60, 4)
    summary, size_df = tx.run(df, tier_cfg())
    assert len(summary) == 2 * 2
    cost = summary.groupby("tier_set")["feature_cost_ms"].first()
    assert cost["T0+1"] > cost["T0"]
    assert summary["pareto_regret"].any() and summary["pareto_accuracy"].any()
    assert set(size_df["n_target"]) == set(df["n_target"])
    assert (size_df["pays_off"] == (size_df["net_rel_best_fixed"] < 1)).all()
    assert (size_df["net_rel_best_fixed_amortised"] <= size_df["net_rel_best_fixed"]).all()
    # per-size totals must add up to the overall net time
    for (ts, model), d in size_df.groupby(["tier_set", "model"]):
        s = summary[(summary.tier_set == ts) & (summary.model == model)].iloc[0]
        assert d["queries"].sum() > 0
        assert s["net_rel_best_fixed"] == pytest.approx(
            s["net_s"] / (s["net_s"] / s["net_rel_best_fixed"]))


def test_payoff_report_and_figures(tmp_path) -> None:
    summary, size_df = tx.run(synthetic_dataset(40, 3), tier_cfg())
    payoff = tx.payoff_report(size_df, "plain_tree")
    assert ("per_query", "best_tier_set") in payoff.columns
    assert set(payoff[("per_query", "best_tier_set")]) <= {"T0", "T0+1"}
    tx.plot_pareto(summary, tmp_path / "a.png")
    tx.plot_net(summary, "plain_tree", tmp_path / "b.png")
    tx.plot_by_size(size_df, "plain_tree", tmp_path / "c.png")
    assert all((tmp_path / f).stat().st_size > 0 for f in ["a.png", "b.png", "c.png"])
