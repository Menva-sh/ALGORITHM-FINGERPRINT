"""Stage 4: timing harness, labels, resumability and the label report."""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import pandas as pd
import pytest

import src.benchmark as bm
from src.algorithms import ALGORITHMS, applicable_algorithms
from src.config import load_config
from src.experiments.label_report import balance_check, build_report
from src.generators import InstanceSpec, build_instance, generate_all, save_instance

CFG = load_config()
CANDIDATES = CFG["algorithms"]["candidates"]


def fast_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["benchmark"]["warmup_runs"] = 1
    cfg["benchmark"]["repeats"] = 3
    cfg["benchmark"]["require_ac_power"] = False  # tests must not wait for a charger
    cfg["generation"]["queries_per_graph"] = 3
    return cfg


@pytest.fixture
def grid_instance(tmp_path):
    """Unweighted grid with coordinates: all five algorithms applicable."""
    spec = InstanceSpec("grid_2d", 100, (("obstacle_fraction", 0.1),), "unit", 0)
    inst = build_instance(spec, fast_cfg())
    return save_instance(inst, tmp_path), inst


# ------------------------------------------------------------------ labels

def test_make_labels_picks_fastest_and_flags_tie() -> None:
    lab = bm.make_labels({"dijkstra": 100.0, "astar": 103.0, "bfs": 200.0},
                         {"dijkstra": 50, "astar": 20, "bfs": 50}, CANDIDATES, 0.05)
    assert lab["best_time"] == "dijkstra" and lab["second_time"] == "astar"
    assert lab["margin_rel"] == pytest.approx(0.03) and lab["tie"]
    assert lab["best_ops"] == "astar"


def test_make_labels_no_tie_and_deterministic_ops_tiebreak() -> None:
    lab = bm.make_labels({"dijkstra": 100.0, "bfs": 150.0}, {"bfs": 7, "dijkstra": 7},
                         CANDIDATES, 0.05)
    assert not lab["tie"] and lab["margin_rel"] == pytest.approx(0.5)
    assert lab["best_ops"] == "dijkstra"  # equal ops -> earlier in candidate order


# ------------------------------------------------------------------ one query

def test_time_query_returns_repeats_per_algorithm(grid_instance) -> None:
    _, inst = grid_instance
    s, t = inst.queries[0]
    algos = applicable_algorithms(inst.graph)
    samples = bm.time_query(inst.graph, s, t, algos, 1, 5, random.Random(0))
    assert set(samples) == set(algos)
    assert all(len(v) == 5 and min(v) > 0 for v in samples.values())


def test_count_ops_detects_cost_mismatch(grid_instance, monkeypatch) -> None:
    _, inst = grid_instance
    s, t = inst.queries[0]
    real = ALGORITHMS["astar"]
    monkeypatch.setitem(ALGORITHMS, "astar",
                        lambda g, a, b: (real(g, a, b)[0] + 1.0,) + real(g, a, b)[1:])
    with pytest.raises(bm.CostMismatchError):
        bm.count_ops(inst.graph, s, t, applicable_algorithms(inst.graph), 1e-9)


# ------------------------------------------------------------------ one graph

def test_benchmark_graph_rows(grid_instance) -> None:
    path, inst = grid_instance
    df = bm.benchmark_graph(path, fast_cfg())
    assert len(df) == len(inst.queries)
    for tier, names in CFG["features"]["tiers"].items():
        assert all(f"f_{n}" in df for n in names)
        assert (df[f"cost_tier{tier}_ns"] > 0).all()
    for a in CANDIDATES:
        assert df[f"app_{a}"].all()
        assert (df[f"time_{a}"] > 0).all()
    for _, row in df.iterrows():
        s, t = int(row["source"]), int(row["target"])
        for a in CANDIDATES:
            _, _, ops = ALGORITHMS[a](inst.graph, s, t)
            assert row[f"pops_{a}"] == ops["nodes_popped"]
            assert row[f"relax_{a}"] == ops["edges_relaxed"]
        times = {a: row[f"time_{a}"] for a in CANDIDATES}
        assert row["best_time"] == min(times, key=times.get)
        assert math.isfinite(row["path_cost"])


def test_inapplicable_algorithms_are_masked(tmp_path) -> None:
    spec = InstanceSpec("erdos_renyi", 100, (("avg_degree", 6),), "uniform", 0)
    path = save_instance(build_instance(spec, fast_cfg()), tmp_path)
    df = bm.benchmark_graph(path, fast_cfg())
    for a in ["astar", "bfs", "bidirectional_bfs"]:
        assert not df[f"app_{a}"].any() and df[f"time_{a}"].isna().all()
    assert set(df["best_time"]) <= {"dijkstra", "bidirectional_dijkstra"}


# ------------------------------------------------------------------ resumable run + merge

def test_worker_is_resumable_and_merge_is_complete(tmp_path) -> None:
    cfg = fast_cfg()
    cfg["generation"]["sizes"] = [60]
    cfg["generation"]["graphs_per_setting"] = 1
    raw, out = tmp_path / "raw", tmp_path / "bench"
    index = generate_all(cfg, raw, verbose=False)
    subset = bm.pending_graphs(raw, out, limit=6)
    bm.run_worker(subset, out, cfg, cpu=None, verbose=False)
    mtimes = {p.name: p.stat().st_mtime_ns for p in out.glob("*.csv")}
    assert len(mtimes) == 6
    rest = bm.pending_graphs(raw, out)
    assert len(rest) == len(index) - 6 and not set(p.stem for p in rest) & {
        n[:-4] for n in mtimes}
    bm.run_worker(rest, out, cfg, cpu=None, verbose=False)
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.csv") if p.name in mtimes} == mtimes
    df = bm.merge(out, tmp_path / "dataset.csv")
    assert df["graph_id"].nunique() == len(index)
    assert len(df) == index["n_queries"].sum()
    assert not list(out.glob("*.tmp*"))


def test_pin_to_cpu_sets_affinity() -> None:
    import psutil
    original = psutil.Process().cpu_affinity()
    try:
        bm.pin_to_cpu(original[0])
        assert psutil.Process().cpu_affinity() == [original[0]]
    finally:
        psutil.Process().cpu_affinity(original)


# ------------------------------------------------------------------ report

def test_label_report_and_balance_check() -> None:
    df = pd.DataFrame({
        "family": ["a", "a", "b", "b", "b"], "n_target": [100] * 5, "weights": ["unit"] * 5,
        "best_time": ["dijkstra", "astar", "dijkstra", "dijkstra", "dijkstra"],
        "best_ops": ["dijkstra", "astar", "astar", "dijkstra", "dijkstra"],
        "tie": [False, True, False, False, False],
        **{f"iqr_rel_{a}": [0.01, 0.2, np.nan, 0.05, 0.02] for a in CANDIDATES},
    })
    report = build_report(df, CFG)
    assert report["by_family"].loc["b", "dijkstra"] == 1.0
    assert report["overall"].loc["all queries", "dijkstra"] == pytest.approx(0.8)
    assert report["noise"].loc["dijkstra", "share_over_10pct"] == pytest.approx(0.25)
    assert balance_check(df, 0.80) == (True, "dijkstra", pytest.approx(0.8))
    assert balance_check(df, 0.70)[0] is False


# ------------------------------------------------------------------ power guard

def test_timed_on_ac_redoes_work_when_power_is_lost(monkeypatch) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["benchmark"]["require_ac_power"] = True
    states = iter([True, False, True, True])  # before, after (lost!), before, after
    monkeypatch.setattr(bm, "on_ac_power", lambda: next(states))
    calls = []
    assert bm.timed_on_ac(cfg, "t", lambda: calls.append(1) or len(calls)) == 2
    assert len(calls) == 2


def test_power_guard_off_never_checks(monkeypatch) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["benchmark"]["require_ac_power"] = False
    monkeypatch.setattr(bm, "on_ac_power", lambda: (_ for _ in ()).throw(AssertionError))
    assert bm.timed_on_ac(cfg, "t", lambda: 7) == 7


def test_worker_rows_record_power_state(tmp_path) -> None:
    cfg = fast_cfg()
    spec = InstanceSpec("erdos_renyi", 60, (("avg_degree", 3),), "unit", 0)
    path = save_instance(build_instance(spec, cfg), tmp_path / "raw")
    bm.run_worker([path], tmp_path / "out", cfg, cpu=None, verbose=False)
    assert "ac_power" in pd.read_csv(tmp_path / "out" / f"{path.stem}.csv")
