"""Stage 2: generators follow the config, are deterministic, and queries are valid."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from src.algorithms import ALGORITHMS, applicable_algorithms
from src.config import load_config
from src.generators import (COORD_FAMILIES, InstanceSpec, build_instance,
                            generate_all, instance_specs, load_instance, sample_queries,
                            save_instance)
from src.graph import Graph

CFG = load_config()


def spec(family: str, n: int, weights: str = "unit", rep: int = 0, **params) -> InstanceSpec:
    return InstanceSpec(family, n, tuple(sorted(params.items())), weights, rep)


def small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["generation"]["sizes"] = [60, 150]
    cfg["generation"]["graphs_per_setting"] = 1
    return cfg


# ------------------------------------------------------------------ specs

def test_spec_count_matches_config_and_ids_unique() -> None:
    gen = CFG["generation"]
    expected = 0
    for fam in gen["families"].values():
        combos = math.prod(len(v) for v in fam["params"].values())
        expected += combos * len(gen["sizes"]) * len(fam["weights"]) * gen["graphs_per_setting"]
    specs = instance_specs(CFG)
    assert len(specs) == expected
    assert len({s.graph_id for s in specs}) == len(specs)


def test_coordinate_families_only_get_coordinate_weights() -> None:
    for name, fam in CFG["generation"]["families"].items():
        if name not in COORD_FAMILIES:
            assert "euclidean" not in fam["weights"], name


# ------------------------------------------------------------------ sizes and densities

def test_erdos_renyi_size_and_degree() -> None:
    inst = build_instance(spec("erdos_renyi", 1000, avg_degree=6), CFG)
    assert inst.graph.n == 1000
    assert inst.meta["avg_degree"] == pytest.approx(6, rel=0.15)


def test_barabasi_albert_edge_count() -> None:
    for m in CFG["generation"]["families"]["barabasi_albert"]["params"]["m"]:
        g = build_instance(spec("barabasi_albert", 1000, m=m), CFG).graph
        assert g.n == 1000 and g.m == m * (1000 - m)


def test_watts_strogatz_edge_count() -> None:
    g = build_instance(spec("watts_strogatz", 1000, k=8, p=0.1), CFG).graph
    assert g.n == 1000 and g.m == 1000 * 8 // 2


def test_random_geometric_size_and_degree() -> None:
    inst = build_instance(spec("random_geometric", 1000, avg_degree=8), CFG)
    assert inst.graph.n == 1000 and inst.graph.has_coordinates
    assert inst.meta["avg_degree"] == pytest.approx(8, rel=0.2)  # boundary effect lowers it


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25])
def test_grid_size_and_obstacles(frac: float) -> None:
    g = build_instance(spec("grid_2d", 1000, obstacle_fraction=frac), CFG).graph
    side = round(math.sqrt(1000))
    assert g.n == side * side - round(frac * side * side)
    if frac == 0.0:
        assert g.m == 2 * side * (side - 1)
    assert g.has_coordinates


# ------------------------------------------------------------------ weights and flags

@pytest.mark.parametrize("family,params", [
    ("erdos_renyi", {"avg_degree": 6}), ("barabasi_albert", {"m": 3}),
    ("watts_strogatz", {"k": 4, "p": 0.1}), ("grid_2d", {"obstacle_fraction": 0.1}),
    ("random_geometric", {"avg_degree": 8}),
])
def test_flags_per_family_and_weights(family: str, params: dict) -> None:
    for weights in CFG["generation"]["families"][family]["weights"]:
        g = build_instance(spec(family, 300, weights, **params), CFG).graph
        assert g.weighted == (weights != "unit")
        assert g.has_coordinates == (family in COORD_FAMILIES)


def test_weight_ranges() -> None:
    dists = CFG["generation"]["weight_distributions"]
    uni = build_instance(spec("erdos_renyi", 500, "uniform", avg_degree=6), CFG).graph
    w = [x for _, _, x in uni.edges()]
    assert dists["uniform"]["low"] <= min(w) and max(w) <= dists["uniform"]["high"]
    heavy = build_instance(spec("erdos_renyi", 500, "heavy_tailed", avg_degree=6), CFG).graph
    w = [x for _, _, x in heavy.edges()]
    assert min(w) >= dists["heavy_tailed"]["scale"] and max(w) > 10 * np.median(w)


def test_euclidean_weights_make_heuristic_scale_at_least_one() -> None:
    for family, params in [("grid_2d", {"obstacle_fraction": 0.0}),
                           ("random_geometric", {"avg_degree": 8})]:
        g = build_instance(spec(family, 400, "euclidean", **params), CFG).graph
        assert g.heuristic_scale >= 1.0 - 1e-12


# ------------------------------------------------------------------ determinism

def test_same_seed_gives_identical_instance() -> None:
    for family, params in [("erdos_renyi", {"avg_degree": 3}), ("grid_2d", {"obstacle_fraction": 0.25}),
                           ("random_geometric", {"avg_degree": 4})]:
        s = spec(family, 300, "unit", **params)
        a, b = build_instance(s, CFG), build_instance(s, CFG)
        assert a.graph.edges() == b.graph.edges()
        assert a.graph.coords == b.graph.coords
        assert a.queries == b.queries


def test_different_rep_or_seed_gives_different_graph() -> None:
    a = build_instance(spec("erdos_renyi", 300, rep=0, avg_degree=6), CFG)
    b = build_instance(spec("erdos_renyi", 300, rep=1, avg_degree=6), CFG)
    other = copy.deepcopy(CFG)
    other["seed"] += 1
    c = build_instance(spec("erdos_renyi", 300, rep=0, avg_degree=6), other)
    assert a.graph.edges() != b.graph.edges()
    assert a.graph.edges() != c.graph.edges()


# ------------------------------------------------------------------ queries

def test_queries_same_component_and_all_algorithms_reach_target() -> None:
    k = CFG["generation"]["queries_per_graph"]
    for s in instance_specs(small_cfg()):
        inst = build_instance(s, CFG)
        labels = inst.graph.component_labels()
        assert len(inst.queries) == k, s.graph_id
        assert len(set(inst.queries)) == len(inst.queries)
        for src, tgt in inst.queries:
            assert src != tgt and labels[src] == labels[tgt]
            for name in applicable_algorithms(inst.graph):
                assert ALGORITHMS[name](inst.graph, src, tgt)[0] < math.inf


def test_sample_queries_edge_cases() -> None:
    rng = np.random.default_rng(0)
    assert sample_queries(Graph.from_edges(5, []), 10, rng) == []
    pairs = sample_queries(Graph.from_edges(5, [(1, 3, 1.0)]), 10, rng)
    assert sorted(pairs) == [(1, 3), (3, 1)]


# ------------------------------------------------------------------ storage

def test_save_load_roundtrip(tmp_path) -> None:
    for s in [spec("random_geometric", 200, "euclidean", avg_degree=8),
              spec("barabasi_albert", 200, "unit", m=3)]:
        inst = build_instance(s, CFG)
        g, queries, meta = load_instance(save_instance(inst, tmp_path))
        assert g.edges() == inst.graph.edges()
        assert g.coords == inst.graph.coords and g.weighted == inst.graph.weighted
        assert queries == inst.queries and meta == inst.meta


def test_generate_all_is_resumable(tmp_path) -> None:
    cfg = small_cfg()
    first = generate_all(cfg, tmp_path, verbose=False)
    mtimes = {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.npz")}
    second = generate_all(cfg, tmp_path, verbose=False)
    assert len(first) == len(instance_specs(cfg)) == len(mtimes)
    assert {p: p.stat().st_mtime_ns for p in tmp_path.glob("*.npz")} == mtimes
    assert first.equals(second)
