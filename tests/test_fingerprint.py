"""Stage 3: tiered fingerprints are correct, timed, and Tier-2 estimates are close to Tier-3."""
from __future__ import annotations

import math
import random
from collections import Counter

import networkx as nx
import numpy as np
import pytest
from scipy.stats import entropy

from src.config import load_config
from src.fingerprint import TIER_FUNCTIONS, extract
from src.generators import InstanceSpec, build_instance
from src.graph import Graph

CFG = load_config()


def from_nx(nxg: nx.Graph, weighted: bool, rng: random.Random) -> Graph:
    nxg = nx.convert_node_labels_to_integers(nxg)
    edges = [(u, v, rng.uniform(0.5, 20.0) if weighted else 1.0) for u, v in nxg.edges()]
    return Graph.from_edges(nxg.number_of_nodes(), edges, weighted=weighted)


def to_nx(g: Graph) -> nx.Graph:
    nxg = nx.Graph()
    nxg.add_nodes_from(range(g.n))
    nxg.add_weighted_edges_from(g.edges())
    return nxg


def small_graphs() -> list[Graph]:
    """Small graphs with trees, disconnected parts, triangles, isolated nodes."""
    rng = random.Random(CFG["seed"])
    out = []
    for i in range(40):
        n = rng.randint(1, 60)
        kind = i % 5
        if kind == 0:
            nxg = nx.gnp_random_graph(n, rng.uniform(0.02, 0.3), seed=i)
        elif kind == 1:
            nxg = nx.barabasi_albert_graph(max(n, 3), 2, seed=i)
        elif kind == 2:
            nxg = nx.watts_strogatz_graph(max(n, 5), 4, 0.2, seed=i)
        elif kind == 3:
            nxg = nx.random_tree(max(n, 1), seed=i) if hasattr(nx, "random_tree") \
                else nx.random_labeled_tree(max(n, 1), seed=i)
        else:
            nxg = nx.disjoint_union(nx.complete_graph(4), nx.cycle_graph(max(n, 3)))
        out.append(from_nx(nxg, weighted=(i % 2 == 0), rng=rng))
    return out


def lcc_diameter(nxg: nx.Graph) -> int:
    """networkx diameter of the largest component (ties -> smallest node id)."""
    comps = list(nx.connected_components(nxg))
    best = max(comps, key=lambda c: (len(c), -min(c)))
    return nx.diameter(nxg.subgraph(best))


# ------------------------------------------------------------------ exact values vs networkx

def test_tier0_matches_networkx() -> None:
    for g in small_graphs():
        f = extract(g, tiers=[0], cfg=CFG).features
        nxg = to_nx(g)
        assert f["n_nodes"] == nxg.number_of_nodes()
        assert f["n_edges"] == nxg.number_of_edges()
        assert f["density"] == pytest.approx(nx.density(nxg))
        assert f["is_weighted"] == float(g.weighted)
        assert f["has_coordinates"] == 0.0


def test_tier1_matches_networkx() -> None:
    for g in small_graphs():
        f = extract(g, tiers=[1], cfg=CFG).features
        nxg = to_nx(g)
        degs = np.array([d for _, d in nxg.degree()])
        assert f["deg_mean"] == pytest.approx(degs.mean())
        assert f["deg_var"] == pytest.approx(degs.var())
        assert f["deg_max"] == degs.max()
        assert f["deg_entropy"] == pytest.approx(entropy(list(Counter(degs.tolist()).values())))
        w = np.array([d["weight"] for _, _, d in nxg.edges(data=True)])
        if g.weighted and len(w):
            assert f["w_min"] == pytest.approx(w.min()) and f["w_max"] == pytest.approx(w.max())
            assert f["w_mean"] == pytest.approx(w.mean()) and f["w_std"] == pytest.approx(w.std())
        elif not g.weighted:
            assert (f["w_min"], f["w_max"], f["w_mean"], f["w_std"]) == (1.0, 1.0, 1.0, 0.0)


def test_tier3_matches_networkx() -> None:
    for g in small_graphs():
        f = extract(g, tiers=[3], cfg=CFG).features
        nxg = to_nx(g)
        assert f["diameter_exact"] == lcc_diameter(nxg)
        assert f["clustering_exact"] == pytest.approx(nx.average_clustering(nxg))
        if g.m:
            eig = np.linalg.eigvalsh(nx.to_numpy_array(nxg, weight=None))
            assert f["spectral_radius"] == pytest.approx(eig[-1], abs=1e-6)
            assert f["spectral_gap"] == pytest.approx(eig[-1] - eig[-2], abs=1e-6)


def test_exact_diameter_on_known_graphs() -> None:
    rng = random.Random(0)
    for nxg, d in [(nx.path_graph(10), 9), (nx.cycle_graph(11), 5),
                   (nx.grid_2d_graph(7, 4), 9), (nx.complete_graph(6), 1), (nx.empty_graph(1), 0)]:
        assert extract(from_nx(nxg, False, rng), tiers=[3], cfg=CFG).features["diameter_exact"] == d


def test_spectral_sparse_path_matches_dense() -> None:
    """n > 50 uses ARPACK; check it against a dense solve."""
    nxg = nx.gnp_random_graph(300, 0.03, seed=1)
    f = extract(from_nx(nxg, False, random.Random(0)), tiers=[3], cfg=CFG).features
    eig = np.linalg.eigvalsh(nx.to_numpy_array(nxg))
    assert f["spectral_radius"] == pytest.approx(eig[-1], abs=1e-6)
    assert f["spectral_gap"] == pytest.approx(eig[-1] - eig[-2], abs=1e-6)


# ------------------------------------------------------------------ tier 2 vs tier 3

MID_SIZE = [("erdos_renyi", {"avg_degree": 6}), ("barabasi_albert", {"m": 3}),
            ("watts_strogatz", {"k": 8, "p": 0.1}), ("grid_2d", {"obstacle_fraction": 0.1}),
            ("random_geometric", {"avg_degree": 8})]


@pytest.mark.parametrize("family,params", MID_SIZE)
def test_tier2_estimates_close_to_exact(family: str, params: dict) -> None:
    spec = InstanceSpec(family, 1000, tuple(sorted(params.items())), "unit", 0)
    g = build_instance(spec, CFG).graph
    f = extract(g, tiers=[2, 3], cfg=CFG).features
    assert f["diameter_est"] <= f["diameter_exact"]          # sweeps give a lower bound
    assert f["diameter_est"] >= 0.8 * f["diameter_exact"]
    assert abs(f["clustering_est"] - f["clustering_exact"]) <= 0.05


def test_tier2_is_exact_when_sample_covers_graph() -> None:
    g = from_nx(nx.watts_strogatz_graph(100, 6, 0.1, seed=3), False, random.Random(0))
    f = extract(g, tiers=[2, 3], cfg=CFG).features  # sample size 200 >= n
    assert f["clustering_est"] == pytest.approx(f["clustering_exact"])


# ------------------------------------------------------------------ timing and API

def test_feature_names_match_config_tiers() -> None:
    g = small_graphs()[1]
    for tier, names in CFG["features"]["tiers"].items():
        assert set(extract(g, tiers=[tier], cfg=CFG).features) == set(names), tier


def test_costs_recorded_per_function_and_tier() -> None:
    g = small_graphs()[2]
    fp = extract(g, tiers=[0, 1, 2, 3], cfg=CFG)
    assert set(fp.tier_cost_ns) == {0, 1, 2, 3}
    for tier, fns in TIER_FUNCTIONS.items():
        assert fp.tier_cost_ns[tier] == sum(fp.function_cost_ns[f.__name__] for f in fns)
    assert all(ns > 0 for ns in fp.function_cost_ns.values())


def test_only_requested_tiers_computed() -> None:
    fp = extract(small_graphs()[0], tiers=[0, 1], cfg=CFG)
    assert set(fp.tier_cost_ns) == {0, 1}
    assert "diameter_exact" not in fp.features and "clustering_est" not in fp.features


def test_tier2_deterministic_for_seed() -> None:
    spec = InstanceSpec("erdos_renyi", 1000, (("avg_degree", 6),), "unit", 0)
    g = build_instance(spec, CFG).graph
    a = extract(g, tiers=[2], cfg=CFG, seed=7).features
    b = extract(g, tiers=[2], cfg=CFG, seed=7).features
    assert a == b


def test_tier3_cost_grows_faster_than_tier1() -> None:
    """Sanity check on the cost tiers on a 2500-node graph."""
    spec = InstanceSpec("random_geometric", 2500, (("avg_degree", 8),), "unit", 0)
    fp = extract(build_instance(spec, CFG).graph, cfg=CFG)
    assert fp.tier_cost_ns[0] < fp.tier_cost_ns[1] < fp.tier_cost_ns[3]
    assert math.isfinite(fp.features["spectral_radius"])
