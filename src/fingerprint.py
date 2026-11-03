"""Tiered graph fingerprints. Every feature function is timed with perf_counter_ns.

Tier-0  free metadata, O(1)          n_nodes, n_edges, density, is_weighted, has_coordinates
Tier-1  one linear scan, O(V+E)      degree stats + entropy, weight stats
Tier-2  sampled estimates            clustering on a node sample, diameter by BFS sweeps
Tier-3  exact and costly             exact diameter (iFUB), exact clustering, adjacency spectrum

All features are computed in pure Python on the shared Graph structure (same language as
the algorithms), except the spectrum, which uses scipy's ARPACK; its cost is therefore an
UNDER-estimate relative to the Python algorithms.

Graphs are treated as undirected. Diameters are hop diameters of the largest connected
component (ties broken by smallest node id). Average clustering includes nodes with
degree < 2 as 0, like networkx.average_clustering.

Run:  python -m src.fingerprint   # tier-cost report on a sample of saved instances
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from src.config import PROJECT_ROOT, load_config
from src.graph import Graph

Features = dict[str, float]
FeatureFn = Callable[[Graph, dict[str, Any], np.random.Generator], Features]


@dataclass
class Fingerprint:
    features: Features
    tier_cost_ns: dict[int, int]       # total runtime per tier
    function_cost_ns: dict[str, int]   # runtime per feature function


# ------------------------------------------------------------------ helpers

def _bfs(g: Graph, src: int) -> tuple[list[int], list[int]]:
    """Hop distances from src (-1 if unreached) and nodes in visit order."""
    dist = [-1] * g.n
    dist[src] = 0
    order = [src]
    queue = deque([src])
    adj = g.adj
    while queue:
        u = queue.popleft()
        du = dist[u] + 1
        for v, _ in adj[u]:
            if dist[v] < 0:
                dist[v] = du
                order.append(v)
                queue.append(v)
    return dist, order


def _largest_component(g: Graph) -> list[int]:
    """Nodes of the largest connected component."""
    labels = g.component_labels()
    sizes = Counter(labels)
    best = max(sizes, key=lambda c: (sizes[c], -c))
    return [v for v in range(g.n) if labels[v] == best]


def _local_clustering(u: int, nbr: Callable[[int], set[int]]) -> float:
    """Fraction of neighbour pairs of u that are connected."""
    nu = nbr(u)
    d = len(nu)
    if d < 2:
        return 0.0
    links = sum(len(nu & nbr(v)) for v in nu)  # each neighbour edge counted twice
    return links / (d * (d - 1))


# ------------------------------------------------------------------ tier 0

def tier0_metadata(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """O(1) metadata stored on the graph."""
    n, m = g.n, g.m
    pairs = n * (n - 1) if g.directed else n * (n - 1) / 2
    return {
        "n_nodes": float(n), "n_edges": float(m),
        "density": m / pairs if pairs else 0.0,
        "is_weighted": float(g.weighted), "has_coordinates": float(g.has_coordinates),
    }


# ------------------------------------------------------------------ tier 1

def degree_stats(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Mean, population variance, max and Shannon entropy (nats) of the degree distribution."""
    degrees = [len(a) for a in g.adj]
    n = len(degrees)
    mean = sum(degrees) / n
    var = sum((d - mean) ** 2 for d in degrees) / n
    entropy = -sum((c / n) * math.log(c / n) for c in Counter(degrees).values())
    return {"deg_mean": mean, "deg_var": var, "deg_max": float(max(degrees)),
            "deg_entropy": entropy}


def weight_stats(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Min, max, mean and population std of edge weights (constants if unweighted)."""
    if not g.weighted:
        return {"w_min": 1.0, "w_max": 1.0, "w_mean": 1.0, "w_std": 0.0}
    # Scanning adjacency lists sees each undirected edge twice; that leaves these stats unchanged.
    weights = [w for a in g.adj for _, w in a]
    if not weights:
        return {"w_min": 0.0, "w_max": 0.0, "w_mean": 0.0, "w_std": 0.0}
    k = len(weights)
    mean = sum(weights) / k
    std = math.sqrt(sum((w - mean) ** 2 for w in weights) / k)
    return {"w_min": min(weights), "w_max": max(weights), "w_mean": mean, "w_std": std}


# ------------------------------------------------------------------ tier 2

def clustering_estimate(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Average local clustering over a uniform sample of nodes (exact if n <= sample)."""
    size = min(params["tier2"]["clustering_sample_size"], g.n)
    sample = rng.choice(g.n, size=size, replace=False).tolist()
    cache: dict[int, set[int]] = {}

    def nbr(x: int) -> set[int]:
        if x not in cache:
            cache[x] = {v for v, _ in g.adj[x]}
        return cache[x]

    return {"clustering_est": sum(_local_clustering(u, nbr) for u in sample) / size}


def diameter_estimate(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Lower bound on the LCC diameter from repeated BFS sweeps.

    Start at a random LCC node; each sweep runs a BFS and jumps to the farthest node.
    """
    lcc = _largest_component(g)
    node = lcc[rng.integers(len(lcc))]
    best = 0
    for _ in range(params["tier2"]["diameter_bfs_sweeps"]):
        dist, order = _bfs(g, node)
        node = order[-1]
        best = max(best, dist[node])
    return {"diameter_est": float(best)}


# ------------------------------------------------------------------ tier 3

def diameter_exact(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Exact LCC hop diameter with iFUB (Crescenzi et al. 2013), rooted at the max-degree node.

    Processes BFS levels from the deepest up; stops once the best eccentricity found
    exceeds twice the next level, which bounds every remaining node's eccentricity.
    """
    lcc = _largest_component(g)
    root = max(lcc, key=lambda v: (len(g.adj[v]), -v))
    dist, order = _bfs(g, root)
    i = dist[order[-1]]
    levels: list[list[int]] = [[] for _ in range(i + 1)]
    for v in order:
        levels[dist[v]].append(v)
    lb, ub = i, 2 * i
    while ub > lb:
        bi = 0
        for v in levels[i]:
            d, o = _bfs(g, v)
            bi = max(bi, d[o[-1]])
        if max(lb, bi) > 2 * (i - 1):
            return {"diameter_exact": float(max(lb, bi))}
        lb, ub, i = max(lb, bi), 2 * (i - 1), i - 1
    return {"diameter_exact": float(lb)}


def clustering_exact(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Average local clustering over all nodes."""
    sets = [{v for v, _ in a} for a in g.adj]
    total = sum(_local_clustering(u, sets.__getitem__) for u in range(g.n))
    return {"clustering_exact": total / g.n}


def spectral(g: Graph, params: dict[str, Any], rng: np.random.Generator) -> Features:
    """Largest adjacency eigenvalue and gap to the second (unweighted structure)."""
    n = g.n
    if g.m == 0:
        return {"spectral_radius": 0.0, "spectral_gap": 0.0}
    rows = [u for u in range(n) for _ in g.adj[u]]
    cols = [v for a in g.adj for v, _ in a]
    a = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    if n <= 50:
        vals = np.linalg.eigvalsh(a.toarray())[-2:]
    else:
        try:
            vals = np.sort(eigsh(a, k=2, which="LA", v0=np.ones(n), return_eigenvectors=False))
        except ArpackNoConvergence:
            vals = np.linalg.eigvalsh(a.toarray())[-2:]
    return {"spectral_radius": float(vals[-1]), "spectral_gap": float(vals[-1] - vals[-2])}


# ------------------------------------------------------------------ extraction

TIER_FUNCTIONS: dict[int, list[FeatureFn]] = {
    0: [tier0_metadata],
    1: [degree_stats, weight_stats],
    2: [clustering_estimate, diameter_estimate],
    3: [diameter_exact, clustering_exact, spectral],
}


def extract(g: Graph, tiers: tuple[int, ...] | list[int] = (0, 1, 2, 3),
            cfg: dict[str, Any] | None = None, seed: int | None = None) -> Fingerprint:
    """Compute the requested tiers; each feature function is timed separately."""
    cfg = cfg or load_config()
    params = cfg["features"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    features: Features = {}
    tier_cost: dict[int, int] = {}
    fn_cost: dict[str, int] = {}
    for tier in sorted(tiers):
        tier_cost[tier] = 0
        for fn in TIER_FUNCTIONS[tier]:
            start = perf_counter_ns()
            out = fn(g, params, rng)
            elapsed = perf_counter_ns() - start
            features.update(out)
            fn_cost[fn.__name__] = elapsed
            tier_cost[tier] += elapsed
    return Fingerprint(features, tier_cost, fn_cost)


# ------------------------------------------------------------------ cost report

def cost_report(cfg: dict[str, Any]) -> pd.DataFrame:
    """Extract all tiers on a seeded sample of saved instances; one row per graph."""
    from src.generators import derive_seed, load_instance

    raw = PROJECT_ROOT / cfg["paths"]["raw_instances"]
    index = pd.read_csv(raw / "index.csv")
    k = cfg["features"]["cost_report_graphs_per_cell"]
    sample = index.groupby(["family", "n_target"]).sample(n=k, random_state=cfg["seed"])
    rows = []
    for gid in sample["graph_id"]:
        g, _, meta = load_instance(raw / f"{gid}.npz")
        fp = extract(g, cfg=cfg, seed=derive_seed(cfg["seed"], gid + ":features"))
        row = {"graph_id": gid, "family": meta["family"], "n_target": meta["n_target"],
               "n": g.n, "m": g.m}
        row.update({f"tier{t}_ms": ns / 1e6 for t, ns in fp.tier_cost_ns.items()})
        row.update({f"{name}_ms": ns / 1e6 for name, ns in fp.function_cost_ns.items()})
        row.update(fp.features)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = cost_report(cfg)
    out = PROJECT_ROOT / cfg["paths"]["results"] / "fingerprint_costs.csv"
    df.to_csv(out, index=False)
    tier_cols = [f"tier{t}_ms" for t in range(4)]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"{len(df)} graphs, saved to {out}\n")
        print("Mean feature cost per tier (ms) by graph size:")
        print(df.groupby("n_target")[tier_cols].mean().round(3).to_string())
        print("\nMean feature cost per tier (ms) by family:")
        print(df.groupby("family")[tier_cols].mean().round(3).to_string())
        fn_cols = [c for c in df.columns if c.endswith("_ms") and not c.startswith("tier")]
        print("\nMean cost per feature function (ms) by graph size:")
        print(df.groupby("n_target")[fn_cols].mean().round(3).to_string())
        print("\nTier-2 estimate vs Tier-3 exact:")
        err = pd.DataFrame({
            "family": df["family"],
            "diam_est/exact": df["diameter_est"] / df["diameter_exact"],
            "clust_abs_err": (df["clustering_est"] - df["clustering_exact"]).abs(),
        })
        print(err.groupby("family").agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
