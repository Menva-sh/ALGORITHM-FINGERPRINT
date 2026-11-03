"""Synthetic graph families, query sampling and instance storage, driven by config.yaml.

An *instance* = one graph + its (source, target) queries + metadata, saved as
data/raw_instances/<graph_id>.npz, with a summary row in data/raw_instances/index.csv.

Every graph gets its own seed derived from (global seed, graph_id), so an instance is
reproducible on its own and does not change if other families are added or removed.

Run:  python -m src.generators            # generate everything (resumable) + summary
      python -m src.generators --summary  # only print the summary of what exists
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.graph import Graph

COORD_FAMILIES = {"grid_2d", "random_geometric"}


@dataclass(frozen=True)
class InstanceSpec:
    """Everything needed to build one graph deterministically."""

    family: str
    n_target: int
    params: tuple[tuple[str, Any], ...]  # sorted (name, value) pairs
    weights: str
    rep: int

    @property
    def graph_id(self) -> str:
        p = "__".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}__n{self.n_target}__{p}__{self.weights}__r{self.rep}"


@dataclass
class Instance:
    spec: InstanceSpec
    seed: int
    graph: Graph
    queries: list[tuple[int, int]]
    meta: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ seeds and specs

def derive_seed(global_seed: int, key: str) -> int:
    """Stable 32-bit seed from the global seed and a string key."""
    digest = hashlib.sha256(f"{global_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def instance_specs(cfg: dict[str, Any]) -> list[InstanceSpec]:
    """All (family, size, params, weights, rep) combinations listed in the config."""
    gen = cfg["generation"]
    specs = []
    for family, fam_cfg in gen["families"].items():
        names = sorted(fam_cfg["params"])
        for values in itertools.product(*(fam_cfg["params"][k] for k in names)):
            params = tuple(zip(names, values))
            for n, weights, rep in itertools.product(
                gen["sizes"], fam_cfg["weights"], range(gen["graphs_per_setting"])
            ):
                specs.append(InstanceSpec(family, n, params, weights, rep))
    return specs


# ------------------------------------------------------------------ topology

def _topology(family: str, n: int, params: dict[str, Any], seed: int,
              rng: np.random.Generator) -> tuple[int, list[tuple[int, int]], list | None]:
    """Return (n_nodes, edge list, coords or None) for one family."""
    if family == "erdos_renyi":
        g = nx.fast_gnp_random_graph(n, params["avg_degree"] / (n - 1), seed=seed)
        return n, list(g.edges()), None
    if family == "barabasi_albert":
        g = nx.barabasi_albert_graph(n, params["m"], seed=seed)
        return n, list(g.edges()), None
    if family == "watts_strogatz":
        g = nx.watts_strogatz_graph(n, params["k"], params["p"], seed=seed)
        return n, list(g.edges()), None
    if family == "random_geometric":
        radius = math.sqrt(params["avg_degree"] / (n * math.pi))
        g = nx.random_geometric_graph(n, radius, seed=seed)
        pos = nx.get_node_attributes(g, "pos")
        return n, list(g.edges()), [tuple(pos[i]) for i in range(n)]
    if family == "grid_2d":
        return _grid_with_obstacles(n, params["obstacle_fraction"], rng)
    raise ValueError(f"unknown family {family!r}")


def _grid_with_obstacles(n: int, obstacle_fraction: float,
                         rng: np.random.Generator) -> tuple[int, list[tuple[int, int]], list]:
    """4-connected side x side grid (side = round(sqrt(n))) with random cells removed."""
    side = max(2, round(math.sqrt(n)))
    cells = side * side
    blocked = set(rng.choice(cells, size=round(obstacle_fraction * cells), replace=False).tolist())
    new_id: dict[int, int] = {}
    coords = []
    for c in range(cells):
        if c not in blocked:
            new_id[c] = len(coords)
            coords.append((float(c % side), float(c // side)))
    edges = []
    for c, u in new_id.items():
        x, y = c % side, c // side
        if x + 1 < side and c + 1 in new_id:
            edges.append((u, new_id[c + 1]))
        if y + 1 < side and c + side in new_id:
            edges.append((u, new_id[c + side]))
    return len(coords), edges, coords


# ------------------------------------------------------------------ weights

def _weights(kind: str, edges: list[tuple[int, int]], coords: list | None,
             dist_cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray | None:
    """Edge weights for one distribution (None means unweighted)."""
    m = len(edges)
    if kind == "unit":
        return None
    if kind == "uniform":
        return rng.uniform(dist_cfg["low"], dist_cfg["high"], size=m)
    if kind == "heavy_tailed":
        return dist_cfg["scale"] * (1.0 + rng.pareto(dist_cfg["shape"], size=m))
    if kind == "euclidean":
        if coords is None:
            raise ValueError("euclidean weights need coordinates")
        lengths = np.array([math.dist(coords[u], coords[v]) for u, v in edges])
        return lengths * rng.uniform(1.0, 1.0 + dist_cfg["noise"], size=m)
    raise ValueError(f"unknown weight distribution {kind!r}")


# ------------------------------------------------------------------ components and queries

def sample_queries(g: Graph, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to k distinct (s, t) pairs, s != t, both in the same connected component.

    s is uniform over nodes in components of size >= 2; t is uniform over the rest of
    s's component. Returns fewer than k only if fewer distinct pairs exist.
    """
    labels = g.component_labels()
    members: dict[int, list[int]] = {}
    for node, c in enumerate(labels):
        members.setdefault(c, []).append(node)
    eligible = [v for v in range(g.n) if len(members[labels[v]]) >= 2]
    if not eligible:
        return []
    max_pairs = sum(len(m) * (len(m) - 1) for m in members.values())
    target = min(k, max_pairs)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < target:
        s = eligible[rng.integers(len(eligible))]
        comp = members[labels[s]]
        t = comp[rng.integers(len(comp))]
        if t != s and (s, t) not in seen:
            seen.add((s, t))
            pairs.append((s, t))
    return pairs


# ------------------------------------------------------------------ build / save / load

def build_instance(spec: InstanceSpec, cfg: dict[str, Any]) -> Instance:
    """Generate the graph and queries for one spec (deterministic)."""
    seed = derive_seed(cfg["seed"], spec.graph_id)
    rng = np.random.default_rng(seed)
    params = dict(spec.params)
    n, edges, coords = _topology(spec.family, spec.n_target, params, seed, rng)
    w = _weights(spec.weights, edges, coords,
                 cfg["generation"]["weight_distributions"][spec.weights], rng)
    weighted = w is not None
    triples = [(u, v, float(x)) for (u, v), x in zip(edges, w)] if weighted else edges
    graph = Graph.from_edges(n, triples, directed=False, weighted=weighted, coords=coords)
    queries = sample_queries(graph, cfg["generation"]["queries_per_graph"],
                             np.random.default_rng(derive_seed(seed, "queries")))
    labels = graph.component_labels()
    sizes = np.bincount(labels) if n else np.array([0])
    meta = {
        "graph_id": spec.graph_id, "family": spec.family, "n_target": spec.n_target,
        "params": json.dumps(params), "weights": spec.weights, "rep": spec.rep,
        "seed": seed, "n": n, "m": graph.m, "avg_degree": 2 * graph.m / n if n else 0.0,
        "n_components": int(len(sizes)), "largest_component": int(sizes.max()),
        "has_coordinates": graph.has_coordinates, "weighted": weighted,
        "n_queries": len(queries),
    }
    return Instance(spec, seed, graph, queries, meta)


def save_instance(inst: Instance, out_dir: Path) -> Path:
    """Write one instance as a compressed .npz file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = inst.graph
    edges = g.edges()
    path = out_dir / f"{inst.spec.graph_id}.npz"
    np.savez_compressed(
        path,
        u=np.array([e[0] for e in edges], dtype=np.int32),
        v=np.array([e[1] for e in edges], dtype=np.int32),
        w=np.array([e[2] for e in edges], dtype=np.float64) if g.weighted else np.empty(0),
        coords=np.array(g.coords, dtype=np.float64) if g.coords else np.empty((0, 2)),
        queries=np.array(inst.queries, dtype=np.int32).reshape(-1, 2),
        meta=np.array(json.dumps(inst.meta)),
    )
    return path


def load_meta(path: Path) -> dict[str, Any]:
    """Read only the metadata of a saved instance."""
    with np.load(path) as data:
        return json.loads(str(data["meta"]))


def load_instance(path: Path) -> tuple[Graph, list[tuple[int, int]], dict[str, Any]]:
    """Read a saved instance back into (Graph, queries, meta)."""
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        u, v, w = data["u"].tolist(), data["v"].tolist(), data["w"].tolist()
        coords = [tuple(c) for c in data["coords"].tolist()] or None
        queries = [tuple(q) for q in data["queries"].tolist()]
    edges = list(zip(u, v, w)) if meta["weighted"] else list(zip(u, v))
    g = Graph.from_edges(meta["n"], edges, directed=False, weighted=meta["weighted"],
                         coords=coords)
    return g, queries, meta


def generate_all(cfg: dict[str, Any], out_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Build and save every instance in the config (skips files that already exist)."""
    specs = instance_specs(cfg)
    rows = []
    for i, spec in enumerate(specs):
        path = out_dir / f"{spec.graph_id}.npz"
        if path.exists():
            rows.append(load_meta(path))
        else:
            inst = build_instance(spec, cfg)
            save_instance(inst, out_dir)
            rows.append(inst.meta)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(specs)} instances")
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "index.csv", index=False)
    return index


def summarize(index: pd.DataFrame) -> pd.DataFrame:
    """Per-family counts and size/density statistics."""
    return index.groupby("family").agg(
        graphs=("graph_id", "count"),
        queries=("n_queries", "sum"),
        weighted=("weighted", "sum"),
        with_coords=("has_coordinates", "sum"),
        n_min=("n", "min"), n_max=("n", "max"),
        avg_degree_mean=("avg_degree", "mean"),
        disconnected=("n_components", lambda c: int((c > 1).sum())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="only summarise index.csv")
    args = parser.parse_args()
    cfg = load_config()
    out_dir = PROJECT_ROOT / cfg["paths"]["raw_instances"]
    if args.summary:
        index = pd.read_csv(out_dir / "index.csv")
    else:
        print(f"Generating {len(instance_specs(cfg))} instances into {out_dir}")
        index = generate_all(cfg, out_dir)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nPer family:")
        print(summarize(index).round(2).to_string())
        print("\nGraphs per family x weights:")
        print(pd.crosstab(index["family"], index["weights"], margins=True).to_string())
        print("\nGraphs per family x size:")
        print(pd.crosstab(index["family"], index["n_target"], margins=True).to_string())


if __name__ == "__main__":
    main()
