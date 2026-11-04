"""Real graphs (testing only): load edge-list files and OpenStreetMap road networks.

Input files in paths.real_graphs (data/real_graphs):
  <name>.edges   one edge per line: "u v" or "u v w"; lines starting with # or % are
                 comments; node ids may be any tokens (SNAP / Network Repository format).
                 Edges are treated as UNDIRECTED; duplicates keep the smallest weight.
  <name>.coords  optional, "id x y" per line (projected coordinates, e.g. metres)
A graph is weighted if a third column exists and not every weight is 1.

Road network:  python -m src.real_graphs --osm "<place name>" --name <name>
               (needs `pip install osmnx` and internet; writes <name>.edges/.coords with
               edge length in metres and projected node coordinates)
Build:         python -m src.real_graphs
               -> one instance per file in the `real` profile's raw_instances + index.csv
               then: python -m src.benchmark --profile real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.generators import (Instance, InstanceSpec, derive_seed, sample_queries,
                            save_instance)
from src.graph import Graph


def read_edge_list(edges_path: Path, coords_path: Path | None = None
                   ) -> tuple[Graph, dict[str, int]]:
    """Graph from an edge-list file (and optional coordinates); returns (graph, id map)."""
    coords_text = (coords_path.read_text(encoding="utf-8")
                   if coords_path is not None and coords_path.exists() else None)
    try:
        return parse_edge_list(edges_path.read_text(encoding="utf-8"), coords_text)
    except ValueError as e:
        raise ValueError(f"{edges_path.name}: {e}") from e


def parse_edge_list(edges_text: str, coords_text: str | None = None
                    ) -> tuple[Graph, dict[str, int]]:
    """Graph from edge-list text (and optional "id x y" coordinate text)."""
    ids: dict[str, int] = {}
    raw: list[tuple[int, int, float]] = []
    has_weight_col = False
    for line in edges_text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith(("#", "%")):
            continue
        u, v = (ids.setdefault(p, len(ids)) for p in parts[:2])
        if len(parts) >= 3:
            has_weight_col = True
            raw.append((u, v, float(parts[2])))
        else:
            raw.append((u, v, 1.0))
    weighted = has_weight_col and any(w != 1.0 for _, _, w in raw)
    if not ids:
        raise ValueError("no edges found")
    coords = None
    if coords_text is not None:
        xy: dict[int, tuple[float, float]] = {}
        for line in coords_text.splitlines():
            parts = line.split()
            if parts and not parts[0].startswith(("#", "%")) and parts[0] in ids:
                xy[ids[parts[0]]] = (float(parts[1]), float(parts[2]))
        if len(xy) != len(ids):
            raise ValueError(f"coordinates missing for {len(ids) - len(xy)} nodes")
        coords = [xy[i] for i in range(len(ids))]
    edges = raw if weighted else [(u, v) for u, v, _ in raw]
    g = Graph.from_edges(len(ids), edges, directed=False, weighted=weighted, coords=coords)
    return g, ids


def build_instance(name: str, g: Graph, cfg: dict[str, Any]) -> Instance:
    """Instance (graph + queries + metadata) for one real graph."""
    weights = ("euclidean" if g.has_coordinates else "real") if g.weighted else "unit"
    spec = InstanceSpec("real", g.n, (("name", name),), weights, 0)
    seed = derive_seed(cfg["seed"], spec.graph_id)
    queries = sample_queries(g, cfg["generation"]["queries_per_graph"],
                             np.random.default_rng(derive_seed(seed, "queries")))
    sizes = np.bincount(g.component_labels())
    meta = {"graph_id": spec.graph_id, "family": "real", "n_target": g.n,
            "params": json.dumps({"name": name}), "weights": weights, "rep": 0, "seed": seed,
            "n": g.n, "m": g.m, "avg_degree": 2 * g.m / g.n, "n_components": int(len(sizes)),
            "largest_component": int(sizes.max()), "has_coordinates": g.has_coordinates,
            "weighted": g.weighted, "n_queries": len(queries)}
    return Instance(spec, seed, g, queries, meta)


def build_all(cfg: dict[str, Any], src_dir: Path, out_dir: Path) -> pd.DataFrame:
    """One saved instance per <name>.edges file in src_dir, plus index.csv."""
    rows = []
    for edges in sorted(src_dir.glob("*.edges")):
        g, _ = read_edge_list(edges, edges.with_suffix(".coords"))
        inst = build_instance(edges.stem, g, cfg)
        save_instance(inst, out_dir)
        rows.append(inst.meta)
        print(f"{edges.stem}: n={g.n} m={g.m} weighted={g.weighted} "
              f"coords={g.has_coordinates} queries={len(inst.queries)}")
    index = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_dir / "index.csv", index=False)
    return index


def osm_to_files(place: str, name: str, out_dir: Path) -> None:
    """Download a drivable road network with OSMnx; write <name>.edges and <name>.coords."""
    import osmnx as ox

    G = ox.project_graph(ox.graph_from_place(place, network_type="drive"))
    to_undirected = getattr(getattr(ox, "convert", None), "to_undirected", None) \
        or ox.utils_graph.get_undirected
    G = to_undirected(G)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.edges", "w", encoding="utf-8") as f:
        f.write(f"# OpenStreetMap drive network of {place}; weight = length in metres\n")
        for u, v, data in G.edges(data=True):
            f.write(f"{u} {v} {data['length']}\n")
    with open(out_dir / f"{name}.coords", "w", encoding="utf-8") as f:
        for node, data in G.nodes(data=True):
            f.write(f"{node} {data['x']} {data['y']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real graphs for testing.")
    parser.add_argument("--osm", help="place name to download a road network for")
    parser.add_argument("--name", help="file name for the downloaded network")
    args = parser.parse_args()
    cfg = load_config(profile="real")
    src = PROJECT_ROOT / cfg["paths"]["real_graphs"]
    if args.osm:
        osm_to_files(args.osm, args.name or "road", src)
    index = build_all(cfg, src, PROJECT_ROOT / cfg["paths"]["raw_instances"])
    print(f"{len(index)} real graphs -> {cfg['paths']['raw_instances']}")


if __name__ == "__main__":
    main()
