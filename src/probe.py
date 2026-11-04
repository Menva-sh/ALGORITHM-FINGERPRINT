"""Probing features: run the default algorithm (Dijkstra from s) for a tiny budget of k
settled nodes and describe what it saw. These are PER-QUERY features (static
fingerprints are per graph, so they give every query on a graph the same choice).

Features (all floats):
  probe_settled        nodes settled before stopping (<= k; fewer if t or the
                       component is reached first)
  probe_found_target   1 if t was settled within the budget
  probe_queue_final    heap size when the probe stopped
  probe_queue_mean/max heap size over the probe (sampled after every settled node)
  probe_growth         least-squares slope of heap size vs settled count (frontier growth)
  probe_branch_mean/var newly discovered nodes per settled node (frontier branching)
  probe_radius         distance of the last settled node from s

probe time = median wall-clock time of the whole call (search + summary), in ns.

Run:  python -m src.probe   # all queries, all k in config; writes features.probe.out
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from heapq import heappop, heappush
from multiprocessing import Process
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.graph import Graph

INF = math.inf


def _slope(ys: list[int]) -> float:
    """Least-squares slope of ys against 1..len(ys) (0 if fewer than 2 points)."""
    n = len(ys)
    if n < 2:
        return 0.0
    mean_x = (n + 1) / 2
    mean_y = sum(ys) / n
    num = sum((i + 1 - mean_x) * (y - mean_y) for i, y in enumerate(ys))
    den = sum((i + 1 - mean_x) ** 2 for i in range(n))
    return num / den


def probe_features(g: Graph, s: int, t: int, k: int) -> dict[str, float]:
    """Run Dijkstra from s until k nodes are settled (or t is settled); summarise it."""
    dist = {s: 0.0}
    heap = [(0.0, s)]
    adj = g.adj
    settled, found, radius = 0, False, 0.0
    queue_sizes: list[int] = []
    new_counts: list[int] = []
    while heap and settled < k:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        settled += 1
        radius = d
        if u == t:
            found = True
            break
        new = 0
        for v, w in adj[u]:
            nd = d + w
            old = dist.get(v, INF)
            if nd < old:
                if old == INF:
                    new += 1
                dist[v] = nd
                heappush(heap, (nd, v))
        new_counts.append(new)
        queue_sizes.append(len(heap))
    n = len(new_counts)
    branch_mean = sum(new_counts) / n if n else 0.0
    return {
        "probe_settled": float(settled),
        "probe_found_target": float(found),
        "probe_queue_final": float(len(heap)),
        "probe_queue_mean": sum(queue_sizes) / n if n else 0.0,
        "probe_queue_max": float(max(queue_sizes)) if n else 0.0,
        "probe_growth": _slope(queue_sizes),
        "probe_branch_mean": branch_mean,
        "probe_branch_var": sum((c - branch_mean) ** 2 for c in new_counts) / n if n else 0.0,
        "probe_radius": radius,
    }


def timed_probe(g: Graph, s: int, t: int, k: int, repeats: int) -> tuple[dict[str, float], int]:
    """Probe features and the median time (ns) of `repeats` runs."""
    times, feats = [], {}
    for _ in range(repeats):
        start = perf_counter_ns()
        feats = probe_features(g, s, t, k)
        times.append(perf_counter_ns() - start)
    return feats, int(statistics.median(times))


# ------------------------------------------------------------------ dataset

def probe_graph(path: Path, k_values: list[int], repeats: int) -> list[dict[str, Any]]:
    """Probe rows for every query of one saved instance and every k."""
    from src.generators import load_instance

    g, queries, meta = load_instance(path)
    rows = []
    for qi, (s, t) in enumerate(queries):
        for k in k_values:
            feats, ns = timed_probe(g, s, t, k, repeats)
            rows.append({"graph_id": meta["graph_id"], "query_idx": qi, "k": k,
                         **feats, "probe_ns": ns})
    return rows


def run_worker(paths: list[Path], out_file: Path, cfg: dict[str, Any], cpu: int | None) -> None:
    """Probe the given instances on one (optionally pinned) CPU; one CSV per worker."""
    from src.benchmark import pin_to_cpu, timed_on_ac

    pin_to_cpu(cpu)
    p = cfg["features"]["probe"]
    rows = []
    for i, path in enumerate(paths):
        rows += timed_on_ac(cfg, f"cpu {cpu}",
                            lambda: probe_graph(path, p["k_values"], p["repeats"]))
        if (i + 1) % 200 == 0:
            print(f"[cpu {cpu}] {i + 1}/{len(paths)}", flush=True)
    pd.DataFrame(rows).to_csv(out_file, index=False)


def main() -> None:
    from src.benchmark import on_ac_power

    parser = argparse.ArgumentParser(description="Compute probe features for all queries.")
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--profile", help="config profile, e.g. large")
    args = parser.parse_args()
    cfg = load_config(profile=args.profile)
    if args.allow_battery:
        cfg["benchmark"]["require_ac_power"] = False
    if cfg["benchmark"]["require_ac_power"] and not on_ac_power():
        raise SystemExit("On battery power: plug in (or pass --allow-battery for a test run).")
    raw = PROJECT_ROOT / cfg["paths"]["raw_instances"]
    out = PROJECT_ROOT / cfg["features"]["probe"]["out"]
    index = pd.read_csv(raw / "index.csv").sort_values(["n_target", "graph_id"])
    paths = [raw / f"{gid}.npz" for gid in index["graph_id"]]
    cpus = cfg["benchmark"]["cpus"] or [None]
    parts = [out.with_name(f"{out.stem}_part{i}.csv") for i in range(len(cpus))]
    start = time.time()
    procs = [Process(target=run_worker, args=(paths[i::len(cpus)], parts[i], cfg, cpu))
             for i, cpu in enumerate(cpus)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.sort_values(["graph_id", "query_idx", "k"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    for p in parts:
        p.unlink()
    print(f"{len(df)} probe rows -> {out} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
