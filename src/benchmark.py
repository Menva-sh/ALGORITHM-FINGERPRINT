"""Timing harness: features, runtimes, op counts and labels for every (graph, query).

Per graph:   extract all feature tiers once (each function timed).
Per query:   1. run every applicable COUNTING algorithm once -> op counts; check that all
                return the same cost (raise otherwise: that is a bug)
             2. warm-up runs of every TIMED (counter-free) algorithm
             3. `repeats` rounds; each round times every algorithm once in a seeded random
                order (so slow drift does not favour one algorithm); GC is off while timing
             label = median time; noise = IQR / median
Only the search call is timed (graph loading and feature extraction are outside the timer).

Output: one CSV per graph in benchmark.out_dir (written atomically, so the run is
resumable and safe to stop), merged into paths.dataset.

Parallel runs: one worker process per CPU in benchmark.cpus, each pinned to its own CPU.
Use `--calibrate` to see per-CPU speed and choose CPUs of ONE type on separate physical cores.

Run:  python -m src.benchmark --calibrate         # per-CPU speed check
      python -m src.benchmark [--cpus 0,2,4,6] [--limit N]
      python -m src.benchmark --merge-only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import time
from multiprocessing import Process
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd
import psutil

from src.algorithms import ALGORITHMS, TIMED_ALGORITHMS, applicable_algorithms, same_cost
from src.config import PROJECT_ROOT, load_config
from src.fingerprint import extract
from src.generators import derive_seed, load_instance
from src.graph import Graph


class CostMismatchError(RuntimeError):
    """Raised when candidate algorithms disagree on the shortest-path cost."""


# ------------------------------------------------------------------ one query

def count_ops(g: Graph, s: int, t: int, algos: list[str],
              rel_tol: float) -> tuple[float, dict[str, dict[str, int]]]:
    """Run the counting versions once; return the agreed cost and op counts per algorithm."""
    cost, ops = None, {}
    for a in algos:
        c, _, o = ALGORITHMS[a](g, s, t)
        if cost is None:
            cost = c
        elif not same_cost(c, cost, rel_tol):
            raise CostMismatchError(f"{a} returned {c}, {algos[0]} returned {cost} for ({s}, {t})")
        ops[a] = o
    return float(cost), ops


def time_query(g: Graph, s: int, t: int, algos: list[str], warmup: int, repeats: int,
               rng: random.Random) -> dict[str, list[int]]:
    """Per-algorithm list of `repeats` wall-clock times (ns) of the counter-free search."""
    for a in algos:
        for _ in range(warmup):
            TIMED_ALGORITHMS[a](g, s, t)
    samples: dict[str, list[int]] = {a: [] for a in algos}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            order = list(algos)
            rng.shuffle(order)
            for a in order:
                fn = TIMED_ALGORITHMS[a]
                start = perf_counter_ns()
                fn(g, s, t)
                samples[a].append(perf_counter_ns() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def make_labels(times: dict[str, float], ops: dict[str, int], candidates: list[str],
                tie_rel: float) -> dict[str, Any]:
    """Best algorithm by time and by op count, runner-up, relative margin and tie flag.

    Exact equal values are broken by the order of `candidates` (deterministic).
    """
    rank = {a: i for i, a in enumerate(candidates)}
    by_time = sorted(times, key=lambda a: (times[a], rank[a]))
    best, second = by_time[0], by_time[1]
    margin = (times[second] - times[best]) / times[best] if times[best] > 0 else 0.0
    return {
        "best_time": best, "second_time": second, "margin_rel": margin,
        "tie": margin <= tie_rel,
        "best_ops": min(ops, key=lambda a: (ops[a], rank[a])),
    }


# ------------------------------------------------------------------ one graph

def benchmark_graph(path: Path, cfg: dict[str, Any], cpu: int = -1) -> pd.DataFrame:
    """All rows (one per query) for one saved instance."""
    bcfg = cfg["benchmark"]
    candidates = cfg["algorithms"]["candidates"]
    g, queries, meta = load_instance(path)
    gid = meta["graph_id"]
    tiers = bcfg.get("feature_tiers", sorted(cfg["features"]["tiers"]))
    fp = extract(g, tiers=tiers, cfg=cfg, seed=derive_seed(cfg["seed"], gid + ":features"))
    algos = applicable_algorithms(g)
    rng = random.Random(derive_seed(cfg["seed"], gid + ":timing"))

    base: dict[str, Any] = {k: meta[k] for k in
                            ["graph_id", "family", "n_target", "params", "weights", "rep"]}
    base.update({f"f_{k}": v for k, v in fp.features.items()})
    base.update({f"cost_tier{t}_ns": ns for t, ns in fp.tier_cost_ns.items()})
    base.update({f"cost_{name}_ns": ns for name, ns in fp.function_cost_ns.items()})
    base.update({"cpu": cpu, "machine": platform.node()})

    rows = []
    for qi, (s, t) in enumerate(queries):
        cost, ops = count_ops(g, s, t, algos, bcfg["cost_rel_tolerance"])
        samples = time_query(g, s, t, algos, bcfg["warmup_runs"], bcfg["repeats"], rng)
        row = dict(base, query_idx=qi, source=s, target=t, path_cost=cost)
        medians, totals = {}, {}
        for a in candidates:
            ok = a in algos
            row[f"app_{a}"] = ok
            if not ok:
                for col in ["time", "iqr_rel", "pops", "relax", "ops"]:
                    row[f"{col}_{a}"] = np.nan
                continue
            q25, med, q75 = np.percentile(samples[a], [25, 50, 75])
            medians[a] = float(med)
            totals[a] = ops[a]["nodes_popped"] + ops[a]["edges_relaxed"]
            row.update({f"time_{a}": med, f"iqr_rel_{a}": (q75 - q25) / med,
                        f"pops_{a}": ops[a]["nodes_popped"],
                        f"relax_{a}": ops[a]["edges_relaxed"], f"ops_{a}": totals[a]})
        row.update(make_labels(medians, totals, candidates, bcfg["tie_threshold_rel"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ workers

def result_path(out_dir: Path, graph_id: str) -> Path:
    return out_dir / f"{graph_id}.csv"


def pending_graphs(raw_dir: Path, out_dir: Path, limit: int | None = None) -> list[Path]:
    """Instance files without a finished result, smallest graphs first."""
    index = pd.read_csv(raw_dir / "index.csv").sort_values(["n_target", "graph_id"])
    if limit is not None:
        index = index.head(limit)
    return [raw_dir / f"{gid}.npz" for gid in index["graph_id"]
            if not result_path(out_dir, gid).exists()]


def pin_to_cpu(cpu: int | None) -> None:
    if cpu is not None:
        psutil.Process().cpu_affinity([cpu])


def wait_for_ac(cfg: dict[str, Any], label: str, poll_s: float = 15.0) -> None:
    """Block while on battery (if benchmark.require_ac_power); timings must not mix power states."""
    if not cfg["benchmark"]["require_ac_power"] or on_ac_power():
        return
    print(f"[{label}] on battery: paused until AC power returns", flush=True)
    while not on_ac_power():
        time.sleep(poll_s)
    print(f"[{label}] AC power back: resuming", flush=True)


def timed_on_ac(cfg: dict[str, Any], label: str, work: Any) -> Any:
    """Run work() with AC power before AND after; redo it if power was lost meanwhile."""
    while True:
        wait_for_ac(cfg, label)
        result = work()
        if not cfg["benchmark"]["require_ac_power"] or on_ac_power():
            return result
        print(f"[{label}] power lost during measurement: discarding and redoing", flush=True)


def run_worker(paths: list[Path], out_dir: Path, cfg: dict[str, Any], cpu: int | None,
               verbose: bool = True) -> None:
    """Benchmark the given instances on one (optionally pinned) CPU; atomic per-graph writes."""
    pin_to_cpu(cpu)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(paths):
        final = result_path(out_dir, path.stem)
        if final.exists():
            continue
        start = time.time()
        df = timed_on_ac(cfg, f"cpu {cpu}",
                         lambda: benchmark_graph(path, cfg, cpu=-1 if cpu is None else cpu))
        df["ac_power"] = on_ac_power()
        tmp = final.with_suffix(f".tmp{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, final)
        if verbose:
            print(f"[cpu {cpu}] {i + 1}/{len(paths)} {path.stem} {time.time() - start:.1f}s",
                  flush=True)


def merge(out_dir: Path, dataset_path: Path) -> pd.DataFrame:
    """Concatenate per-graph CSVs into the dataset (sorted, deterministic)."""
    files = sorted(out_dir.glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["graph_id", "query_idx"]).reset_index(drop=True)
    df.to_csv(dataset_path, index=False)
    return df


# ------------------------------------------------------------------ machine checks

def on_ac_power() -> bool:
    """True if plugged in or no battery present."""
    battery = psutil.sensors_battery()
    return battery is None or bool(battery.power_plugged)


def cpu_topology() -> pd.DataFrame | None:
    """Windows only: physical core and efficiency class per logical CPU (higher = P-core)."""
    if platform.system() != "Windows":
        return None
    import ctypes
    import struct

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    need = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = ctypes.create_string_buffer(need.value)
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        return None
    rows, off = [], 0
    while off < need.value:
        size, _ = struct.unpack_from("<II", buf, off)
        _, _, logical, core, llc, _, eff = struct.unpack_from("<IHBBBBB", buf, off + 8)
        rows.append({"cpu": logical, "physical_core": core, "llc": llc, "efficiency_class": eff})
        off += size
    return pd.DataFrame(rows)


def calibrate(cpus: list[int], repeats: int = 15) -> pd.DataFrame:
    """Median time of a fixed Dijkstra workload on each CPU (to spot P- vs E-cores)."""
    import networkx as nx

    nxg = nx.grid_2d_graph(45, 45)
    idx = {v: i for i, v in enumerate(nxg.nodes)}
    g = Graph.from_edges(len(idx), [(idx[u], idx[v], 1.5) for u, v in nxg.edges])
    s, t = idx[(0, 0)], idx[(44, 44)]
    original = psutil.Process().cpu_affinity()
    rows = []
    try:
        for cpu in cpus:
            pin_to_cpu(cpu)
            for _ in range(3):
                TIMED_ALGORITHMS["dijkstra"](g, s, t)
            ts = []
            for _ in range(repeats):
                start = perf_counter_ns()
                TIMED_ALGORITHMS["dijkstra"](g, s, t)
                ts.append(perf_counter_ns() - start)
            rows.append({"cpu": cpu, "median_ms": statistics.median(ts) / 1e6,
                         "iqr_rel": (np.percentile(ts, 75) - np.percentile(ts, 25))
                         / statistics.median(ts)})
    finally:
        psutil.Process().cpu_affinity(original)
    return pd.DataFrame(rows)


def write_run_info(out_dir: Path, cpus: list[int | None]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.node(),
            "processor": platform.processor(), "python": platform.python_version(),
            "cpus": cpus, "ac_power": on_ac_power(), "cpu_freq": psutil.cpu_freq()._asdict()}
    topo = cpu_topology()
    if topo is not None:
        info["cpu_topology"] = topo.to_dict(orient="records")
    (out_dir / f"run_info_{int(time.time())}.json").write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------ CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all instances.")
    parser.add_argument("--cpus", help="comma-separated logical CPUs, one worker each")
    parser.add_argument("--limit", type=int, help="only the first N graphs (smallest first)")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--profile", help="config profile, e.g. large")
    args = parser.parse_args()

    cfg = load_config(profile=args.profile)
    if args.allow_battery:
        cfg["benchmark"]["require_ac_power"] = False  # workers then neither wait nor redo
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_instances"]
    out_dir = PROJECT_ROOT / cfg["benchmark"]["out_dir"]
    dataset = PROJECT_ROOT / cfg["paths"]["dataset"]

    if args.calibrate:
        table = calibrate(list(range(psutil.cpu_count())))
        topo = cpu_topology()
        if topo is not None:
            table = topo.merge(table, on="cpu")
        print(table.round(3).to_string(index=False))
        return
    if not args.merge_only:
        if cfg["benchmark"]["require_ac_power"] and not on_ac_power() and not args.allow_battery:
            raise SystemExit("On battery power: plug in (or pass --allow-battery for a test run).")
        cpus: list[int | None] = ([int(c) for c in args.cpus.split(",")] if args.cpus
                                  else cfg["benchmark"]["cpus"] or [None])
        paths = pending_graphs(raw_dir, out_dir, args.limit)
        print(f"{len(paths)} graphs to benchmark on CPUs {cpus}", flush=True)
        write_run_info(out_dir, cpus)
        if len(cpus) == 1:
            run_worker(paths, out_dir, cfg, cpus[0])
        else:
            procs = [Process(target=run_worker, args=(paths[i::len(cpus)], out_dir, cfg, cpu))
                     for i, cpu in enumerate(cpus)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
    df = merge(out_dir, dataset)
    print(f"dataset: {len(df)} rows from {df['graph_id'].nunique()} graphs -> {dataset}")


if __name__ == "__main__":
    main()
