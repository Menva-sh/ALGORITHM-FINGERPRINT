"""Logic behind the Streamlit demo (app/demo.py), kept here so it can be tested.

The demo's selector is the deployed one: a plain-Python tree on Tier-0 features, trained
on every graph of the main (synthetic) dataset.
"""
from __future__ import annotations

import random
from time import perf_counter_ns
from typing import Any

import numpy as np
import pandas as pd

from src.algorithms import ALGORITHMS, applicable_algorithms
from src.benchmark import count_ops, time_query
from src.graph import Graph
from src.models import (PlainTreeSelector, build_models, class_scores, feature_columns,
                        masked_probs)


def train_selector(df: pd.DataFrame, cfg: dict[str, Any]) -> PlainTreeSelector:
    """Plain tree on model.feature_tiers features, trained on all rows of df."""
    X = df[feature_columns(df, cfg, cfg["model"]["feature_tiers"])].to_numpy(dtype=float)
    return build_models(cfg)["plain_tree"].fit(X, df[cfg["model"]["label"]].to_numpy())


def recommend(model: PlainTreeSelector, features: dict[str, float], g: Graph,
              cfg: dict[str, Any]) -> dict[str, Any]:
    """Selector choice, probabilities over applicable algorithms, timing and SHAP values."""
    import shap

    C = cfg["algorithms"]["candidates"]
    names = [n for t in cfg["model"]["feature_tiers"] for n in cfg["features"]["tiers"][t]]
    x = [float(features[n]) for n in names]
    applicable = applicable_algorithms(g)
    mask = [a in applicable for a in C]
    start = perf_counter_ns()
    idx = model.choose(x, mask)
    inference_ns = perf_counter_ns() - start
    probs = masked_probs(class_scores(model, np.array([x]), C), np.array([mask]))[0]
    choice = C[idx]

    vals = shap.TreeExplainer(model.tree_).shap_values(np.array([x]))
    arr = np.stack(vals, axis=-1) if isinstance(vals, list) else np.asarray(vals)
    classes = list(model.tree_.classes_)
    contrib = (arr[0, :, classes.index(choice)] if choice in classes
               else np.zeros(len(names)))
    return {"choice": choice, "confidence": float(probs[idx]),
            "probs": {a: float(p) for a, p in zip(C, probs) if a in applicable},
            "inference_ns": inference_ns,
            "shap": pd.Series(contrib, index=names).sort_values(key=abs, ascending=False)}


def run_all(g: Graph, s: int, t: int, cfg: dict[str, Any]) -> pd.DataFrame:
    """Time every applicable algorithm on (s, t) like the benchmark (warm-up + median)."""
    b = cfg["benchmark"]
    algos = applicable_algorithms(g)
    cost, ops = count_ops(g, s, t, algos, b["cost_rel_tolerance"])
    samples = time_query(g, s, t, algos, b["warmup_runs"], b["repeats"], random.Random(0))
    rows = []
    for a in algos:
        _, path, _ = ALGORITHMS[a](g, s, t)
        rows.append({"algorithm": a, "median_us": float(np.median(samples[a])) / 1e3,
                     "nodes_popped": ops[a]["nodes_popped"],
                     "edges_relaxed": ops[a]["edges_relaxed"], "path_cost": cost,
                     "path_nodes": len(path)})
    return pd.DataFrame(rows).sort_values("median_us").reset_index(drop=True)


# ------------------------------------------------------------------ plain-language text

NAMES = {"dijkstra": "Dijkstra", "bidirectional_dijkstra": "Bidirectional Dijkstra",
         "astar": "A*", "bfs": "BFS", "bidirectional_bfs": "Bidirectional BFS"}


def reason(choice: str, features: dict[str, float]) -> str:
    """One-sentence, rule-of-thumb reason for a choice (matches the distilled rules)."""
    weighted, coords = features["is_weighted"] > 0, features["has_coordinates"] > 0
    if choice == "astar":
        what = ("Edges are weighted and nodes have map coordinates" if weighted
                else "Nodes have map coordinates")
        return f"{what}, so A* can use the straight-line distance to head toward the target."
    if choice == "bidirectional_bfs":
        what = ("The graph is unweighted and has no map coordinates" if not coords
                else "The graph is unweighted")
        return f"{what}, so searching from both ends at once is usually quickest."
    if choice == "bfs":
        return ("The graph is unweighted and fairly small, so a plain breadth-first search "
                "has the least overhead.")
    if choice == "bidirectional_dijkstra":
        what = "there are no coordinates to aim with" if not coords else "it is not distance-based"
        return f"Edges are weighted and {what}, so searching from both ends at once is usually quickest."
    return ("On graphs like this the extra bookkeeping of the other searches rarely pays off, "
            "so plain Dijkstra is expected to be quickest.")


def feature_label(name: str, value: float) -> str:
    """Readable label for a Tier-0 feature and its value."""
    if name == "is_weighted":
        return "Graph is weighted" if value > 0 else "Graph is unweighted"
    if name == "has_coordinates":
        return "Has map coordinates" if value > 0 else "No map coordinates"
    if name == "n_nodes":
        return f"Graph size ({int(value):,} nodes)"
    if name == "n_edges":
        return f"Number of edges ({int(value):,})"
    if name == "density":
        return f"Density ({value:.3g})"
    return name


def why_items(shap_values: pd.Series, features: dict[str, float],
              top: int = 3) -> list[tuple[str, float]]:
    """Features that pushed TOWARD the choice, as (label, share of the largest push)."""
    pos = shap_values[shap_values > 0].sort_values(ascending=False).head(top)
    if pos.empty:
        return []
    return [(feature_label(n, features[n]), float(v / pos.iloc[0])) for n, v in pos.items()]


def confidence_label(p: float) -> tuple[str, str]:
    """(text, kind) badge for the selector's probability of its top choice."""
    if p >= 0.8:
        return "Confident pick", "good"
    if p >= 0.5:
        return "Fairly sure", "good"
    return "Close call", "warn"


def nearest_graph(features: dict[str, float], graphs: pd.DataFrame, names: list[str]) -> int:
    """Row index of the benchmark graph whose (log-scaled, standardised) features are closest."""
    X = np.log1p(graphs[[f"f_{n}" for n in names]].to_numpy(dtype=float))
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    x = np.log1p(np.array([features[n] for n in names], dtype=float))
    return int(np.argmin((((X - mu) / sd - (x - mu) / sd) ** 2).sum(axis=1)))
