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
