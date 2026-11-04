"""Evaluate selectors against baselines: accuracy, regret and NET time.

Split: graphs (not rows) are split into train/test (GroupShuffleSplit); GroupKFold CV on
the training graphs reports stability. Every graph's queries stay on one side.

Per test query, for a method choosing algorithm a:
  regret      = time(a) / time(best applicable) - 1
  net time    = feature time + inference time + time(a)
Feature time is charged in full to EVERY query (one query per graph: worst case) and,
separately, amortised over the graph's queries (features computed once per graph).

Baselines: oracle (best algorithm, zero overhead), best single fixed algorithm (the
always-applicable algorithm with the lowest total TRAIN time), always-Dijkstra, random
applicable algorithm (seeded), and running all applicable candidates one after another.

Run:  python -m src.evaluate      # regenerates results/model_*.csv and results/fig_*.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.models import (applicability_mask, build_models, feature_columns,  # noqa: E402
                        feature_cost_ns, inference_ns, select)

# Fixed colour per algorithm (validated categorical palette, slots 1-5): colour follows
# the entity in every figure.
ALGO_COLORS = {"dijkstra": "#2a78d6", "bidirectional_dijkstra": "#eb6834", "astar": "#1baf7a",
               "bfs": "#eda100", "bidirectional_bfs": "#e87ba4"}
SELECTOR_COLOR, BASELINE_COLOR = "#2a78d6", "#a3a29c"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


# ------------------------------------------------------------------ data

def time_matrix(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    """Median search time (ns) per row and candidate; NaN where not applicable."""
    return df[[f"time_{a}" for a in candidates]].to_numpy(dtype=float)


def split_by_graph(df: pd.DataFrame, test_fraction: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of train and test; every graph lies entirely on one side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train, test = next(gss.split(df, groups=df["graph_id"]))
    return train, test


def best_fixed_algorithm(T: np.ndarray, candidates: list[str]) -> int:
    """Index of the always-applicable algorithm with the lowest total time."""
    always = ~np.isnan(T).any(axis=0)
    totals = np.where(always, np.nan_to_num(T, nan=0.0).sum(axis=0), np.inf)
    return int(np.argmin(totals))


# ------------------------------------------------------------------ metrics

def choice_metrics(choice: np.ndarray, T: np.ndarray, label_idx: np.ndarray,
                   feat_ns: np.ndarray, feat_amortised_ns: np.ndarray, infer_ns: float,
                   tie_rel: float) -> dict[str, float]:
    """Accuracy, regret and time totals for one method's choices."""
    rows = np.arange(len(choice))
    chosen = T[rows, choice]
    if np.isnan(chosen).any():
        raise ValueError("a method chose an inapplicable algorithm")
    best = np.nanmin(T, axis=1)
    regret = chosen / best - 1.0
    search = chosen.sum()
    infer_total = infer_ns * len(choice)
    return {
        "accuracy": float((choice == label_idx).mean()),
        "accuracy_tie_aware": float((chosen <= best * (1 + tie_rel)).mean()),
        "regret_mean": float(regret.mean()),
        "regret_p95": float(np.quantile(regret, 0.95)),
        "search_s": search / 1e9,
        "feature_s": feat_ns.sum() / 1e9,
        "inference_s": infer_total / 1e9,
        "net_s": (feat_ns.sum() + infer_total + search) / 1e9,
        "net_amortised_s": (feat_amortised_ns.sum() + infer_total + search) / 1e9,
    }


def add_savings(table: pd.DataFrame, run_all_s: float) -> pd.DataFrame:
    """Net time saved relative to the best fixed algorithm and to running all candidates."""
    ref = table.loc["best_fixed", "net_s"]
    for col in ["net_s", "net_amortised_s"]:
        suffix = "" if col == "net_s" else "_amortised"
        table[f"saved_vs_best_fixed{suffix}"] = 1 - table[col] / ref
        table[f"saved_vs_run_all{suffix}"] = 1 - table[col] / run_all_s
    return table


# ------------------------------------------------------------------ experiment

def evaluate(df: pd.DataFrame, cfg: dict[str, Any], tiers: list[int] | None = None,
             model_names: list[str] | None = None, cv: bool = True,
             extra_features: list[str] | None = None,
             extra_cost_col: str | None = None) -> dict[str, Any]:
    """Train on train graphs, evaluate all methods on test graphs, CV on train graphs.

    tiers defaults to model.feature_tiers; model_names restricts the selectors trained
    (default: all); cv=False skips cross-validation. extra_features / extra_cost_col add
    PER-QUERY features (e.g. probes) and their cost column (ns); that cost is charged in
    full to its query and never amortised over the graph.
    """
    m = cfg["model"]
    candidates = cfg["algorithms"]["candidates"]
    tiers = m["feature_tiers"] if tiers is None else tiers
    tie = cfg["benchmark"]["tie_threshold_rel"]
    cols = feature_columns(df, cfg, tiers) + list(extra_features or [])
    X = df[cols].to_numpy(dtype=float)
    y = df[m["label"]].to_numpy()
    label_idx = np.array([candidates.index(a) for a in y])
    T = time_matrix(df, candidates)
    mask = applicability_mask(df, candidates)
    graph_feat = feature_cost_ns(df, tiers)
    query_feat = (df[extra_cost_col].to_numpy(dtype=float) if extra_cost_col
                  else np.zeros(len(df)))
    feat = graph_feat + query_feat
    feat_amortised = (graph_feat / df.groupby("graph_id")["graph_id"].transform("size")
                      .to_numpy() + query_feat)

    train, test = split_by_graph(df, m["test_fraction"], cfg["seed"])
    Tt, mt = T[test], mask[test]
    zeros = np.zeros(len(test))
    rows: dict[str, dict[str, float]] = {}
    choices: dict[str, np.ndarray] = {}

    def record(name: str, choice: np.ndarray, use_features: bool, infer: float) -> None:
        f, fa = (feat[test], feat_amortised[test]) if use_features else (zeros, zeros)
        rows[name] = choice_metrics(choice, Tt, label_idx[test], f, fa, infer, tie)
        choices[name] = choice

    fitted = {}
    for name, model in build_models(cfg).items():
        if model_names is not None and name not in model_names:
            continue
        model.fit(X[train], y[train])
        fitted[name] = model
        infer = inference_ns(model, X[test], mt, candidates, m["inference_timing_samples"],
                             cfg["seed"])
        record(name, select(model, X[test], mt, candidates), True, infer)

    record("oracle", np.nanargmin(Tt, axis=1), False, 0.0)
    fixed = best_fixed_algorithm(T[train], candidates)
    record("best_fixed", np.full(len(test), fixed), False, 0.0)
    record("always_dijkstra", np.full(len(test), candidates.index("dijkstra")), False, 0.0)
    rng = np.random.default_rng(cfg["seed"])
    record("random", np.array([rng.choice(np.flatnonzero(r)) for r in mt]), False, 0.0)

    run_all_s = np.nansum(Tt) / 1e9
    table = add_savings(pd.DataFrame(rows).T, run_all_s)
    table.loc["run_all", ["search_s", "net_s", "net_amortised_s"]] = run_all_s
    table.attrs["best_fixed"] = candidates[fixed]

    return {
        "holdout": table,
        "cv": cross_validate(X[train], y[train], df["graph_id"].to_numpy()[train],
                             T[train], mask[train], label_idx[train], cfg) if cv else None,
        "by_group": breakdown(df.iloc[test], choices, Tt, label_idx[test], tie,
                              feat[test], fitted, cfg),
        "_choices": pd.DataFrame({k: v for k, v in choices.items()}),
        "_test_labels": pd.Series(label_idx[test]),
        "_test_index": test,
    }


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, T: np.ndarray,
                   mask: np.ndarray, label_idx: np.ndarray, cfg: dict[str, Any]) -> pd.DataFrame:
    """GroupKFold on the training graphs: accuracy and regret mean +- std per model."""
    candidates = cfg["algorithms"]["candidates"]
    tie = cfg["benchmark"]["tie_threshold_rel"]
    out = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=cfg["model"]["cv_folds"])
                                    .split(X, y, groups)):
        for name, model in build_models(cfg).items():
            model = clone(model).fit(X[tr], y[tr])
            choice = select(model, X[va], mask[va], candidates)
            z = np.zeros(len(va))
            met = choice_metrics(choice, T[va], label_idx[va], z, z, 0.0, tie)
            out.append({"model": name, "fold": fold, **{k: met[k] for k in
                        ["accuracy", "accuracy_tie_aware", "regret_mean", "regret_p95"]}})
    res = pd.DataFrame(out).drop(columns="fold").groupby("model")
    return res.mean().add_suffix("_mean").join(res.std().add_suffix("_std"))


def breakdown(test_df: pd.DataFrame, choices: dict[str, np.ndarray], T: np.ndarray,
              label_idx: np.ndarray, tie: float, feat: np.ndarray, fitted: dict,
              cfg: dict[str, Any]) -> pd.DataFrame:
    """Selectors vs best fixed vs oracle per weight type, size and family."""
    out = []
    for by in ["weights", "n_target", "family"]:
        for value, idx in test_df.reset_index(drop=True).groupby(by).groups.items():
            idx = np.asarray(idx)
            z = np.zeros(len(idx))
            for name in [n for n in ["plain_tree", "random_forest", "best_fixed", "oracle"]
                         if n in choices]:
                f = feat[idx] if name in fitted else z
                met = choice_metrics(choices[name][idx], T[idx], label_idx[idx], f, z, 0.0, tie)
                out.append({"group_by": by, "group": value, "method": name, "queries": len(idx),
                            "accuracy": met["accuracy"], "regret_mean": met["regret_mean"],
                            "search_s": met["search_s"], "net_s_no_inference": met["net_s"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures

def _style(ax: plt.Axes) -> None:
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_net_time(table: pd.DataFrame, path: Path) -> None:
    """Net time of every method relative to the best fixed algorithm (lower is better)."""
    ref = table.loc["best_fixed", "net_s"]
    rel = (table["net_s"] / ref).sort_values(ascending=False)
    selectors = {"plain_tree", "random_forest", "svm", "mlp"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [SELECTOR_COLOR if n in selectors else BASELINE_COLOR for n in rel.index]
    ax.barh(rel.index, rel.values, color=colors, height=0.6)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
    for i, v in enumerate(rel.values):
        ax.text(v + 0.05, i, f"{v:.2f}x", va="center", fontsize=9, color=TEXT,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_xlabel("net time relative to best fixed algorithm (lower is better)", color=MUTED)
    ax.set_title("Net time on test graphs (features charged per query)", loc="left",
                 fontsize=11, color=TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regret_cdf(choices: pd.DataFrame, T: np.ndarray, path: Path) -> None:
    """Share of test queries with regret <= x, per method."""
    best = np.nanmin(T, axis=1)
    series = [("plain_tree", "#2a78d6"), ("random_forest", "#eb6834"), ("svm", "#1baf7a"),
              ("mlp", "#eda100"), ("best_fixed", "#6250d6"), ("random", "#a3a29c")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    rows = np.arange(len(T))
    for name, color in series:
        regret = np.sort(T[rows, choices[name].to_numpy()] / best - 1.0)
        ax.plot(np.maximum(regret, 1e-3), np.arange(1, len(regret) + 1) / len(regret),
                color=color, linewidth=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("regret = time(chosen) / time(best) - 1  (values below 0.1% drawn at 0.1%)",
                  color=MUTED)
    ax.set_ylabel("share of test queries", color=MUTED)
    ax.set_title("Regret distribution on test graphs", loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(choice: np.ndarray, labels: np.ndarray, candidates: list[str],
                   path: Path) -> None:
    """Row-normalised confusion matrix: true best algorithm vs chosen algorithm."""
    k = len(candidates)
    counts = np.zeros((k, k))
    np.add.at(counts, (labels, choice), 1)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    short = [a.replace("bidirectional_", "bi-") for a in candidates]
    ax.set_xticks(range(k), short, rotation=30, ha="right", fontsize=9, color=MUTED)
    ax.set_yticks(range(k), short, fontsize=9, color=MUTED)
    for i in range(k):
        for j in range(k):
            if counts[i, j]:
                ax.text(j, i, f"{share[i, j]:.2f}\n({int(counts[i, j])})", ha="center",
                        va="center", fontsize=8, color="white" if share[i, j] > 0.6 else TEXT)
    ax.set_xlabel("chosen by random forest", color=MUTED)
    ax.set_ylabel("true fastest", color=MUTED)
    ax.set_title("Random forest choices on test graphs", loc="left", fontsize=11, color=TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ CLI

def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    out = PROJECT_ROOT / cfg["paths"]["results"]
    candidates = cfg["algorithms"]["candidates"]
    res = evaluate(df, cfg)
    holdout = res["holdout"]
    holdout.to_csv(out / "model_results.csv")
    res["cv"].to_csv(out / "model_cv.csv")
    res["by_group"].to_csv(out / "model_by_group.csv", index=False)

    _, test = split_by_graph(df, cfg["model"]["test_fraction"], cfg["seed"])
    T = time_matrix(df, candidates)[test]
    plot_net_time(holdout, out / "fig_net_time.png")
    plot_regret_cdf(res["_choices"], T, out / "fig_regret_cdf.png")
    plot_confusion(res["_choices"]["random_forest"].to_numpy(), res["_test_labels"].to_numpy(),
                   candidates, out / "fig_confusion_rf.png")

    show = ["accuracy", "accuracy_tie_aware", "regret_mean", "regret_p95", "search_s",
            "feature_s", "inference_s", "net_s", "saved_vs_best_fixed", "saved_vs_run_all",
            "net_amortised_s", "saved_vs_best_fixed_amortised"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"features: tiers {cfg['model']['feature_tiers']}, label: {cfg['model']['label']}, "
              f"best fixed algorithm (train): {holdout.attrs['best_fixed']}")
        print(f"test: {len(test)} queries on {df.iloc[test]['graph_id'].nunique()} graphs\n")
        print(holdout[show].round(4).to_string(), "\n")
        print("Group CV on training graphs:")
        print(res["cv"].round(4).to_string(), "\n")
        g = res["by_group"]
        print("Selectors vs best fixed vs oracle by weight type (test, no inference time):")
        print(g[g.group_by == "weights"].pivot(index="group", columns="method",
              values=["accuracy", "regret_mean", "net_s_no_inference"]).round(4).to_string())
    for tiers in cfg["model"]["comparison_tiers"]:
        name = "_".join(map(str, tiers))
        comp = evaluate(df, cfg, tiers)["holdout"]
        comp.to_csv(out / f"model_results_tiers_{name}.csv")
        print(f"\nComparison, tiers {tiers}:")
        print(comp[["accuracy", "regret_mean", "feature_s", "inference_s", "net_s",
                    "saved_vs_best_fixed", "saved_vs_best_fixed_amortised"]].round(4).to_string())
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
