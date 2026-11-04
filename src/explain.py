"""Stage 8 (contribution 4): why does the selector choose what it chooses?

For each tier set in experiments.explain.tier_sets, a random forest is trained on the
training graphs and explained on the test graphs:
  - SHAP (TreeExplainer): mean |SHAP| per feature and algorithm
  - permutation importance with correlated features permuted TOGETHER (groups from
    hierarchical clustering of |Spearman| correlation), measured as the drop in accuracy
    and rise in regret of the masked selection
  - distillation: a shallow decision tree trained on the forest's predictions, its
    fidelity (agreement with the forest on test queries) and a plain-text rule list
  - textbook checks: common advice, how often it holds in the measured labels, and how
    often the distilled rules follow it

Run:  python -m src.explain   # results/explain_*.csv, rules_*.txt, fig_shap_*, fig_perm_*
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import (ALGO_COLORS, GRID, MUTED, TEXT, _style, split_by_graph,  # noqa: E402
                          time_matrix)
from src.experiments.tiers import tier_name  # noqa: E402
from src.models import applicability_mask, build_models, feature_columns, select  # noqa: E402

BINARY_FEATURES = {"is_weighted", "has_coordinates"}


# ------------------------------------------------------------------ feature groups

def correlated_groups(X: np.ndarray, threshold: float) -> list[list[int]]:
    """Feature index groups whose members have |Spearman rho| >= threshold (avg linkage)."""
    k = X.shape[1]
    if k == 1:
        return [[0]]
    rho = spearmanr(X).correlation
    rho = np.nan_to_num(np.atleast_2d(rho), nan=0.0)  # constant columns -> uncorrelated
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"),
                      t=1.0 - threshold, criterion="distance")
    return [sorted(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]


def grouped_permutation_importance(model: Any, X: np.ndarray, mask: np.ndarray,
                                   label_idx: np.ndarray, T: np.ndarray,
                                   groups: list[list[int]], names: list[str],
                                   candidates: list[str], repeats: int,
                                   seed: int) -> pd.DataFrame:
    """Accuracy drop and regret rise when a whole feature group is permuted jointly."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(X))
    best = np.nanmin(T, axis=1)

    def score(Xs: np.ndarray) -> tuple[float, float]:
        choice = select(model, Xs, mask, candidates)
        return float((choice == label_idx).mean()), float((T[rows, choice] / best - 1).mean())

    base_acc, base_reg = score(X)
    out = []
    for g in groups:
        drops, rises = [], []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, g] = X[rng.permutation(len(X))][:, g]
            acc, reg = score(Xp)
            drops.append(base_acc - acc)
            rises.append(reg - base_reg)
        out.append({"group": " + ".join(names[i] for i in g), "size": len(g),
                    "accuracy_drop": np.mean(drops), "accuracy_drop_std": np.std(drops),
                    "regret_rise": np.mean(rises), "regret_rise_std": np.std(rises)})
    return pd.DataFrame(out).sort_values("accuracy_drop", ascending=False)


# ------------------------------------------------------------------ SHAP

def shap_importance(rf: RandomForestClassifier, X: np.ndarray,
                    names: list[str]) -> pd.DataFrame:
    """Mean |SHAP value| per feature (rows) and class (columns)."""
    import shap

    vals = shap.TreeExplainer(rf).shap_values(X)
    arr = np.stack(vals, axis=-1) if isinstance(vals, list) else np.asarray(vals)
    if arr.ndim == 2:  # single-output edge case
        arr = arr[:, :, None]
    return pd.DataFrame(np.abs(arr).mean(axis=0), index=names, columns=list(rf.classes_))


# ------------------------------------------------------------------ distillation

def distill(rf: Any, X_train: np.ndarray, depth: int, seed: int) -> DecisionTreeClassifier:
    """Shallow tree trained to imitate the forest's predictions on the training rows."""
    return DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(
        X_train, rf.predict(X_train))


def _condition(name: str, threshold: float, left: bool) -> str:
    if name in BINARY_FEATURES and 0 < threshold < 1:
        return f"{name} = {0 if left else 1}"
    return f"{name} {'<=' if left else '>'} {threshold:.4g}"


def rule_list(tree: DecisionTreeClassifier, names: list[str]) -> list[str]:
    """One plain-text rule per leaf: IF conditions THEN algorithm [support, purity]."""
    t = tree.tree_
    rules: list[str] = []

    def walk(node: int, conds: list[str]) -> None:
        if t.children_left[node] == -1:
            dist = t.value[node][0] / t.value[node][0].sum()
            cls = tree.classes_[int(dist.argmax())]
            cond = " AND ".join(conds) if conds else "always"
            rules.append(f"IF {cond} THEN {cls}   "
                         f"[{t.n_node_samples[node]} training queries, "
                         f"{dist.max():.0%} of them labelled {cls} by the forest]")
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [_condition(f, thr, True)])
        walk(t.children_right[node], conds + [_condition(f, thr, False)])

    walk(0, [])
    return rules


# ------------------------------------------------------------------ textbook checks

def textbook_checks(test_df: pd.DataFrame, chosen: list[str]) -> pd.DataFrame:
    """Common advice vs measured labels and vs the distilled rules' choices."""
    d = test_df.assign(chosen=chosen)
    bfs = {"bfs", "bidirectional_bfs"}
    bidir = {"bidirectional_dijkstra", "bidirectional_bfs"}
    checks = [
        ("Unweighted graphs: a BFS variant is fastest",
         d["weights"] == "unit", bfs),
        ("Coordinates + distance weights: A* is fastest",
         d["weights"] == "euclidean", {"astar"}),
        ("Weighted, no coordinates: bidirectional Dijkstra is fastest",
         d["weights"].isin(["uniform", "heavy_tailed"]), {"bidirectional_dijkstra"}),
        ("Large graphs (>= 2500 nodes): a bidirectional search is fastest",
         d["n_target"] >= 2500, bidir),
        ("Small graphs (100 nodes): a bidirectional search is fastest",
         d["n_target"] == 100, bidir),
    ]
    rows = []
    for text, where, algos in checks:
        sub = d[where]
        rows.append({"advice": text, "queries": len(sub),
                     "holds_in_labels": sub["best_time"].isin(algos).mean(),
                     "rules_follow": sub["chosen"].isin(algos).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ experiment

def explain_tier_set(df: pd.DataFrame, cfg: dict[str, Any],
                     tiers: list[int]) -> dict[str, Any]:
    exp, C, seed = cfg["experiments"]["explain"], cfg["algorithms"]["candidates"], cfg["seed"]
    cols = feature_columns(df, cfg, tiers)
    names = [c[2:] for c in cols]
    X = df[cols].to_numpy(dtype=float)
    y = df[cfg["model"]["label"]].to_numpy()
    label_idx = np.array([C.index(a) for a in y])
    T, mask = time_matrix(df, C), applicability_mask(df, C)
    train, test = split_by_graph(df, cfg["model"]["test_fraction"], seed)
    rf = build_models(cfg)["random_forest"].fit(X[train], y[train])

    rng = np.random.default_rng(seed)
    sample = rng.choice(test, size=min(exp["shap_samples"], len(test)), replace=False)
    shap_df = shap_importance(rf, X[sample], names)

    groups = correlated_groups(X[train], exp["corr_threshold"])
    perm = grouped_permutation_importance(rf, X[test], mask[test], label_idx[test], T[test],
                                          groups, names, C, exp["perm_repeats"], seed)

    tree = distill(rf, X[train], exp["distill_depth"], seed)
    rf_choice = select(rf, X[test], mask[test], C)
    tree_choice = select(tree, X[test], mask[test], C)
    rows = np.arange(len(test))
    best = np.nanmin(T[test], axis=1)
    distill_row = {
        "tier_set": tier_name(tiers), "depth": exp["distill_depth"],
        "leaves": int(tree.get_n_leaves()),
        "fidelity": float((tree_choice == rf_choice).mean()),
        "forest_accuracy": float((rf_choice == label_idx[test]).mean()),
        "rules_accuracy": float((tree_choice == label_idx[test]).mean()),
        "forest_regret": float((T[test][rows, rf_choice] / best - 1).mean()),
        "rules_regret": float((T[test][rows, tree_choice] / best - 1).mean()),
    }
    checks = textbook_checks(df.iloc[test], [C[i] for i in tree_choice])
    return {"shap": shap_df, "permutation": perm, "distill": distill_row,
            "rules": rule_list(tree, names), "textbook": checks.assign(tier_set=tier_name(tiers))}


# ------------------------------------------------------------------ figures

def plot_shap(shap_df: pd.DataFrame, title: str, path, top: int = 15) -> None:
    """Stacked horizontal bars: mean |SHAP| per feature, split by algorithm."""
    d = shap_df.loc[shap_df.sum(axis=1).sort_values(ascending=True).index[-top:]]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(d) + 1.6))
    left = np.zeros(len(d))
    for algo in d.columns:
        ax.barh(d.index, d[algo], left=left, color=ALGO_COLORS.get(algo, MUTED), label=algo,
                edgecolor="white", linewidth=1)
        left += d[algo].to_numpy()
    ax.set_xlabel("mean |SHAP value| (summed over algorithms)", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation(perm: pd.DataFrame, title: str, path) -> None:
    """Accuracy drop per correlated feature group (mean +- std over repeats)."""
    d = perm.sort_values("accuracy_drop")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.6))
    ax.barh(d["group"], d["accuracy_drop"], xerr=d["accuracy_drop_std"], color=ALGO_COLORS["dijkstra"],
            height=0.6, error_kw={"ecolor": MUTED, "elinewidth": 1})
    for i, v in enumerate(d["accuracy_drop"]):
        ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel("accuracy drop when the group is permuted", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.tick_params(axis="y", labelsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    out = PROJECT_ROOT / cfg["paths"]["results"]
    distill_rows, checks = [], []
    for tiers in cfg["experiments"]["explain"]["tier_sets"]:
        name = tier_name(tiers)
        res = explain_tier_set(df, cfg, tiers)
        res["shap"].to_csv(out / f"explain_shap_{name}.csv")
        res["permutation"].to_csv(out / f"explain_permutation_{name}.csv", index=False)
        (out / f"rules_{name}.txt").write_text(
            f"Rules distilled from the random forest ({name} features)\n"
            f"fidelity to the forest on test queries: {res['distill']['fidelity']:.3f}\n\n"
            + "\n".join(res["rules"]) + "\n", encoding="utf-8")
        plot_shap(res["shap"], f"SHAP importance, random forest ({name})",
                  out / f"fig_shap_{name}.png")
        plot_permutation(res["permutation"], f"Grouped permutation importance ({name})",
                         out / f"fig_permutation_{name}.png")
        distill_rows.append(res["distill"])
        checks.append(res["textbook"])
        print(f"== {name}\nPermutation importance (correlated groups):")
        print(res["permutation"].round(4).to_string(index=False))
        print("\nDistilled rules:\n" + "\n".join(res["rules"]) + "\n")
    distill_df = pd.DataFrame(distill_rows)
    checks_df = pd.concat(checks, ignore_index=True)
    distill_df.to_csv(out / "explain_distillation.csv", index=False)
    checks_df.to_csv(out / "explain_textbook.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_colwidth", 70):
        print("Distillation (test graphs):")
        print(distill_df.round(4).to_string(index=False), "\n")
        print("Textbook advice vs measured labels vs distilled rules (test graphs):")
        print(checks_df.round(3).to_string(index=False))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
