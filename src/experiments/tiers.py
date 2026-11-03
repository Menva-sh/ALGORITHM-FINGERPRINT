"""Stage 6 (contribution 1): selection quality vs feature cost, per feature-tier set.

For every tier set in experiments.tiers.tier_sets the selectors are trained on the
training graphs and evaluated on the same held-out test graphs (src.evaluate). Reported:
  - quality (accuracy, regret) against the mean feature cost per query, with the
    Pareto frontier over (cost, regret) and (cost, accuracy)
  - net time relative to the best fixed algorithm, features charged per query and
    amortised over each graph's queries
  - the same per graph size: where do expensive features stop paying off?

Run:  python -m src.experiments.tiers   # writes results/tiers_*.csv and fig_tiers_*.png
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import GRID, MUTED, TEXT, _style, evaluate, time_matrix  # noqa: E402
from src.models import feature_cost_ns  # noqa: E402

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4


def tier_name(tiers: list[int]) -> str:
    """[0, 1, 2] -> 'T0+1+2'."""
    return "T" + "+".join(map(str, tiers))


def pareto_mask(cost: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """True for points not dominated by another (lower-or-equal cost AND loss, one strict)."""
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        dominated = (cost <= cost[i]) & (loss <= loss[i]) & ((cost < cost[i]) | (loss < loss[i]))
        keep[i] = not dominated.any()
    return keep


def run(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per tier set x model, breakdown per tier set x model x graph size)."""
    exp = cfg["experiments"]["tiers"]
    candidates = cfg["algorithms"]["candidates"]
    summary, by_size = [], []
    for tiers in exp["tier_sets"]:
        res = evaluate(df, cfg, tiers, model_names=exp["models"], cv=False)
        hold, test = res["holdout"], res["_test_index"]
        tdf = df.iloc[test].reset_index(drop=True)
        T = time_matrix(tdf, candidates)
        feat = feature_cost_ns(tdf, tiers)
        feat_am = feat / tdf.groupby("graph_id")["graph_id"].transform("size").to_numpy()
        fixed = T[:, candidates.index(hold.attrs["best_fixed"])]
        rows = np.arange(len(tdf))
        for model in exp["models"]:
            h = hold.loc[model]
            infer = h["inference_s"] * 1e9 / len(tdf)
            summary.append({
                "tier_set": tier_name(tiers), "model": model,
                "feature_cost_ms": feat.mean() / 1e6, "inference_us": infer / 1e3,
                "overhead_ms": (feat.mean() + infer) / 1e6,
                "accuracy": h["accuracy"], "regret_mean": h["regret_mean"],
                "regret_p95": h["regret_p95"], "net_s": h["net_s"],
                "net_rel_best_fixed": h["net_s"] / hold.loc["best_fixed", "net_s"],
                "saved_vs_best_fixed": h["saved_vs_best_fixed"],
                "saved_vs_best_fixed_amortised": h["saved_vs_best_fixed_amortised"],
            })
            chosen = T[rows, res["_choices"][model].to_numpy()]
            for n, idx in tdf.groupby("n_target").groups.items():
                idx = np.asarray(idx)
                ref = fixed[idx].sum()
                by_size.append({
                    "tier_set": tier_name(tiers), "model": model, "n_target": n,
                    "queries": len(idx), "feature_cost_ms": feat[idx].mean() / 1e6,
                    "search_rel_best_fixed": chosen[idx].sum() / ref,
                    "net_rel_best_fixed": (feat[idx] + infer + chosen[idx]).sum() / ref,
                    "net_rel_best_fixed_amortised": (feat_am[idx] + infer + chosen[idx]).sum() / ref,
                })
    summary_df = pd.DataFrame(summary)
    # Frontier over the full selection overhead (features + inference): a model with cheap
    # features but slow inference is not cheap to use.
    cost = summary_df["overhead_ms"].to_numpy()
    summary_df["pareto_regret"] = pareto_mask(cost, summary_df["regret_mean"].to_numpy())
    summary_df["pareto_accuracy"] = pareto_mask(cost, -summary_df["accuracy"].to_numpy())
    size_df = pd.DataFrame(by_size)
    size_df["pays_off"] = size_df["net_rel_best_fixed"] < 1.0
    size_df["pays_off_amortised"] = size_df["net_rel_best_fixed_amortised"] < 1.0
    return summary_df, size_df


def payoff_report(size_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per size: net time vs best fixed for each tier set, and the cheapest-net tier set."""
    d = size_df[size_df["model"] == model]
    out = {}
    for col, label in [("net_rel_best_fixed", "per_query"),
                       ("net_rel_best_fixed_amortised", "amortised")]:
        p = d.pivot(index="n_target", columns="tier_set", values=col)
        out[(label, "best_tier_set")] = p.idxmin(axis=1)
        for t in p.columns:
            out[(label, t)] = p[t]
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures

def plot_pareto(summary: pd.DataFrame, path) -> None:
    """Regret and accuracy vs mean selection overhead (features + inference) per query.

    Pareto frontier dashed; two panels, one y-axis each.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, flag, title in [
            (axes[0], "regret_mean", "pareto_regret", "Mean regret (lower is better)"),
            (axes[1], "accuracy", "pareto_accuracy", "Accuracy (higher is better)")]:
        front = summary[summary[flag]].sort_values("overhead_ms")
        ax.plot(front["overhead_ms"], front[metric], color=MUTED, linestyle="--",
                linewidth=1, label="Pareto frontier", zorder=1)
        groups = summary.groupby("model", sort=False)
        for k, (color, (model, d)) in enumerate(zip(SERIES_COLORS, groups)):
            ax.scatter(d["overhead_ms"], d[metric], s=60, color=color, label=model,
                       edgecolor="white", linewidth=2, zorder=3)
            offset = (6, 6) if k == 0 else (6, -13)  # first model labelled above, second below
            for _, r in d.iterrows():
                ax.annotate(r["tier_set"], (r["overhead_ms"], r[metric]),
                            textcoords="offset points", xytext=offset, fontsize=8,
                            color=TEXT)
        ax.set_xscale("log")
        ax.set_xlabel("mean selection overhead per query: features + inference (ms, log)",
                      color=MUTED)
        ax.margins(x=0.08, y=0.12)  # room for point labels
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_net(summary: pd.DataFrame, model: str, path) -> None:
    """Net time relative to the best fixed algorithm per tier set (log scale)."""
    d = summary[summary["model"] == model].reset_index(drop=True)
    ref = d["net_s"] / d["net_rel_best_fixed"]  # best fixed net time (same for all rows)
    per_query = d["net_rel_best_fixed"].to_numpy()
    amortised = (1 - d["saved_vs_best_fixed_amortised"]).to_numpy()
    assert np.allclose(ref, ref.iloc[0])
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for offset, vals, color, label in [(-0.18, per_query, SERIES_COLORS[0], "features per query"),
                                       (0.18, amortised, SERIES_COLORS[1],
                                        "features amortised over graph's queries")]:
        ax.barh(y + offset, vals, height=0.34, color=color, label=label)
        for yi, v in zip(y + offset, vals):
            ax.text(v * 1.08, yi, f"{v:.2f}x", va="center", fontsize=8, color=TEXT,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, d["tier_set"])
    ax.invert_yaxis()
    ax.set_xlabel("net time relative to best fixed algorithm (log scale, < 1 saves time)",
                  color=MUTED)
    ax.set_title(f"Net time per feature tier set ({model})", loc="left", fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_size(size_df: pd.DataFrame, model: str, path) -> None:
    """Net time vs best fixed per graph size, one line per tier set; two panels."""
    d = size_df[size_df["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in [(axes[0], "net_rel_best_fixed", "Features charged per query"),
                           (axes[1], "net_rel_best_fixed_amortised",
                            "Features amortised over the graph's queries")]:
        for color, (ts, g) in zip(SERIES_COLORS, d.groupby("tier_set", sort=False)):
            g = g.sort_values("n_target")
            ax.plot(g["n_target"], g[col], color=color, linewidth=2, marker="o", markersize=5,
                    label=ts)
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("graph size (nodes, log scale)", color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_ylabel(f"net time / best fixed ({model}, log)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, title="tier set", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ CLI

def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    out = PROJECT_ROOT / cfg["paths"]["results"]
    summary, size_df = run(df, cfg)
    primary = cfg["experiments"]["tiers"]["models"][0]
    summary.to_csv(out / "tiers_summary.csv", index=False)
    size_df.to_csv(out / "tiers_by_size.csv", index=False)
    payoff = payoff_report(size_df, primary)
    payoff.to_csv(out / "tiers_payoff_by_size.csv")
    plot_pareto(summary, out / "fig_tiers_pareto.png")
    plot_net(summary, primary, out / "fig_tiers_net.png")
    plot_by_size(size_df, primary, out / "fig_tiers_by_size.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary.round(4).to_string(index=False), "\n")
        print(f"Net time / best fixed by graph size ({primary}; < 1 means the tier set pays off):")
        print(payoff.round(3).to_string(), "\n")
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
