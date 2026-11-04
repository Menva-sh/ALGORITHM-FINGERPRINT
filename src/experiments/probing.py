"""Stage 7 (contribution 2): static features vs static + probe features.

For each static tier set in experiments.probing.static_sets and each probe budget k,
the selectors are trained with and without the probe features of that k and evaluated
on the same held-out test graphs. Probe time is a per-query cost: it is charged in full
to every query (never amortised over the graph), on top of the static feature cost.

Reported: accuracy / regret / net time per variant, accuracy gained per millisecond of
probing, and where (per weight type) the probe changes the decisions.

Needs data/probes.csv from `python -m src.probe`.
Run:  python -m src.experiments.probing   # writes results/probing_*.csv, fig_probing_*.png
"""
from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import GRID, MUTED, TEXT, _style, evaluate  # noqa: E402
from src.experiments.tiers import SERIES_COLORS, tier_name  # noqa: E402


def probe_feature_names(cfg: dict[str, Any]) -> list[str]:
    return [f"f_{n}" for n in cfg["features"]["probe"]["names"]]


def merge_probes(df: pd.DataFrame, probes: pd.DataFrame, k: int,
                 cfg: dict[str, Any]) -> pd.DataFrame:
    """df with the probe features (as f_probe_*) and cost_probe_ns of budget k.

    Keeps df's row order (so the graph split is identical); every row must have a probe.
    """
    names = cfg["features"]["probe"]["names"]
    p = probes[probes["k"] == k][["graph_id", "query_idx", *names, "probe_ns"]]
    p = p.rename(columns={**{n: f"f_{n}" for n in names}, "probe_ns": "cost_probe_ns"})
    out = df.merge(p, on=["graph_id", "query_idx"], how="left", validate="one_to_one")
    if out["cost_probe_ns"].isna().any():
        raise ValueError(f"missing probe rows for k={k}")
    return out


def _rows(res: dict[str, Any], static: list[int], k: int, models: list[str],
          probe_ms: float) -> list[dict[str, Any]]:
    hold = res["holdout"]
    n = len(res["_test_index"])
    out = []
    for model in models:
        h = hold.loc[model]
        out.append({
            "static_set": tier_name(static), "k": k, "model": model,
            "probe_ms": probe_ms,
            "overhead_ms": (h["feature_s"] + h["inference_s"]) * 1e3 / n,
            "accuracy": h["accuracy"], "accuracy_tie_aware": h["accuracy_tie_aware"],
            "regret_mean": h["regret_mean"], "regret_p95": h["regret_p95"],
            "search_rel_best_fixed": h["search_s"] / hold.loc["best_fixed", "net_s"],
            "net_rel_best_fixed": h["net_s"] / hold.loc["best_fixed", "net_s"],
            "saved_vs_best_fixed": h["saved_vs_best_fixed"],
            "saved_vs_best_fixed_amortised": h["saved_vs_best_fixed_amortised"],
        })
    return out


def run(df: pd.DataFrame, probes: pd.DataFrame,
        cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per static set x k x model, accuracy per weight type)."""
    exp = cfg["experiments"]["probing"]
    models, extra = exp["models"], probe_feature_names(cfg)
    rows, groups = [], []
    for static in exp["static_sets"]:
        variants = [(0, df, None)] + [(k, merge_probes(df, probes, k, cfg), "cost_probe_ns")
                                      for k in cfg["features"]["probe"]["k_values"]]
        for k, dk, cost_col in variants:
            res = evaluate(dk, cfg, static, model_names=models, cv=False,
                           extra_features=extra if k else None, extra_cost_col=cost_col)
            test = res["_test_index"]
            probe_ms = dk.iloc[test]["cost_probe_ns"].mean() / 1e6 if k else 0.0
            rows += _rows(res, static, k, models, probe_ms)
            g = res["by_group"]
            g = g[g["group_by"] == "weights"].assign(static_set=tier_name(static), k=k)
            groups.append(g)
    summary = pd.DataFrame(rows)
    base = summary[summary["k"] == 0].set_index(["static_set", "model"])
    keys = list(zip(summary["static_set"], summary["model"]))
    summary["accuracy_gain"] = summary["accuracy"].to_numpy() - base.loc[keys, "accuracy"].to_numpy()
    summary["regret_drop"] = base.loc[keys, "regret_mean"].to_numpy() - summary["regret_mean"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["accuracy_gain_pp_per_ms"] = np.where(
            summary["k"] > 0, 100 * summary["accuracy_gain"] / summary["probe_ms"], np.nan)
    return summary, pd.concat(groups, ignore_index=True)


# ------------------------------------------------------------------ figures

def plot_tradeoff(summary: pd.DataFrame, model: str, path) -> None:
    """Accuracy and net time vs selection overhead; one line per static set through k."""
    d = summary[summary["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy (higher is better)"),
                              (axes[1], "net_rel_best_fixed",
                               "Net time / best fixed (lower is better)")]:
        for i, (color, (static, g)) in enumerate(zip(SERIES_COLORS,
                                                     d.groupby("static_set", sort=False))):
            g = g.sort_values("k")
            ax.plot(g["overhead_ms"], g[metric], color=color, linewidth=2, marker="o",
                    markersize=6, label=f"{static} static + probe")
            if metric == "net_rel_best_fixed" and i > 0:
                continue  # later sets cluster tightly on this panel; labels would collide
            for _, r in g.iterrows():
                ax.annotate("static" if r["k"] == 0 else f"k={int(r['k'])}",
                            (r["overhead_ms"], r[metric]), textcoords="offset points",
                            xytext=(6, 5), fontsize=8, color=TEXT)
        if metric == "net_rel_best_fixed":
            ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1fx"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks([0.5, 1, 2, 5])
        ax.set_xscale("log")
        ax.set_xlabel("selection overhead per query: features + probe + inference (ms, log)",
                      color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.margins(x=0.1, y=0.1)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Static vs static + probe features ({model}, test graphs)", x=0.01,
                 ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_weights(groups: pd.DataFrame, static: str, model: str, path) -> None:
    """Accuracy per weight type for each probe budget (k=0 is static only)."""
    d = groups[(groups["static_set"] == static) & (groups["method"] == model)]
    pivot = d.pivot(index="group", columns="k", values="accuracy")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(pivot.columns)
    y = np.arange(len(pivot))
    for i, (color, k) in enumerate(zip(SERIES_COLORS, pivot.columns)):
        vals = pivot[k].to_numpy()
        ax.barh(y + i * width - 0.4 + width / 2, vals, height=width * 0.9, color=color,
                label="static only" if k == 0 else f"+ probe k={k}")
        for yi, v in zip(y + i * width - 0.4 + width / 2, vals):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7, color=TEXT)
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("accuracy on test queries", color=MUTED)
    ax.set_title(f"Accuracy by weight type ({static}, {model})", loc="left", fontsize=11,
                 color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=len(pivot.columns))
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ CLI

def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    probes = pd.read_csv(PROJECT_ROOT / cfg["features"]["probe"]["out"])
    out = PROJECT_ROOT / cfg["paths"]["results"]
    summary, groups = run(df, probes, cfg)
    exp = cfg["experiments"]["probing"]
    primary, first_static = exp["models"][0], tier_name(exp["static_sets"][0])
    summary.to_csv(out / "probing_summary.csv", index=False)
    groups.to_csv(out / "probing_by_weights.csv", index=False)
    plot_tradeoff(summary, primary, out / "fig_probing_tradeoff.png")
    plot_by_weights(groups, first_static, primary, out / "fig_probing_by_weights.png")
    show = ["static_set", "k", "model", "probe_ms", "overhead_ms", "accuracy", "regret_mean",
            "accuracy_gain", "regret_drop", "accuracy_gain_pp_per_ms", "net_rel_best_fixed",
            "saved_vs_best_fixed_amortised"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summary[show].round(4).to_string(index=False), "\n")
        g = groups[(groups["static_set"] == first_static) & (groups["method"] == primary)]
        print(f"Accuracy by weight type ({first_static}, {primary}; k=0 is static only):")
        print(g.pivot(index="group", columns="k", values="accuracy").round(3).to_string())
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
