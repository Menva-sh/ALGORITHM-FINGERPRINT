"""Stage 8 (contribution 3): calibration, conformal prediction sets and abstention.

Splits (all by graph): test graphs as in src.evaluate; the training graphs are split
again into a proper-training part and a conformal-calibration part.

  - Calibration: ECE of the top-choice confidence, raw vs isotonic-calibrated (group CV).
  - Conformal: for each target coverage, a threshold on 1 - p(true best algorithm) from the
    calibration graphs gives a SET of plausible algorithms per test query.
  - Abstention: a one-algorithm set is run directly; a larger set ("not sure") either runs
    the safe default (best fixed algorithm) or the top-2 interleaved on one core. Regret and
    net time (features + inference + search) are compared with no abstention.

Note: conformal coverage is guaranteed for exchangeable rows; queries of one graph are
correlated, so the guarantee is only approximate here and the EMPIRICAL coverage is reported.

Run:  python -m src.experiments.uncertainty  # results/uncertainty_*.csv, fig_*.png
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import (GRID, MUTED, TEXT, _style, best_fixed_algorithm,  # noqa: E402
                          split_by_graph, time_matrix)
from src.experiments.tiers import SERIES_COLORS  # noqa: E402
from src.models import (abstaining_cost, applicability_mask, build_models,  # noqa: E402
                        calibrated, class_scores, conformal_sets, conformal_threshold,
                        expected_calibration_error, feature_columns, feature_cost_ns,
                        inference_ns, masked_probs)

POLICIES = ["none", "fallback", "top2"]


def split_train_calibration(df: pd.DataFrame, train: np.ndarray, fraction: float,
                            seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into proper-train and calibration rows, by graph."""
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    proper, cal = next(gss.split(train, groups=df["graph_id"].to_numpy()[train]))
    return train[proper], train[cal]


def reliability(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Per confidence bin: mean confidence, accuracy, count."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = [{"bin": b, "confidence": confidence[bins == b].mean(),
             "accuracy": correct[bins == b].mean(), "count": int((bins == b).sum())}
            for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    m, exp = cfg["model"], cfg["experiments"]["uncertainty"]
    C, tiers, seed = cfg["algorithms"]["candidates"], exp["tiers"], cfg["seed"]
    X = df[feature_columns(df, cfg, tiers)].to_numpy(dtype=float)
    y = df[m["label"]].to_numpy()
    label_idx = np.array([C.index(a) for a in y])
    T, mask = time_matrix(df, C), applicability_mask(df, C)
    feat = feature_cost_ns(df, tiers)
    groups = df["graph_id"].to_numpy()
    train, test = split_by_graph(df, m["test_fraction"], seed)
    proper, cal = split_train_calibration(df, train, m["calibration_fraction"], seed)
    fixed = best_fixed_algorithm(T[train], C)
    Tt, mt, lt = T[test], mask[test], label_idx[test]
    rows_t = np.arange(len(test))
    best = np.nanmin(Tt, axis=1)
    ref_net = Tt[:, fixed].sum()

    calib, curves, abst = [], [], []
    for name in exp["models"]:
        model = build_models(cfg)[name].fit(X[proper], y[proper])
        infer = inference_ns(model, X[test], mt, C, m["inference_timing_samples"], seed)
        cal_model = calibrated(build_models(cfg)[name], X[proper], y[proper], groups[proper], cfg)
        for variant, est in [("raw", model), ("calibrated", cal_model)]:
            P = masked_probs(class_scores(est, X[test], C), mt)
            conf, correct = P.max(axis=1), P.argmax(axis=1) == lt
            calib.append({"model": name, "variant": variant, "accuracy": correct.mean(),
                          "mean_confidence": conf.mean(),
                          "ece": expected_calibration_error(conf, correct, m["ece_bins"])})
            curves.append(reliability(conf, correct, m["ece_bins"]).assign(model=name,
                                                                            variant=variant))
        # conformal sets from the RAW model (conformal needs no calibrated probabilities)
        P_cal = masked_probs(class_scores(model, X[cal], C), mask[cal])
        P = masked_probs(class_scores(model, X[test], C), mt)
        for cov in m["conformal_coverage"]:
            thr = conformal_threshold(P_cal, label_idx[cal], cov)
            sets = conformal_sets(P, mt, thr)
            size = sets.sum(axis=1)
            top_ok = P.argmax(axis=1) == lt
            sure = size == 1
            for policy in POLICIES:
                cost = abstaining_cost(sets, P, Tt, fixed, policy)
                regret = cost / best - 1
                net = feat[test].sum() + infer * len(test) + cost.sum()
                abst.append({
                    "model": name, "target_coverage": cov, "threshold": thr,
                    "empirical_coverage": sets[rows_t, lt].mean(),
                    "mean_set_size": size.mean(), "share_sure": sure.mean(),
                    "top1_accuracy_sure": top_ok[sure].mean() if sure.any() else np.nan,
                    "top1_accuracy_unsure": top_ok[~sure].mean() if (~sure).any() else np.nan,
                    "policy": policy, "regret_mean": regret.mean(),
                    "regret_p95": np.quantile(regret, 0.95),
                    "net_rel_best_fixed": net / ref_net,
                })
    return {"calibration": pd.DataFrame(calib), "reliability": pd.concat(curves),
            "abstention": pd.DataFrame(abst)}


# ------------------------------------------------------------------ figures

def plot_reliability(curves: pd.DataFrame, path) -> None:
    """Reliability diagram per model: raw vs calibrated."""
    models = list(dict.fromkeys(curves["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1,
                label="perfect calibration")
        for color, variant in zip(SERIES_COLORS, ["raw", "calibrated"]):
            d = curves[(curves["model"] == model) & (curves["variant"] == variant)]
            ax.plot(d["confidence"], d["accuracy"], color=color, linewidth=2, marker="o",
                    markersize=5, label=variant)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence of top choice", color=MUTED)
        ax.set_ylabel("accuracy", color=MUTED)
        ax.set_title(f"Reliability: {model}", loc="left", fontsize=11, color=TEXT)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_abstention(abst: pd.DataFrame, model: str, path) -> None:
    """Regret and net time vs target coverage per policy (no abstention dashed)."""
    d = abst[abst["model"] == model]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("share_sure", "Share of queries with a 1-algorithm set", None),
              ("regret_mean", "Mean regret (lower is better)", None),
              ("net_rel_best_fixed", "Net time / best fixed (lower is better)", 1.0)]
    for ax, (metric, title, ref) in zip(axes, panels):
        if metric == "share_sure":
            g = d[d["policy"] == "none"].sort_values("target_coverage")
            ax.plot(g["target_coverage"], g[metric], color=SERIES_COLORS[0], linewidth=2,
                    marker="o")
        else:
            none = d[d["policy"] == "none"][metric].iloc[0]
            ax.axhline(none, color=MUTED, linestyle="--", linewidth=1, label="no abstention")
            for color, policy in zip(SERIES_COLORS[1:], ["fallback", "top2"]):
                g = d[d["policy"] == policy].sort_values("target_coverage")
                label = "unsure -> best fixed" if policy == "fallback" else "unsure -> top-2"
                ax.plot(g["target_coverage"], g[metric], color=color, linewidth=2,
                        marker="o", label=label)
        if ref is not None:
            ax.axhline(ref, color=GRID, linewidth=1.5)
        ax.set_xlabel("target coverage of conformal sets", color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xticks(sorted(d["target_coverage"].unique()))
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Selector with abstention ({model}, test graphs)", x=0.01, ha="left",
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    out = PROJECT_ROOT / cfg["paths"]["results"]
    res = run(df, cfg)
    res["calibration"].to_csv(out / "uncertainty_calibration.csv", index=False)
    res["reliability"].to_csv(out / "uncertainty_reliability.csv", index=False)
    res["abstention"].to_csv(out / "uncertainty_abstention.csv", index=False)
    primary = cfg["experiments"]["uncertainty"]["models"][0]
    plot_reliability(res["reliability"], out / "fig_reliability.png")
    plot_abstention(res["abstention"], primary, out / "fig_abstention.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("Calibration (test graphs):")
        print(res["calibration"].round(4).to_string(index=False), "\n")
        print("Conformal sets and abstention (test graphs):")
        print(res["abstention"].round(4).to_string(index=False))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
