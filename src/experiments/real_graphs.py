"""Stage 9: train on synthetic graphs, test on real graphs.

Selectors are trained on ALL main-dataset (synthetic) graphs and tested on every query of
the real-graph dataset (`--profile real`). Real graphs are never used for training. The
reference is the in-distribution result on synthetic test graphs (group split), so the
regret change shows the cost of the synthetic -> real shift. Results are also given per
real graph, because a handful of graphs is not a distribution.

Run:  python -m src.experiments.real_graphs   # writes results/shift_real*.csv
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluate import split_by_graph, time_matrix, train_test_evaluate
from src.experiments.size_shift import feature_set_name, score, with_probes


def per_graph(train_df: pd.DataFrame, test_df: pd.DataFrame, fs: dict[str, Any],
              cfg: dict[str, Any], model: str) -> pd.DataFrame:
    """Accuracy, regret and net time per real graph for one model."""
    probe = bool(fs.get("probe_k"))
    from src.experiments.probing import probe_feature_names
    res = train_test_evaluate(train_df, test_df, cfg, fs["tiers"], [model],
                              extra_features=probe_feature_names(cfg) if probe else None,
                              extra_cost_col="cost_probe_ns" if probe else None)
    C = cfg["algorithms"]["candidates"]
    T = time_matrix(test_df, C)
    choice = res["_choices"][model].to_numpy()
    fixed = C.index(res["holdout"].attrs["best_fixed"])
    rows_idx = np.arange(len(test_df))
    chosen, best = T[rows_idx, choice], np.nanmin(T, axis=1)
    out = []
    for gid, idx in test_df.reset_index(drop=True).groupby("graph_id").groups.items():
        idx = np.asarray(idx)
        r = test_df.iloc[idx[0]]
        out.append({"graph_id": gid, "feature_set": feature_set_name(fs), "model": model,
                    "n": int(r["f_n_nodes"]), "m": int(r["f_n_edges"]), "weights": r["weights"],
                    "queries": len(idx),
                    "chosen_most": pd.Series([C[i] for i in choice[idx]]).mode()[0],
                    "best_most": test_df.iloc[idx]["best_time"].mode()[0],
                    "accuracy": float((choice[idx] == [C.index(a) for a in
                                                       test_df.iloc[idx]["best_time"]]).mean()),
                    "regret_mean": float((chosen[idx] / best[idx] - 1).mean()),
                    "search_rel_best_fixed": float(chosen[idx].sum() / T[idx, fixed].sum()),
                    "oracle_rel_best_fixed": float(best[idx].sum() / T[idx, fixed].sum())})
    return pd.DataFrame(out)


def run(df: pd.DataFrame, df_real: pd.DataFrame, cfg: dict[str, Any],
        probes: pd.DataFrame | None = None,
        probes_real: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = cfg["experiments"]["size_shift"]  # same feature sets and models as size shift
    models = exp["models"]
    train, test = split_by_graph(df, cfg["model"]["test_fraction"], cfg["seed"])
    rows, graphs = [], []
    for fs in exp["feature_sets"]:
        if fs.get("probe_k") and (probes is None or probes_real is None):
            continue
        d = with_probes(df, probes, fs, cfg)
        r = with_probes(df_real, probes_real, fs, cfg)
        ref = score(d.iloc[train], d.iloc[test], fs, cfg, models)
        real = score(d, r, fs, cfg, models)
        for model in models:
            rows.append({
                "feature_set": feature_set_name(fs), "method": model,
                "accuracy_synthetic": ref.loc[model, "accuracy"],
                "accuracy_real": real.loc[model, "accuracy"],
                "regret_synthetic": ref.loc[model, "regret_mean"],
                "regret_real": real.loc[model, "regret_mean"],
                "regret_change": real.loc[model, "regret_mean"] - ref.loc[model, "regret_mean"],
                "net_rel_best_fixed_real": real.loc[model, "net_s"] / real.loc["best_fixed", "net_s"],
                "oracle_rel_best_fixed_real":
                    real.loc["oracle", "net_s"] / real.loc["best_fixed", "net_s"],
            })
            graphs.append(per_graph(d, r, fs, cfg, model))
    return pd.DataFrame(rows), pd.concat(graphs, ignore_index=True)


def main() -> None:
    cfg = load_config()
    real_cfg = load_config(profile="real")
    real_path = PROJECT_ROOT / real_cfg["paths"]["dataset"]
    if not real_path.exists():
        raise SystemExit("no real-graph dataset: run `python -m src.real_graphs` then "
                         "`python -m src.benchmark --profile real`")
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    df_real = pd.read_csv(real_path)
    p_main = PROJECT_ROOT / cfg["features"]["probe"]["out"]
    p_real = PROJECT_ROOT / real_cfg["features"]["probe"]["out"]
    summary, graphs = run(df, df_real, cfg, pd.read_csv(p_main) if p_main.exists() else None,
                          pd.read_csv(p_real) if p_real.exists() else None)
    out = PROJECT_ROOT / cfg["paths"]["results"]
    summary.to_csv(out / "shift_real.csv", index=False)
    graphs.to_csv(out / "shift_real_by_graph.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print(summary.round(4).to_string(index=False), "\n")
        print(graphs.round(4).to_string(index=False))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
