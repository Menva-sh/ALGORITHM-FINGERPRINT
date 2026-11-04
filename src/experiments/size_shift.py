"""Stage 9: train on small graphs, test on much larger ones.

Scenarios (each shifted scenario is compared with an in-distribution reference on the
SAME test graphs, so the regret change is attributable to the shift):
  within the main dataset
    large->large    group split inside graphs with n_target >= large_min (reference)
    small->large    train on graphs with n_target <= small_max, same test graphs
  large-graph profile (20k-50k nodes, `--profile large`), if data/dataset_large.csv exists
    xl->xl          group split inside the large profile (reference)
    main->xl        train on ALL main-dataset graphs (<= 5000 nodes), same test graphs

Feature sets come from experiments.size_shift.feature_sets (static tiers, optionally
with probe features of budget k; probe cost is charged per query).

Run:  python -m src.experiments.size_shift   # writes results/shift_size*.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluate import split_by_graph, train_test_evaluate
from src.experiments.probing import merge_probes, probe_feature_names
from src.experiments.tiers import tier_name


def feature_set_name(fs: dict[str, Any]) -> str:
    name = tier_name(fs["tiers"])
    return f"{name}+probe{fs['probe_k']}" if fs.get("probe_k") else name


def with_probes(df: pd.DataFrame, probes: pd.DataFrame | None, fs: dict[str, Any],
                cfg: dict[str, Any]) -> pd.DataFrame:
    """df with the probe columns of budget fs['probe_k'] (unchanged if no probe)."""
    if not fs.get("probe_k"):
        return df
    if probes is None:
        raise FileNotFoundError("probe features requested but no probe file")
    return merge_probes(df, probes, fs["probe_k"], cfg)


def score(train_df: pd.DataFrame, test_df: pd.DataFrame, fs: dict[str, Any],
          cfg: dict[str, Any], models: list[str]) -> pd.DataFrame:
    """Holdout table for one train/test pair and feature set."""
    probe = bool(fs.get("probe_k"))
    res = train_test_evaluate(train_df, test_df, cfg, fs["tiers"], model_names=models,
                              extra_features=probe_feature_names(cfg) if probe else None,
                              extra_cost_col="cost_probe_ns" if probe else None)
    return res["holdout"]


def _rows(table: pd.DataFrame, scenario: str, fs: dict[str, Any], models: list[str],
          test_df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for model in models + ["best_fixed", "oracle"]:
        h = table.loc[model]
        out.append({"scenario": scenario, "feature_set": feature_set_name(fs), "method": model,
                    "test_graphs": test_df["graph_id"].nunique(), "test_queries": len(test_df),
                    "test_n_min": int(test_df["n_target"].min()),
                    "test_n_max": int(test_df["n_target"].max()),
                    "accuracy": h["accuracy"], "regret_mean": h["regret_mean"],
                    "regret_p95": h["regret_p95"],
                    "net_rel_best_fixed": h["net_s"] / table.loc["best_fixed", "net_s"],
                    "net_rel_best_fixed_amortised":
                        h["net_amortised_s"] / table.loc["best_fixed", "net_s"]})
    return out


def pair(reference_df: pd.DataFrame, shifted_train: pd.DataFrame, names: tuple[str, str],
         fs: dict[str, Any], cfg: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    """Reference (group split inside reference_df) vs shifted training set, same test graphs."""
    train, test = split_by_graph(reference_df, cfg["model"]["test_fraction"], cfg["seed"])
    test_df = reference_df.iloc[test]
    rows = _rows(score(reference_df.iloc[train], test_df, fs, cfg, models), names[0], fs,
                 models, test_df)
    shifted_train = shifted_train[~shifted_train["graph_id"].isin(test_df["graph_id"])]
    rows += _rows(score(shifted_train, test_df, fs, cfg, models), names[1], fs, models, test_df)
    return rows


def run(df: pd.DataFrame, cfg: dict[str, Any], probes: pd.DataFrame | None = None,
        df_large: pd.DataFrame | None = None,
        probes_large: pd.DataFrame | None = None) -> pd.DataFrame:
    exp = cfg["experiments"]["size_shift"]
    models = exp["models"]
    rows: list[dict[str, Any]] = []
    for fs in exp["feature_sets"]:
        if fs.get("probe_k") and probes is None:
            continue
        d = with_probes(df, probes, fs, cfg)
        small = d[d["n_target"] <= exp["small_max"]]
        large = d[d["n_target"] >= exp["large_min"]]
        rows += pair(large, small, ("large->large", "small->large"), fs, cfg, models)
        if df_large is not None and (not fs.get("probe_k") or probes_large is not None):
            xl = with_probes(df_large, probes_large, fs, cfg)
            rows += pair(xl, d, ("xl->xl", "main->xl"), fs, cfg, models)
    return pd.DataFrame(rows)


def shift_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Regret / accuracy change of each shifted scenario vs its in-distribution reference."""
    refs = {"small->large": "large->large", "main->xl": "xl->xl"}
    out = []
    for shifted, ref in refs.items():
        s = table[table["scenario"] == shifted].set_index(["feature_set", "method"])
        r = table[table["scenario"] == ref].set_index(["feature_set", "method"])
        for key in s.index.intersection(r.index):
            if key[1] in ("best_fixed", "oracle"):
                continue
            out.append({"shift": shifted, "reference": ref,
                        "feature_set": key[0], "method": key[1],
                        "accuracy_reference": r.loc[key, "accuracy"],
                        "accuracy_shifted": s.loc[key, "accuracy"],
                        "regret_reference": r.loc[key, "regret_mean"],
                        "regret_shifted": s.loc[key, "regret_mean"],
                        "regret_change": s.loc[key, "regret_mean"] - r.loc[key, "regret_mean"],
                        "net_rel_best_fixed_shifted": s.loc[key, "net_rel_best_fixed"]})
    return pd.DataFrame(out)


def _maybe_read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    cfg = load_config()
    large_cfg = load_config(profile="large")
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    probes = _maybe_read(PROJECT_ROOT / cfg["features"]["probe"]["out"])
    df_large = _maybe_read(PROJECT_ROOT / large_cfg["paths"]["dataset"])
    probes_large = _maybe_read(PROJECT_ROOT / large_cfg["features"]["probe"]["out"])
    if df_large is None:
        print("no large-profile dataset: run generators/benchmark with --profile large")
    table = run(df, cfg, probes, df_large, probes_large)
    summary = shift_summary(table)
    out = PROJECT_ROOT / cfg["paths"]["results"]
    table.to_csv(out / "shift_size.csv", index=False)
    summary.to_csv(out / "shift_size_summary.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print(table.round(4).to_string(index=False), "\n")
        print(summary.round(4).to_string(index=False))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
