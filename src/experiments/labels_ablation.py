"""Stage 9: wall-clock labels vs operation-count labels (and a second machine, if given).

Every model is trained on the same training graphs twice: once with best_time labels and
once with best_ops labels (fewest nodes popped + edges relaxed). Both are judged on
WALL-CLOCK time on the same test graphs: accuracy against best_time, regret and net time.

If experiments.labels.second_machine_dataset points to a dataset.csv benchmarked on another
machine (same graphs and queries), models trained on THIS machine's labels are also tested
on the OTHER machine's timings, against a reference trained on the other machine itself.

Run:  python -m src.experiments.labels_ablation   # writes results/shift_labels.csv
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluate import split_by_graph, train_test_evaluate


def _rows(table: pd.DataFrame, scenario: str, models: list[str]) -> list[dict[str, Any]]:
    ref = table.loc["best_fixed", "net_s"]
    return [{"scenario": scenario, "method": m, "accuracy_vs_time": table.loc[m, "accuracy"],
             "regret_mean": table.loc[m, "regret_mean"],
             "regret_p95": table.loc[m, "regret_p95"],
             "net_rel_best_fixed": table.loc[m, "net_s"] / ref}
            for m in models + ["best_fixed", "oracle"]]


def run(df: pd.DataFrame, cfg: dict[str, Any],
        df_other: pd.DataFrame | None = None) -> pd.DataFrame:
    models, tiers = cfg["experiments"]["labels"]["models"], cfg["model"]["feature_tiers"]
    train, test = split_by_graph(df, cfg["model"]["test_fraction"], cfg["seed"])
    tr, te = df.iloc[train], df.iloc[test]
    rows = []
    for label, scenario in [("best_time", "time labels"), ("best_ops", "op-count labels")]:
        res = train_test_evaluate(tr, te, cfg, tiers, models, train_label=label,
                                  eval_label="best_time")
        rows += _rows(res["holdout"], scenario, models)
    if df_other is not None:
        te_other = df_other[df_other["graph_id"].isin(te["graph_id"])]
        tr_other = df_other[df_other["graph_id"].isin(tr["graph_id"])]
        for label, train_df, scenario in [
                ("best_time", tr_other, "other machine: own time labels (reference)"),
                ("best_time", tr, "other machine: this machine's time labels"),
                ("best_ops", tr, "other machine: this machine's op-count labels")]:
            res = train_test_evaluate(train_df, te_other, cfg, tiers, models,
                                      train_label=label, eval_label="best_time")
            rows += _rows(res["holdout"], scenario, models)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    other_path = cfg["experiments"]["labels"]["second_machine_dataset"]
    df_other = pd.read_csv(PROJECT_ROOT / other_path) if other_path else None
    if df_other is None:
        print("no second-machine dataset configured: hardware shift not measured\n")
    table = run(df, cfg, df_other)
    out = PROJECT_ROOT / cfg["paths"]["results"]
    table.to_csv(out / "shift_labels.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    print(f"\nwrote results to {out}")


if __name__ == "__main__":
    main()
