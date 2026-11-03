"""Label report: how often each algorithm wins, ties, and how noisy the timings are.

Check this BEFORE modelling: if one algorithm wins more than model.label_balance_max of the
queries, change the graph mix in config.yaml and regenerate.

Run:  python -m src.experiments.label_report
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config


def win_shares(df: pd.DataFrame, by: str | None, label: str = "best_time") -> pd.DataFrame:
    """Share of queries won by each algorithm, overall or per group."""
    if by is None:
        return df[label].value_counts(normalize=True).to_frame("share").T
    return pd.crosstab(df[by], df[label], normalize="index")


def noise_table(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Per algorithm: distribution of IQR/median across queries."""
    rows = []
    for a in candidates:
        col = df[f"iqr_rel_{a}"].dropna()
        if col.empty:
            continue
        rows.append({"algorithm": a, "queries": len(col), "median": col.median(),
                     "p95": col.quantile(0.95), "share_over_10pct": (col > 0.10).mean()})
    return pd.DataFrame(rows).set_index("algorithm")


def balance_check(df: pd.DataFrame, max_share: float) -> tuple[bool, str, float]:
    """(ok, top algorithm, its share of best_time labels)."""
    shares = df["best_time"].value_counts(normalize=True)
    return bool(shares.iloc[0] <= max_share), str(shares.index[0]), float(shares.iloc[0])


def build_report(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    candidates = cfg["algorithms"]["candidates"]
    non_tie = df[~df["tie"]]
    return {
        "overall": pd.concat([win_shares(df, None).rename(index={"share": "all queries"}),
                              win_shares(non_tie, None).rename(index={"share": "non-tie only"}),
                              win_shares(df, None, "best_ops").rename(index={"share": "by op count"})]),
        "by_family": win_shares(df, "family"),
        "by_size": win_shares(df, "n_target"),
        "by_weights": win_shares(df, "weights"),
        "ties": df.groupby("family")["tie"].mean().to_frame("tie_rate"),
        "time_vs_ops": pd.crosstab(df["best_time"], df["best_ops"]),
        "noise": noise_table(df, candidates),
    }


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    out: Path = PROJECT_ROOT / cfg["paths"]["results"]
    report = build_report(df, cfg)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"{len(df)} queries on {df['graph_id'].nunique()} graphs\n")
        for name, table in report.items():
            table.to_csv(out / f"labels_{name}.csv")
            print(f"== {name}")
            print(table.round(3).to_string(), "\n")
    agree = (df["best_time"] == df["best_ops"]).mean()
    print(f"best by time == best by op count: {agree:.1%}")
    print(f"tie rate overall: {df['tie'].mean():.1%}")
    ok, top, share = balance_check(df, cfg["model"]["label_balance_max"])
    verdict = "OK" if ok else "WARNING: fix the graph mix before modelling"
    print(f"class balance: {top} wins {share:.1%} "
          f"(limit {cfg['model']['label_balance_max']:.0%}) -> {verdict}")


if __name__ == "__main__":
    main()
