"""2D "algorithm map": UMAP of graph fingerprints, coloured by the winning algorithm.

One point per graph of the main dataset: its fingerprint (all tiers, log1p + standardised)
embedded in 2D with UMAP; colour = the algorithm that is fastest on most of its queries.

Run:  python -m src.experiments.algorithm_map   # results/algorithm_map.csv + figure
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import ALGO_COLORS, MUTED, TEXT  # noqa: E402
from src.models import feature_columns  # noqa: E402


def per_graph(df: pd.DataFrame, cfg: dict[str, Any], tiers: list[int]) -> pd.DataFrame:
    """One row per graph: fingerprint, metadata, majority winner and its share."""
    cols = feature_columns(df, cfg, tiers)
    g = df.groupby("graph_id")
    out = g[cols + ["family", "weights", "n_target"]].first()
    out["winner"] = g["best_time"].agg(lambda s: s.value_counts().index[0])
    out["winner_share"] = g["best_time"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return out.reset_index()


def embed(graphs: pd.DataFrame, cfg: dict[str, Any], tiers: list[int]) -> pd.DataFrame:
    """Add UMAP coordinates x, y (seeded)."""
    import umap

    m = cfg["experiments"]["algorithm_map"]
    cols = [c for c in graphs.columns if c.startswith("f_")]
    X = StandardScaler().fit_transform(np.log1p(graphs[cols].to_numpy(dtype=float)))
    xy = umap.UMAP(n_neighbors=m["n_neighbors"], min_dist=m["min_dist"],
                   random_state=cfg["seed"]).fit_transform(X)
    return graphs.assign(x=xy[:, 0], y=xy[:, 1])


def plot(graphs: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for algo, d in graphs.groupby("winner"):
        ax.scatter(d["x"], d["y"], s=14, color=ALGO_COLORS.get(algo, MUTED), label=algo,
                   edgecolor="white", linewidth=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Algorithm map: UMAP of graph fingerprints, coloured by winner", loc="left",
                 fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=9, markerscale=1.6, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    tiers = cfg["experiments"]["algorithm_map"]["tiers"]
    df = pd.read_csv(PROJECT_ROOT / cfg["paths"]["dataset"])
    graphs = embed(per_graph(df, cfg, tiers), cfg, tiers)
    out = PROJECT_ROOT / cfg["paths"]["results"]
    graphs.to_csv(out / "algorithm_map.csv", index=False)
    plot(graphs, out / "fig_algorithm_map.png")
    print(graphs["winner"].value_counts().to_string())
    print(f"wrote {len(graphs)} graphs to {out / 'algorithm_map.csv'}")


if __name__ == "__main__":
    main()
