"""Streamlit demo: pick a graph and a query, see the fingerprint, the selector's choice
(with confidence and SHAP explanation), and the real timings of every algorithm.

Run:  streamlit run app/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.algorithms import ALGORITHMS  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.demo_logic import recommend, run_all, train_selector  # noqa: E402
from src.evaluate import ALGO_COLORS  # noqa: E402
from src.fingerprint import extract  # noqa: E402
from src.generators import (InstanceSpec, build_instance, load_instance,  # noqa: E402
                            sample_queries)
from src.graph import Graph  # noqa: E402
from src.real_graphs import parse_edge_list  # noqa: E402

st.set_page_config(page_title="Algorithm selector", layout="wide")
CFG = load_config()
C = CFG["algorithms"]["candidates"]
COLOR = alt.Scale(domain=C, range=[ALGO_COLORS[a] for a in C])


# ------------------------------------------------------------------ cached resources

@st.cache_resource(show_spinner="Training the selector on the benchmark dataset...")
def selector():
    df = pd.read_csv(PROJECT_ROOT / CFG["paths"]["dataset"])
    best_fixed = df[[f"time_{a}" for a in ["dijkstra", "bidirectional_dijkstra"]]].sum().idxmin()
    return train_selector(df, CFG), best_fixed.removeprefix("time_")


@st.cache_resource(show_spinner="Building graph...")
def generated_graph(family: str, n: int, params: tuple, weights: str, rep: int) -> Graph:
    return build_instance(InstanceSpec(family, n, params, weights, rep), CFG).graph


@st.cache_resource(show_spinner="Parsing edge list...")
def uploaded_graph(edges_text: str, coords_text: str | None) -> Graph:
    return parse_edge_list(edges_text, coords_text)[0]


@st.cache_resource(show_spinner="Loading road network...")
def road_graph():
    real = load_config(profile="real")
    raw = PROJECT_ROOT / real["paths"]["raw_instances"]
    if not (raw / "index.csv").exists():
        return None
    index = pd.read_csv(raw / "index.csv")
    road = index[index["graph_id"].str.contains("road")]
    if road.empty:
        return None
    g, queries, meta = load_instance(raw / f"{road['graph_id'].iloc[0]}.npz")
    return g, meta


# ------------------------------------------------------------------ shared widgets

def query_picker(g: Graph, key: str) -> tuple[int, int]:
    """Random (s, t) in one component; a button draws a new one."""
    if st.button("New random query", key=f"{key}_btn"):
        st.session_state[f"{key}_seed"] = st.session_state.get(f"{key}_seed", 0) + 1
    seed = st.session_state.get(f"{key}_seed", 0)
    pairs = sample_queries(g, 1, np.random.default_rng(seed))
    if not pairs:
        st.error("This graph has no connected pair of nodes.")
        st.stop()
    s, t = pairs[0]
    st.caption(f"Query: source **{s}** -> target **{t}**")
    return s, t


def selection_panel(g: Graph, s: int, t: int) -> None:
    """Fingerprint, selector choice + explanation, real timings and verdict."""
    model, best_fixed = selector()
    fp = extract(g, tiers=[0, 1, 2], cfg=CFG)
    rec = recommend(model, fp.features, g, CFG)
    runs = run_all(g, s, t, CFG)

    left, right = st.columns(2)
    with left:
        st.subheader("Fingerprint")
        rows = [{"tier": f"Tier-{tier}", "feature": n, "value": fp.features[n]}
                for tier in [0, 1, 2] for n in CFG["features"]["tiers"][tier]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=300)
        st.caption("Cost per tier: " + ", ".join(
            f"Tier-{k} {v / 1e3:.1f} µs" for k, v in fp.tier_cost_ns.items())
            + ". The selector uses Tier-0 only (cheapest features that pay off).")
    with right:
        st.subheader("Selector's choice")
        st.markdown(f"Chosen algorithm: **`{rec['choice']}`** &nbsp; · &nbsp; "
                    f"confidence **{rec['confidence']:.0%}**")
        probs = pd.DataFrame({"algorithm": list(rec["probs"]), "probability": list(rec["probs"].values())})
        st.altair_chart(alt.Chart(probs).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("algorithm:N", sort="-x", title=None),
            color=alt.Color("algorithm:N", scale=COLOR, legend=None),
            tooltip=["algorithm", alt.Tooltip("probability:Q", format=".2f")]), use_container_width=True)
        shap_df = rec["shap"].rename("shap").reset_index().rename(columns={"index": "feature"})
        st.caption(f"Why {rec['choice']}? SHAP contribution of each feature (positive pushes "
                   f"towards {rec['choice']}):")
        st.altair_chart(alt.Chart(shap_df).mark_bar(color="#2a78d6").encode(
            x=alt.X("shap:Q", title="SHAP value"), y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")]), use_container_width=True)

    st.subheader("Real timings (all applicable algorithms)")
    fastest = runs.iloc[0]
    chosen_us = float(runs.set_index("algorithm").loc[rec["choice"], "median_us"])
    overhead_us = (fp.tier_cost_ns[0] + rec["inference_ns"]) / 1e3
    fixed_us = float(runs.set_index("algorithm").loc[best_fixed, "median_us"])
    v1, v2, v3 = st.columns(3)
    right_pick = rec["choice"] == fastest["algorithm"]
    v1.metric("Fastest", fastest["algorithm"], "selector was right" if right_pick
              else f"selector picked {rec['choice']}", delta_color="normal" if right_pick else "inverse")
    v2.metric("Regret", f"{chosen_us / fastest['median_us'] - 1:.1%}")
    v3.metric(f"Net time vs always {best_fixed}",
              f"{chosen_us + overhead_us:.0f} µs vs {fixed_us:.0f} µs",
              f"{1 - (chosen_us + overhead_us) / fixed_us:+.0%} saved")
    st.altair_chart(alt.Chart(runs).mark_bar().encode(
        x=alt.X("median_us:Q", title="median search time (µs)"),
        y=alt.Y("algorithm:N", sort="x", title=None),
        color=alt.Color("algorithm:N", scale=COLOR, legend=None),
        tooltip=["algorithm", alt.Tooltip("median_us:Q", format=".1f"), "nodes_popped",
                 "edges_relaxed"]), use_container_width=True)
    st.dataframe(runs.round(2), hide_index=True)
    st.caption(f"Each algorithm: {CFG['benchmark']['warmup_runs']} warm-up runs, median of "
               f"{CFG['benchmark']['repeats']} timed runs; all return path cost "
               f"{fastest['path_cost']:.4g}. Net time adds Tier-0 features "
               f"+ inference ({overhead_us:.1f} µs).")


# ------------------------------------------------------------------ pages

def page_try() -> None:
    st.title("Which shortest-path algorithm is fastest?")
    st.write("Pick a graph and a query. The selector predicts the fastest algorithm from a "
             "cheap fingerprint; then every algorithm is actually run so you can check it.")
    mode = st.radio("Graph", ["Generate", "Upload edge list"], horizontal=True)
    if mode == "Generate":
        fams = CFG["generation"]["families"]
        c = st.columns(5)
        family = c[0].selectbox("Family", list(fams))
        n = c[1].select_slider("Nodes", CFG["generation"]["sizes"], value=1000)
        params = tuple(sorted((k, c[2].selectbox(k, v, key=f"{family}_{k}"))
                              for k, v in fams[family]["params"].items()))
        weights = c[3].selectbox("Weights", fams[family]["weights"])
        rep = c[4].number_input("Seed (rep)", 0, 99, 0)
        g = generated_graph(family, n, params, weights, int(rep))
    else:
        edges = st.file_uploader("Edge list: one 'u v' or 'u v w' per line", type=["txt", "edges", "csv"])
        coords = st.file_uploader("Optional coordinates: 'id x y' per line", type=["txt", "coords"])
        if edges is None:
            st.info("Upload an edge list to continue.")
            return
        try:
            g = uploaded_graph(edges.getvalue().decode("utf-8"),
                               coords.getvalue().decode("utf-8") if coords else None)
        except ValueError as e:
            st.error(f"Could not read the file: {e}")
            return
    st.caption(f"{g.n} nodes, {g.m} edges, weighted={g.weighted}, "
               f"coordinates={g.has_coordinates}")
    s, t = query_picker(g, "try")
    selection_panel(g, s, t)


def page_map() -> None:
    st.title("Algorithm map")
    path = PROJECT_ROOT / CFG["paths"]["results"] / "algorithm_map.csv"
    if not path.exists():
        st.warning("Run `python -m src.experiments.algorithm_map` first.")
        return
    df = pd.read_csv(path)
    st.write("Each point is one benchmark graph, placed by UMAP from its full fingerprint "
             "(all tiers). Colour = the algorithm that was fastest on most of its queries.")
    fams = st.multiselect("Families", sorted(df["family"].unique()), sorted(df["family"].unique()))
    d = df[df["family"].isin(fams)]
    st.altair_chart(alt.Chart(d).mark_circle(size=40, opacity=0.85).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        color=alt.Color("winner:N", scale=COLOR, title="fastest"),
        tooltip=["family", "n_target", "weights", "winner",
                 alt.Tooltip("winner_share:Q", format=".0%", title="share of queries")]
    ).properties(height=560).interactive(), use_container_width=True)
    st.dataframe(d["winner"].value_counts().rename("graphs"))


def page_road() -> None:
    st.title("Case study: a real road network")
    loaded = road_graph()
    if loaded is None:
        st.warning("Road network not built: run `python -m src.real_graphs`.")
        return
    g, meta = loaded
    st.write(f"OpenStreetMap drive network of Piedmont, California: {g.n} intersections, "
             f"{g.m} road segments, weights = length in metres. Never used for training.")
    s, t = query_picker(g, "road")
    _, path, _ = ALGORITHMS["astar"](g, s, t)
    fig, ax = plt.subplots(figsize=(7, 6))
    xy = np.array(g.coords)
    for u, v, _ in g.edges():
        ax.plot(*xy[[u, v]].T, color="#c9c8c2", linewidth=0.8)
    if path:
        ax.plot(*xy[path].T, color="#e34948", linewidth=2.5)
    ax.scatter(*xy[[s, t]].T, color=["#2a78d6", "#1baf7a"], s=60, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig)
    st.caption("Grey: roads. Red: shortest path. Blue: source, green: target.")
    selection_panel(g, s, t)
    res = PROJECT_ROOT / CFG["paths"]["results"] / "shift_real_by_graph.csv"
    if res.exists():
        st.subheader("Over all 100 benchmark queries on this network (Stage 9)")
        r = pd.read_csv(res)
        st.dataframe(r[r["graph_id"].str.contains("road")].round(3), hide_index=True)


PAGES = {"Try a graph": page_try, "Algorithm map": page_map, "Road network": page_road}
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Selector: plain-Python decision tree on Tier-0 features, trained on "
                   "2,880 synthetic graphs.")
PAGES[choice]()
