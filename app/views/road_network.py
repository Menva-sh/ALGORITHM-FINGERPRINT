"""Page 3: case study on a real road network the model never saw."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app import ui
from src.algorithms import ALGORITHMS
from src.config import PROJECT_ROOT
from src.demo_logic import NAMES, reason

st.markdown("<h1>Case study: a real road network</h1>", unsafe_allow_html=True)
g = ui.road_graph()
if g is None:
    st.warning("Road network files missing in data/real_graphs.")
    st.stop()
st.markdown(f"<p class='lead'>An OpenStreetMap drive network of Piedmont, California: {g.n:,} "
            f"intersections and {g.m:,} road segments, with distances in metres. The model never "
            f"saw this network during training.</p>", unsafe_allow_html=True)

query = ui.random_query(g, "road")
s, t = query
race = st.session_state.get("road_race", 0)
r = ui.analyse("road", g, s, t, race)
rec, runs = r["rec"], r["runs"]
choice = rec["choice"]

left, right = st.columns([1.6, 1], gap="medium")
with left.container(key="card_road"):
    a, b = st.columns([2, 1], vertical_alignment="center")
    a.markdown(f"<span class='muted'>Query: intersection {s} → intersection {t}</span>",
               unsafe_allow_html=True)
    if b.button("New random query", type="primary", use_container_width=True):
        ui.new_query("road")
        st.rerun()
    _, path, _ = ALGORITHMS["astar"](g, s, t)
    xy = np.array(g.coords)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.patch.set_facecolor("#f1ece2")
    ax.set_facecolor("#f1ece2")
    for u, v, _ in g.edges():
        ax.plot(*xy[[u, v]].T, color="#c9bfae", linewidth=1.1, zorder=1)
    if path:
        ax.plot(*xy[path].T, color="#d9472b", linewidth=3.2, solid_capstyle="round", zorder=2)
    ax.scatter(*xy[[s]].T, color="#2e8b57", s=110, edgecolor="white", linewidth=2, zorder=3)
    ax.scatter(*xy[[t]].T, color="#2f6bb3", s=110, edgecolor="white", linewidth=2, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("<div class='legend muted'>"
                "<span><span style='width:22px;height:3px;background:#c9bfae;display:inline-block'></span>Roads</span>"
                "<span><span style='width:22px;height:4px;background:#d9472b;display:inline-block'></span>Shortest path</span>"
                f"<span>{ui.dot('astar', 12)}Start</span><span>{ui.dot('dijkstra', 12)}End</span>"
                "</div>", unsafe_allow_html=True)

with right:
    with st.container(key="rec_road"):
        st.markdown(f"<div class='eyebrow'>RECOMMENDED ALGORITHM</div>"
                    f"<div class='big-algo'>{ui.dot(choice)}{ui.esc(NAMES[choice])}</div>"
                    f"<div style='display:flex;justify-content:space-between;align-items:end'>"
                    f"<b>Confidence</b><span style='font-size:1.7rem;font-weight:700;color:#1f5e58'>"
                    f"{rec['confidence']:.0%}</span></div>{ui.bar(rec['confidence'], ui.TEAL)}"
                    f"<p style='margin-top:12px;color:#33403f'>{ui.esc(reason(choice, r['fp'].features))}</p>",
                    unsafe_allow_html=True)
    fastest = runs.iloc[0]["algorithm"]
    regret = r["chosen_us"] / runs.iloc[0]["median_us"] - 1
    verdict = ("Pick was right, no regret" if choice == fastest
               else f"{NAMES[fastest]} was faster ({regret:.0%} regret)")
    with st.container(key="card_verdict"):
        st.markdown(f"<b class='{'good' if choice == fastest else ''}' style='font-size:1.15rem'>"
                    f"{'✓ ' if choice == fastest else ''}{verdict}</b>"
                    + ui.race_rows(runs, ui.SHORT, label_px=130), unsafe_allow_html=True)
    net = r["chosen_us"] + r["overhead_us"]
    if net > r["fixed_us"]:
        st.markdown(f"<div class='callout warn'><b>Slower overall on this tiny query</b><br>"
                    f"With prediction included it took {net:,.0f} µs, against "
                    f"{r['fixed_us']:,.0f} µs for always using {NAMES[r['best_fixed']]}. The "
                    f"searches here take only about {r['chosen_us']:,.0f} µs, so the model's own "
                    f"time matters. This is expected on a network this small.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='callout good'><b>Faster overall</b><br>With prediction "
                    f"included it took {net:,.0f} µs, against {r['fixed_us']:,.0f} µs for always "
                    f"using {NAMES[r['best_fixed']]}.</div>", unsafe_allow_html=True)
    if st.button("Run the race again", use_container_width=True):
        st.session_state["road_race"] = race + 1
        st.rerun()

res = PROJECT_ROOT / ui.CFG["paths"]["results"] / "shift_real_by_graph.csv"
if res.exists():
    allr = pd.read_csv(res)
    road = allr[allr["graph_id"].str.contains("road")]
    main = road[(road["model"] == "plain_tree") & (road["feature_set"] == "T0")]
    if not main.empty:
        m = main.iloc[0]
        st.write("")
        st.markdown(f"<h2>Across {int(m['queries'])} random queries on this network</h2>"
                    "<div class='muted'>Benchmark results (Stage 9): quick features only, "
                    "plain-tree selector.</div>", unsafe_allow_html=True)
        c = st.columns(3)
        c[0].markdown(ui.stat("Picked the fastest algorithm", f"{m['accuracy']:.0%}",
                              "of queries", "teal"), unsafe_allow_html=True)
        c[1].markdown(ui.stat("Search time vs best fixed algorithm",
                              f"{m['search_rel_best_fixed']:.0%}", "of its time, search only",
                              "teal"), unsafe_allow_html=True)
        c[2].markdown(ui.stat("Best possible (perfect picks)", f"{m['oracle_rel_best_fixed']:.0%}",
                              "of the same baseline"), unsafe_allow_html=True)
        with st.expander("Compare feature sets and models"):
            st.dataframe(road[["feature_set", "model", "accuracy", "regret_mean",
                               "search_rel_best_fixed", "oracle_rel_best_fixed"]].round(3),
                         hide_index=True)
