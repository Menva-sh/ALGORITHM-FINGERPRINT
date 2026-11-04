"""Page 2: every benchmark graph on a 2D map, coloured by the algorithm that wins on it."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import ui
from src.demo_logic import NAMES, nearest_graph

st.markdown("<h1>The algorithm map</h1>", unsafe_allow_html=True)
st.markdown("<p class='lead'>Each dot is one benchmark graph, placed so that similar graphs sit "
            "close together. The colour shows which algorithm won most of its queries. Where "
            "colours mix, the choice is harder.</p>", unsafe_allow_html=True)

df = ui.algorithm_map()
if df is None:
    st.warning("Run `python -m src.experiments.algorithm_map` first.")
    st.stop()

families = list(ui.FAMILY_SHORT)
chosen = st.pills("Graph families", families, selection_mode="multi", default=families,
                  format_func=ui.FAMILY_SHORT.get) or []
d = df[df["family"].isin(chosen)].assign(algorithm=lambda x: x["winner"].map(NAMES),
                                         family_name=lambda x: x["family"].map(ui.FAMILY_SHORT))
order = [a for a in ui.C if a in set(df["winner"])]
order.sort(key=lambda a: -int((df["winner"] == a).sum()))

with st.container(key="card_map"):
    st.markdown("<div class='legend'>" + "".join(
        f"<span>{ui.dot(a, 13)}{ui.esc(NAMES[a])}</span>" for a in order) + "</div>",
        unsafe_allow_html=True)
    color = alt.Scale(domain=[NAMES[a] for a in ui.C], range=[ui.COLORS[a] for a in ui.C])
    points = alt.Chart(d).mark_circle(size=55, opacity=0.9).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        color=alt.Color("algorithm:N", scale=color, legend=None),
        tooltip=[alt.Tooltip("family_name:N", title="family"),
                 alt.Tooltip("n_target:Q", title="nodes", format=","), "weights",
                 alt.Tooltip("algorithm:N", title="fastest"),
                 alt.Tooltip("winner_share:Q", format=".0%", title="of its queries")])
    layers = [points]
    yours = st.session_state.get("your_graph")
    if yours is not None and not d.empty:
        names = [n for t in [0, 1] for n in ui.CFG["features"]["tiers"][t]]
        spot = df.iloc[[nearest_graph(yours, df, names)]].assign(label="Your graph")
        layers += [alt.Chart(spot).mark_point(size=500, shape="circle", filled=False,
                                              strokeWidth=3, color="#888888").encode(x="x:Q", y="y:Q"),
                   alt.Chart(spot).mark_text(dy=-26, fontSize=14, fontWeight="bold",
                                             color="#888888").encode(x="x:Q", y="y:Q", text="label:N")]
    st.altair_chart(alt.layer(*layers).properties(height=480, background="transparent")
                    .configure_view(strokeWidth=0).interactive(), use_container_width=True)
    note = (" The ring marks the benchmark graph most similar to the one you tried on "
            "“Try a graph”." if yours is not None else "")
    st.markdown(f"<div class='muted small'>{len(d):,} of {len(df):,} benchmark graphs shown, "
                f"placed by UMAP from their full fingerprint.{note}</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<h2>Who wins most often</h2>", unsafe_allow_html=True)
counts = d["winner"].value_counts()
st.markdown("<div class='wins'>" + "".join(
    f"<div class='stat' style='border-top:5px solid {ui.COLORS[a]}'>"
    f"<div class='muted'>{ui.esc(NAMES[a])}</div><div class='v'>{int(counts.get(a, 0)):,}</div>"
    f"<div class='muted small'>graphs</div></div>" for a in order) + "</div>",
    unsafe_allow_html=True)
