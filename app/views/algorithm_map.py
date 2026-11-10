"""Page 2: every benchmark graph on a 2D map, coloured and shaped by the algorithm that wins on it."""
from __future__ import annotations

import altair as alt
import streamlit as st

from app import ui
from src.demo_logic import NAMES, nearest_graph

df = ui.algorithm_map()
top = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom")
with top:
    ui.header("Algorithm map", "Each point is a benchmark graph. Similar graphs sit close "
              "together, and colour and shape show the usual winner.")
if df is None:
    st.warning("Run `python -m src.experiments.algorithm_map` first.")
    st.stop()
png = ui.RESULTS / "fig_algorithm_map.png"
if png.exists():
    top.download_button("Export figure", png.read_bytes(), png.name, "image/png")

counts = df["winner"].value_counts()
order = list(counts.index)

families = list(ui.FAMILY_SHORT)
chosen_fams = st.pills("Graph family", families, selection_mode="multi", default=families,
                       format_func=ui.FAMILY_SHORT.get, key="map_fams") or []

yours = st.session_state.get("your_graph")
names = [n for t in [0, 1] for n in ui.CFG["features"]["tiers"][t]]
spot = df.iloc[[nearest_graph(yours, df, names)]] if yours is not None else None

map_col, side_col = st.columns([1.75, 1], gap="small")
with map_col.container(key="panel_map"):
    head = ui.panel_head("map", "Figure 2", "Benchmark graphs placed by UMAP of their fingerprint")
    # Legend doubles as a filter; each pill gets its algorithm's glyph via CSS.
    glyph_css = "".join(
        f".st-key-map_algos button:nth-of-type({i + 1})::before{{content:'';width:11px;height:11px;"
        f"margin-right:8px;background:{ui.COLORS[a]};clip-path:{ui.CLIP[ui.GLYPH[a]]};"
        f"border-radius:{'50%' if a == 'dijkstra' else '0'};display:inline-block}}"
        for i, a in enumerate(order))
    ui.md(f"<style>{glyph_css}.st-key-map_algos button[data-selected='true']"
          "{background:#fff!important;color:var(--ink)!important;border-color:var(--line)!important}"
          ".st-key-map_algos button:not([data-selected='true']){opacity:.45}</style>")
    shown = st.pills("Algorithms", order, selection_mode="multi", default=order,
                     format_func=NAMES.get, key="map_algos", label_visibility="collapsed") or []
    d = df[df["family"].isin(chosen_fams) & df["winner"].isin(shown)].assign(
        algorithm=lambda x: x["winner"].map(NAMES),
        family_name=lambda x: x["family"].map(ui.FAMILY_SHORT))
    algos = [a for a in ui.C if a in set(df["winner"])]
    color = alt.Scale(domain=[NAMES[a] for a in algos], range=[ui.COLORS[a] for a in algos])
    shape = alt.Scale(domain=[NAMES[a] for a in algos], range=[ui.SHAPES[a] for a in algos])
    axis = dict(labels=False, ticks=False, domain=True, domainColor="#BCC3CE", gridColor="#E7EAEF",
                tickCount=4, titleColor="#5F6877", titleFontWeight="normal", titleFontSize=12)
    points = alt.Chart(d).mark_point(size=42, filled=True, opacity=0.9).encode(
        x=alt.X("x:Q", title="UMAP 1", axis=alt.Axis(**axis), scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="UMAP 2", axis=alt.Axis(**axis), scale=alt.Scale(zero=False)),
        color=alt.Color("algorithm:N", scale=color, legend=None),
        shape=alt.Shape("algorithm:N", scale=shape, legend=None),
        tooltip=[alt.Tooltip("algorithm:N", title="usual winner"),
                 alt.Tooltip("family_name:N", title="family"),
                 alt.Tooltip("n_target:Q", title="nodes", format=","), "weights",
                 alt.Tooltip("winner_share:Q", format=".0%", title="of its queries")])
    layers = [points]
    if spot is not None:
        spot = spot.assign(label="Your graph")
        layers += [alt.Chart(spot).mark_point(size=520, shape="circle", filled=False, strokeWidth=2,
                                              color="#171C26").encode(x="x:Q", y="y:Q"),
                   alt.Chart(spot).mark_text(dx=20, dy=-14, align="left", fontSize=12,
                                             fontWeight=500, color="#171C26")
                   .encode(x="x:Q", y="y:Q", text="label:N")]
    st.altair_chart(alt.layer(*layers).properties(height=430, background="#FFFFFF")
                    .configure_view(strokeWidth=0).configure(font="IBM Plex Sans").interactive(),
                    use_container_width=True)
    with head.popover("View as table"):
        st.dataframe(d[["graph_id", "family_name", "n_target", "weights", "algorithm",
                        "winner_share", "x", "y"]], hide_index=True)
    head.download_button("CSV", d.to_csv(index=False), "figure2_algorithm_map.csv", "text/csv",
                         type="tertiary")
    ring = (" The ring marks the benchmark graph most similar to the one on Try a graph."
            if spot is not None else " Try a graph first to see where yours sits.")
    ui.md(ui.caption("Figure 2.", f"Each point is one benchmark graph ({len(d):,} of {len(df):,} "
                     "shown), coloured and shaped by the algorithm that won most of its queries. "
                     f"Positions come from UMAP of the full fingerprint.{ring}"))

side_col.markdown(ui.panel(f"Winners across {len(df):,} graphs", ui.win_rows(counts)),
                  unsafe_allow_html=True)
near = (f"<p>The ring marks the benchmark graph most similar to the one on Try a graph. Its "
        f"usual winner is {ui.esc(NAMES[spot.iloc[0]['winner']])}.</p>" if spot is not None else
        "<p>Run a query on Try a graph and a ring will mark the most similar benchmark graph.</p>")
side_col.markdown("<div style='height:14px'></div>" + ui.panel("Reading the map", (
    "<div style='display:flex;flex-direction:column;gap:10px;font-size:13.5px;color:var(--ink2)'>"
    "<p style='margin:0'>Graphs with similar fingerprints sit close together. Where colours mix, "
    "the choice between algorithms is harder.</p>" + near.replace("<p>", "<p style='margin:0'>")
    + "<p style='margin:0'>The axes are arbitrary UMAP dimensions, so only distances between "
    "points carry meaning.</p></div>")), unsafe_allow_html=True)
