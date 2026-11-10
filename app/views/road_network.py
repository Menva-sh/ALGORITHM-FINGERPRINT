"""Page 3: case study on a real road network the model never saw."""
from __future__ import annotations

import numpy as np
import streamlit as st

from app import ui
from src.algorithms import ALGORITHMS
from src.demo_logic import NAMES, reason

g = ui.road_graph()
top = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom")
if g is None:
    with top:
        ui.header("Road network case study", "Road network files are missing in data/real_graphs.")
    st.stop()
with top:
    ui.header("Road network case study", f"Piedmont, California from OpenStreetMap: {g.n:,} "
              f"intersections and {g.m:,} road segments. Never used in training.")
if top.button("New random query", type="primary"):
    ui.new_query("road")

s, t = ui.random_query(g, "road")
r = ui.analyse("road", g, s, t)
rec, runs = r["rec"], r["runs"]
choice = rec["choice"]


def road_svg(path: list[int], w: int = 700, h: int = 400, pad: int = 18) -> str:
    """The road network as SVG, with the shortest path, start (filled) and end (open)."""
    xy = np.array(g.coords, dtype=float)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = min((w - 2 * pad) / (hi[0] - lo[0]), (h - 2 * pad) / (hi[1] - lo[1]))
    off = (np.array([w, h]) - (hi - lo) * scale) / 2
    p = np.column_stack([(xy[:, 0] - lo[0]) * scale + off[0],
                         h - ((xy[:, 1] - lo[1]) * scale + off[1])])  # north up
    roads = "".join(f"M{p[u, 0]:.1f} {p[u, 1]:.1f}L{p[v, 0]:.1f} {p[v, 1]:.1f}"
                    for u, v, _ in g.edges())
    line = " ".join(f"{p[i, 0]:.1f},{p[i, 1]:.1f}" for i in path)
    (sx, sy), (tx, ty) = p[s], p[t]
    label = "font-family=\"IBM Plex Sans, sans-serif\" font-size=\"12\" font-weight=\"600\" fill=\"#171C26\""
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' role='img' "
            f"aria-label='Road network of Piedmont with the shortest path from intersection {s} to "
            f"intersection {t}' style='width:100%;height:auto;display:block;background:#F6F7F9;"
            "border:1px solid #E7EAEF'>"
            f"<path d='{roads}' stroke='#B7BFCB' stroke-width='1.5' fill='none' stroke-linecap='round'/>"
            f"<polyline points='{line}' fill='none' stroke='#FFFFFF' stroke-width='9' "
            "stroke-linecap='round' stroke-linejoin='round'/>"
            f"<polyline points='{line}' fill='none' stroke='#171C26' stroke-width='4.5' "
            "stroke-linecap='round' stroke-linejoin='round'/>"
            f"<circle cx='{sx:.1f}' cy='{sy:.1f}' r='7' fill='#171C26' stroke='#FFFFFF' stroke-width='2.5'/>"
            f"<circle cx='{tx:.1f}' cy='{ty:.1f}' r='7' fill='#FFFFFF' stroke='#171C26' stroke-width='3'/>"
            f"<text x='{sx + 10:.1f}' y='{sy + 18:.1f}' {label}>{s}</text>"
            f"<text x='{tx + 10:.1f}' y='{ty - 10:.1f}' {label}>{t}</text></svg>")


fig_col, side_col = st.columns([1.5, 1], gap="small")
with fig_col.container(key="panel_road"):
    head = ui.panel_head("road", "Figure 3", f"Shortest path from intersection {s} to intersection {t}")
    _, path, _ = ALGORITHMS[choice](g, s, t)
    svg = road_svg(path)
    head.download_button("SVG", svg, f"figure3_road_{s}_{t}.svg", "image/svg+xml", type="tertiary")
    ui.md("<div class='legend'>"
          "<span><span style='width:22px;height:2px;background:#B7BFCB'></span>Roads</span>"
          "<span><span style='width:22px;height:5px;border-radius:3px;background:var(--ink)'></span>"
          "Shortest path</span><span><span style='width:12px;height:12px;border-radius:50%;"
          "background:var(--ink)'></span>Start</span><span><span style='width:12px;height:12px;"
          "border-radius:50%;border:2px solid var(--ink);background:#fff'></span>End</span></div>")
    ui.md(svg)
    ui.md(ui.caption("Figure 3.", f"Shortest path between intersection {s} (filled) and "
                     f"intersection {t} (open), {len(path) - 1} road segments long. Edge weights "
                     "are road length in metres."))

fastest = runs.iloc[0]["algorithm"]
regret = r["chosen_us"] / runs.iloc[0]["median_us"] - 1
side_col.markdown(ui.panel("Prediction", (
    "<div style='display:flex;align-items:center;justify-content:space-between;gap:12px'>"
    f"<div style='display:flex;align-items:center;gap:10px'>{ui.glyph(choice, 15)}<span class='big'>"
    f"{ui.esc(NAMES[choice])}</span></div><div style='display:flex;align-items:baseline;gap:8px'>"
    f"<span class='sub'>Confidence</span><span style='font-size:24px;line-height:1.1;"
    f"font-weight:600'>{rec['confidence']:.0%}</span></div></div>"
    f"<p style='font-size:13.5px;color:var(--ink2);margin:0'>{ui.esc(reason(choice, r['fp'].features))}</p>"
), "Made before any search runs"), unsafe_allow_html=True)

ok = choice == fastest
side_col.markdown("<div style='height:14px'></div>" + ui.panel("Measured search time", (
    ui.verdict(ok, "Correct, with no regret" if ok
               else f"{NAMES[fastest]} was faster ({regret:.0%} regret)")
    + ui.hbars([{"label": ui.algo_label(a), "value": v, "color": ui.COLORS[a]}
                for a, v in zip(runs["algorithm"], runs["median_us"])], label_w=140)),
    "Median, microseconds"), unsafe_allow_html=True)

net = r["chosen_us"] + r["overhead_us"]
slower = net > r["fixed_us"]
note = (f"<b style='color:var(--ink)'>{'Slower' if slower else 'Faster'} overall on this "
        f"{'small ' if slower else ''}query.</b> With prediction included it took "
        f"{ui.us(net)}, against {ui.us(r['fixed_us'])} for always using {NAMES[r['best_fixed']]}."
        + (f" Searches here take about {ui.us(r['chosen_us'])}, so the model's own time matters."
           if slower else ""))
side_col.markdown(
    "<div style='height:14px'></div><section class='panel'><div style='padding:12px 16px 14px;"
    "display:flex;gap:10px;align-items:flex-start'><svg width='16' height='16' viewBox='0 0 16 16' "
    "fill='none' stroke='#4B5565' stroke-width='1.5' stroke-linecap='round' style='flex:none;"
    "margin-top:2px' aria-hidden='true'><circle cx='8' cy='8' r='6.5'/><path d='M8 7.2V11M8 4.9v.1'/>"
    f"</svg><p style='font-size:13.5px;color:var(--ink2);margin:0'>{note}</p></div></section>",
    unsafe_allow_html=True)

allr = ui.result("shift_real_by_graph.csv")
if allr is not None:
    road = allr[allr["graph_id"].str.contains("road")]
    main = road[(road["model"] == "plain_tree") & (road["feature_set"] == "T0")]
    if not main.empty:
        m = main.iloc[0]
        st.write("")
        with st.container(key="panel_road_stats"):
            head = ui.panel_head("road_stats", f"Across {int(m['queries'])} random queries",
                                 "Quick features, decision tree")
            head.page_link("views/results.py", label="Full table in Results")
            ui.md(ui.stats([
                ("Picked the fastest algorithm", f"{m['accuracy']:.2f}", "Accuracy"),
                ("Mean regret", f"{m['regret_mean']:.3f}", "Lower is better"),
                ("Search time vs best fixed algorithm", f"{m['search_rel_best_fixed']:.3f}",
                 "Lower is better"),
                ("Best possible (perfect picks)", f"{m['oracle_rel_best_fixed']:.3f}", "Same baseline")]))
