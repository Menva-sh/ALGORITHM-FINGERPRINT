"""Page 1: pick a graph and a query, see the pick, the reasons and the real race."""
from __future__ import annotations

import streamlit as st

from app import ui
from src.demo_logic import NAMES, confidence_label, reason, why_items

CFG = ui.CFG
fams = CFG["generation"]["families"]

st.markdown("<h1>Which shortest-path algorithm is <span class='accent'>fastest?</span></h1>",
            unsafe_allow_html=True)
st.markdown("<p class='lead'>Pick a graph and a query. A small model predicts the fastest "
            "algorithm from a quick fingerprint of the graph. Then we run every algorithm so "
            "you can check it.</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ controls
with st.container(key="card_controls"):
    c1, c2, c3 = st.columns([1, 1, 1.05], gap="large")
    family = c1.selectbox("Graph type", list(fams), format_func=ui.FAMILY_NAMES.get)
    n = c2.select_slider("Number of nodes", CFG["generation"]["sizes"], value=1000,
                         format_func=lambda v: f"{v:,}")
    w_opts = fams[family]["weights"]
    weights = c3.segmented_control("Edge weights", w_opts, default=w_opts[0],
                                   format_func=ui.WEIGHT_NAMES.get, key=f"weights_{family}")
    weights = weights or w_opts[0]
    with st.expander("Advanced"):
        a = st.columns(len(fams[family]["params"]) + 1)
        params = tuple(sorted((k, a[i].selectbox(k.replace("_", " "), v, key=f"{family}_{k}"))
                              for i, (k, v) in enumerate(fams[family]["params"].items())))
        rep = a[-1].number_input("Random seed", 0, 99, 0)
        own = st.toggle("Use my own edge list instead")
        if own:
            edges = st.file_uploader("Edge list: one 'u v' or 'u v w' per line",
                                     type=["txt", "edges", "csv"])
            coords = st.file_uploader("Optional coordinates: 'id x y' per line",
                                      type=["txt", "coords"])

    if own and edges is not None:
        text = edges.getvalue().decode("utf-8")
        ctext = coords.getvalue().decode("utf-8") if coords else None
        try:
            g = ui.uploaded_graph(text, ctext)
        except ValueError as e:
            st.error(f"Could not read the edge list: {e}")
            st.stop()
        graph_key = ("upload", hash(text), hash(ctext))
    else:
        if own:
            st.info("Upload an edge list under Advanced, or switch the toggle off.")
            st.stop()
        g = ui.generated_graph(family, n, params, weights, int(rep))
        graph_key = (family, n, params, weights, int(rep))

    q1, q2 = st.columns([1, 4], vertical_alignment="center")
    if q1.button("New random query", type="primary"):
        ui.new_query("try")
    query = ui.random_query(g, "try")
    if query is None:
        st.error("This graph has no two connected nodes.")
        st.stop()
    s, t = query
    q2.markdown(f"<span class='muted'>Query: node {s} → node {t} · {g.n:,} nodes · {g.m:,} edges"
                f" · {'weighted' if g.weighted else 'unweighted'}"
                f"{' · map coordinates' if g.has_coordinates else ''}</span>",
                unsafe_allow_html=True)

race = st.session_state.get("race", 0)
r = ui.analyse(graph_key, g, s, t, race)
fp, rec, runs = r["fp"], r["rec"], r["runs"]
f = fp.features
st.session_state["your_graph"] = f  # shown on the algorithm map

# ------------------------------------------------------------------ recommendation
choice = rec["choice"]
label, kind = confidence_label(rec["confidence"])
with st.container(key="rec_main"):
    left, right = st.columns([2.2, 1], gap="large", vertical_alignment="center")
    left.markdown(f"<div class='eyebrow'>RECOMMENDED ALGORITHM</div>"
                  f"<div class='big-algo'>{ui.dot(choice)}{ui.esc(NAMES[choice])}</div>"
                  f"<div class='lead' style='color:#33403f'>{ui.esc(reason(choice, f))}</div>",
                  unsafe_allow_html=True)
    right.markdown(f"<div style='display:flex;justify-content:space-between;align-items:end'>"
                   f"<b>Confidence</b><span style='font-size:1.9rem;font-weight:700;color:#1f5e58'>"
                   f"{rec['confidence']:.0%}</span></div>{ui.bar(rec['confidence'], ui.TEAL)}"
                   f"<div style='margin-top:12px'><span class='pill {kind}'>{label}</span></div>",
                   unsafe_allow_html=True)

st.write("")
why, fin = st.columns(2, gap="medium")
with why.container(key="card_why"):
    st.markdown("<h3>Why this pick?</h3><div class='muted'>Longer bar means a bigger push "
                "toward this choice.</div>", unsafe_allow_html=True)
    items = why_items(rec["shap"], f)
    rows = "".join(f"<div class='why-row'><div>{ui.esc(lbl)}</div>{ui.bar(w, ui.TEAL)}</div>"
                   for lbl, w in items) or "<p class='muted'>No single feature stood out.</p>"
    st.markdown(rows, unsafe_allow_html=True)
with fin.container(key="card_fp"):
    st.markdown(f"<h3>Graph fingerprint</h3><div class='muted'>Quick features, computed in about "
                f"{fp.tier_cost_ns[0] / 1e3:.1f} microseconds.</div>", unsafe_allow_html=True)
    tiles = [("Nodes", f"{int(f['n_nodes']):,}"), ("Edges", f"{int(f['n_edges']):,}"),
             ("Density", f"{f['density']:.3g}"),
             ("Average degree", f"{2 * f['n_edges'] / f['n_nodes']:.2f}"),
             ("Weighted", "Yes" if f["is_weighted"] else "No"),
             ("Coordinates", "Yes" if f["has_coordinates"] else "No")]
    st.markdown("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        for k, v in tiles) + "</div>", unsafe_allow_html=True)
    with st.expander("See all features"):
        st.dataframe([{"tier": f"Tier-{tier}", "feature": name, "value": f[name]}
                      for tier in [0, 1, 2] for name in CFG["features"]["tiers"][tier]],
                     hide_index=True)
        st.caption("Cost per tier: " + ", ".join(f"Tier-{k} {v / 1e3:,.1f} µs"
                                                 for k, v in fp.tier_cost_ns.items())
                   + ". Only Tier-0 is used: the richer tiers cost more than they save.")

# ------------------------------------------------------------------ the race
st.write("")
h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
h1.markdown("<h2>The real race</h2><div class='muted'>We ran every algorithm that applies, so "
            "you can check the pick.</div>", unsafe_allow_html=True)
if h2.button("Run the race again", use_container_width=True):
    st.session_state["race"] = race + 1
    st.rerun()

fastest = runs.iloc[0]["algorithm"]
right_pick = choice == fastest
regret = r["chosen_us"] / runs.iloc[0]["median_us"] - 1
net = r["chosen_us"] + r["overhead_us"]
saving = 1 - net / r["fixed_us"]
k1, k2, k3 = st.columns(3)
k1.markdown(ui.stat("Was the pick right?",
                    "✓ Yes" if right_pick else f"No, {NAMES[fastest]}",
                    f"Fastest of {len(runs)} algorithms" if right_pick
                    else f"Picked {NAMES[choice]}", "good" if right_pick else "bad"),
            unsafe_allow_html=True)
k2.markdown(ui.stat("Time lost to the best pick", f"{regret:.0%}",
                    "No regret on this query" if regret <= 0 else
                    f"{r['chosen_us'] - runs.iloc[0]['median_us']:,.1f} µs slower than the fastest"),
            unsafe_allow_html=True)
k3.markdown(ui.stat(f"Versus always {NAMES[r['best_fixed']]}",
                    f"{saving:.0%} faster" if saving >= 0 else f"{-saving:.0%} slower",
                    f"{net:,.0f} µs vs {r['fixed_us']:,.0f} µs, prediction time included",
                    "good" if saving >= 0 else "bad"), unsafe_allow_html=True)

st.write("")
with st.container(key="card_race"):
    st.markdown("<b>Search time per algorithm (shorter is better)</b>" + ui.race_rows(runs),
                unsafe_allow_html=True)
    with st.expander("Show the raw numbers"):
        st.dataframe(runs.assign(algorithm=runs["algorithm"].map(NAMES)).round(2),
                     hide_index=True)

b = CFG["benchmark"]
st.markdown(f"<div class='footer'>Algorithm Selection via Learned Complexity Fingerprints · "
            f"final project. Each algorithm: {b['warmup_runs']} warm-up runs, median of "
            f"{b['repeats']} timed runs. Timings on this server are noisier than the "
            f"benchmark's.</div>", unsafe_allow_html=True)
