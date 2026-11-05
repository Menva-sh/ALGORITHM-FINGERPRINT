"""Page 1: choose a graph (generate, upload or draw), see the prediction, verify it."""
from __future__ import annotations

import streamlit as st

from app import ui
from src.demo_logic import NAMES, confidence_label, reason, why_items
from src.inputs import FILE_TYPES, SAMPLE_MAZE, UnsupportedInput

CFG = ui.CFG
fams = CFG["generation"]["families"]

st.markdown("<h1>Which shortest-path algorithm is <span class='accent'>fastest?</span></h1>",
            unsafe_allow_html=True)
st.markdown("<p class='lead'>Give it a graph and two points. A small model predicts the "
            "fastest shortest-path algorithm from a quick fingerprint of the graph, then every "
            "algorithm is run so you can check the prediction.</p>", unsafe_allow_html=True)

FITS = ("<div class='fit'><b>Fits:</b> the shortest route between two points in a network "
        "whose distances are 0 or more: road and transport maps, mazes and grids, social or "
        "communication networks, game maps.<br><b>Does not fit:</b> visiting every stop "
        "(travelling salesman), negative distances, shortest paths between all pairs at once, "
        "flow or capacity problems.</div>")

# ------------------------------------------------------------------ 1. input
ui.step(1, "Choose a graph")
with st.container(key="card_input"):
    mode = st.segmented_control("Input", ["Generate", "Upload a file", "Draw a maze"],
                                default="Generate", label_visibility="collapsed",
                                key="input_mode") or "Generate"
    fixed_query = None
    labels = None

    if mode == "Generate":
        c1, c2, c3 = st.columns([1, 0.9, 1.3], gap="medium")
        family = c1.selectbox("Graph type", list(fams), format_func=ui.FAMILY_NAMES.get)
        n = c2.select_slider("Number of nodes", CFG["generation"]["sizes"], value=1000,
                             format_func=lambda v: f"{v:,}")
        w_opts = fams[family]["weights"]
        weights = c3.segmented_control("Edge weights", w_opts, default=w_opts[0],
                                       format_func=ui.WEIGHT_NAMES.get,
                                       key=f"weights_{family}") or w_opts[0]
        with st.expander("More settings"):
            a = st.columns(len(fams[family]["params"]) + 1)
            params = tuple(sorted((k, a[i].selectbox(k.replace("_", " "), v, key=f"{family}_{k}"))
                                  for i, (k, v) in enumerate(fams[family]["params"].items())))
            rep = a[-1].number_input("Random seed", 0, 99, 0)
        g = ui.generated_graph(family, n, params, weights, int(rep))
        graph_key = (family, n, params, weights, int(rep))
        source_note = ui.FAMILY_NAMES[family]

    elif mode == "Upload a file":
        u1, u2 = st.columns([1.4, 1], gap="large")
        with u1:
            up = st.file_uploader("Graph file", type=FILE_TYPES,
                                  help="Edge list (u v [distance]), distance matrix (CSV), "
                                       "GraphML, GML or node-link JSON.")
            cup = st.file_uploader("Coordinates (optional): one 'node x y' per line",
                                   type=["txt", "csv", "coords"])
        with u2:
            st.markdown("**Accepted formats**")
            st.markdown("- **Edge list**: `A B 4` per line (distance optional)\n"
                        "- **Distance matrix**: square CSV table, 0 = no road\n"
                        "- **GraphML / GML / JSON**: from Gephi, networkx, yEd...")
            st.markdown(FITS, unsafe_allow_html=True)
            st.markdown("**Example files**")
            for ex in sorted((ui.STATIC.parent / "examples").glob("*.*")):
                st.download_button(ex.name, ex.read_bytes(), file_name=ex.name,
                                   key=f"dl_{ex.name}", type="tertiary", icon=":material/download:")
        if up is None:
            st.info("Upload a file to continue, or download an example file first.")
            st.stop()
        coords_text = cup.getvalue().decode("utf-8-sig") if cup else None
        try:
            parsed = ui.parsed_upload(up.name, up.getvalue(), coords_text)
        except UnsupportedInput as e:
            st.error(f"**This input can't be used.** {e}")
            st.stop()
        g, labels = parsed.graph, parsed.labels
        graph_key = ("upload", up.name, hash(up.getvalue()), hash(coords_text))
        source_note = f"{up.name} ({parsed.fmt})"
        for note in parsed.notes:
            st.warning(note)

    else:
        m1, m2 = st.columns([1.4, 1], gap="large")
        text = m1.text_area("Maze", SAMPLE_MAZE, height=230, label_visibility="collapsed",
                            help="'#' = wall, any other character = open, S = start, T = target")
        m2.markdown("**How to draw**\n\n`#` wall, space or `.` open, `S` start, `T` target. "
                    "Moves go up, down, left and right.")
        m2.markdown(FITS, unsafe_allow_html=True)
        try:
            parsed = ui.parsed_maze(text)
        except UnsupportedInput as e:
            st.error(f"**This maze can't be used.** {e}")
            st.stop()
        g, labels = parsed.graph, parsed.labels
        fixed_query = (parsed.source, parsed.target)
        graph_key = ("maze", hash(text))
        source_note = "your maze"

    # ---- query
    st.divider()
    if fixed_query:
        s, t = fixed_query
    else:
        q1, q2 = st.columns([1, 3.2], vertical_alignment="center")
        if q1.button("New random query", type="primary", use_container_width=True):
            ui.new_query("try")
        query = ui.random_query(g, "try")
        if query is None:
            st.error("No two nodes of this graph are connected, so there is no path to find.")
            st.stop()
        s, t = query
        if labels is not None:
            with q2.popover("Choose start and end"):
                index = {name: i for i, name in enumerate(labels)}
                a_ = st.text_input("Start node", labels[s])
                b_ = st.text_input("End node", labels[t])
                if a_ in index and b_ in index and a_ != b_:
                    s, t = index[a_], index[b_]
                else:
                    st.caption("Type two different node names from the file.")
        if g.component_labels()[s] != g.component_labels()[t]:
            st.error("These two nodes are not connected: there is no path between them.")
            st.stop()
    name = (lambda i: labels[i]) if labels is not None else (lambda i: f"node {i}")
    st.markdown(f"<span class='muted'>From <b>{ui.esc(name(s))}</b> to <b>{ui.esc(name(t))}</b>"
                f" · {ui.esc(source_note)} · {g.n:,} nodes · {g.m:,} edges · "
                f"{'weighted' if g.weighted else 'unweighted'}"
                f"{' · map coordinates' if g.has_coordinates else ''}</span>",
                unsafe_allow_html=True)

race = st.session_state.get("race", 0)
r = ui.analyse(graph_key, g, s, t, race)
fp, rec, runs = r["fp"], r["rec"], r["runs"]
f = fp.features
st.session_state["your_graph"] = f  # shown on the algorithm map

# ------------------------------------------------------------------ 2. prediction
ui.step(2, "Prediction", "From the graph's fingerprint only, before any search is run.")
choice = rec["choice"]
label, kind = confidence_label(rec["confidence"])
with st.container(key="rec_main"):
    left, right = st.columns([2.2, 1], gap="large", vertical_alignment="center")
    left.markdown(f"<div class='eyebrow'>RECOMMENDED ALGORITHM</div>"
                  f"<div class='big-algo'>{ui.dot(choice)}{ui.esc(NAMES[choice])}</div>"
                  f"<div class='lead'>{ui.esc(reason(choice, f))}</div>",
                  unsafe_allow_html=True)
    right.markdown(f"<div style='display:flex;justify-content:space-between;align-items:end'>"
                   f"<b>Confidence</b><span style='font-size:1.6rem;font-weight:700' class='accent'>"
                   f"{rec['confidence']:.0%}</span></div>{ui.bar(rec['confidence'], ui.accent())}"
                   f"<div style='margin-top:10px'><span class='tag {kind}'>{label}</span></div>",
                   unsafe_allow_html=True)

st.write("")
why, fin = st.columns(2, gap="medium")
with why.container(key="card_why"):
    st.markdown("<h3>Why this pick?</h3><div class='muted small'>Longer bar means a bigger push "
                "toward this choice.</div>", unsafe_allow_html=True)
    items = why_items(rec["shap"], f)
    rows = "".join(f"<div class='why-row'><div>{ui.esc(lbl)}</div>{ui.bar(w, ui.accent())}</div>"
                   for lbl, w in items) or "<p class='muted'>No single feature stood out.</p>"
    st.markdown(rows, unsafe_allow_html=True)
with fin.container(key="card_fp"):
    st.markdown(f"<h3>Graph fingerprint</h3><div class='muted small'>Quick features, computed in "
                f"about {fp.tier_cost_ns[0] / 1e3:.1f} microseconds.</div>", unsafe_allow_html=True)
    tiles = [("Nodes", f"{int(f['n_nodes']):,}"), ("Edges", f"{int(f['n_edges']):,}"),
             ("Density", f"{f['density']:.3g}"),
             ("Average degree", f"{2 * f['n_edges'] / f['n_nodes']:.2f}"),
             ("Weighted", "Yes" if f["is_weighted"] else "No"),
             ("Coordinates", "Yes" if f["has_coordinates"] else "No")]
    st.markdown("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
        for k, v in tiles) + "</div>", unsafe_allow_html=True)
    with st.expander("See all features"):
        st.dataframe([{"tier": f"Tier-{tier}", "feature": nm, "value": f[nm]}
                      for tier in [0, 1, 2] for nm in CFG["features"]["tiers"][tier]],
                     hide_index=True)
        st.caption("Cost per tier: " + ", ".join(f"Tier-{k} {v / 1e3:,.1f} µs"
                                                 for k, v in fp.tier_cost_ns.items())
                   + ". Only Tier-0 is used: the richer tiers cost more than they save.")

# ------------------------------------------------------------------ 3. verification
h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
with h1:
    ui.step(3, "Verification", "Every algorithm that applies was actually run on this query.")
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
                    "Yes" if right_pick else f"No, {NAMES[fastest]}",
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
