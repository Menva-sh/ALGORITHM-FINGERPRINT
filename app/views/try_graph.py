"""Page 1: choose a graph (generate, upload or draw), see the prediction, verify it."""
from __future__ import annotations

import json

import streamlit as st

from app import ui
from src.demo_logic import NAMES, reason
from src.inputs import FILE_TYPES, SAMPLE_MAZE, UnsupportedInput

CFG = ui.CFG
fams = CFG["generation"]["families"]
EXAMPLES = ui.STATIC.parent / "examples"
READY = [("social_network.graphml", "Social network"),
         ("warehouses_distance_matrix.csv", "Warehouses"),
         ("city_roads_edge_list.csv", "Neighbourhood roads")]  # one per input type

top = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom")
with top:
    ui.header("Try a graph", "Pick a graph and a query. The selector predicts the fastest "
              "algorithm, then every algorithm is run to check it.")
export_slot = top.empty()

FITS = ("<p class='sub'><b>Fits:</b> the shortest route between two points in a network whose "
        "distances are 0 or more: road and transport maps, mazes and grids, social or "
        "communication networks, game maps. <b>Does not fit:</b> visiting every stop, negative "
        "distances, all pairs at once, flow problems.</p>")

# ------------------------------------------------------------------ graph input panel
with st.container(key="panel_input"):
    first = st.columns([2.1, 1.1, 1, 1.15], gap="medium", vertical_alignment="bottom")
    mode = first[0].segmented_control("Graph source", ["Ready-made or upload", "Generate", "Draw a maze"],
                                      default="Ready-made or upload", key="input_mode"
                                      ) or "Ready-made or upload"
    body = st.container()
    st.markdown("<div style='border-top:1px solid var(--grid);margin:2px 0 0'></div>",
                unsafe_allow_html=True)
    qtext, qmore, qbtn = st.columns([3.2, 0.75, 0.85], gap="small", vertical_alignment="center")
    fixed_query = None
    labels = None

    if mode == "Generate":
        family = first[1].selectbox("Graph type", list(fams), format_func=ui.FAMILY_NAMES.get)
        n = first[2].select_slider("Number of nodes", CFG["generation"]["sizes"], value=1000,
                                   format_func=lambda v: f"{v:,}")
        w_opts = fams[family]["weights"]
        weights = first[3].segmented_control("Edge weights", w_opts, default=w_opts[0],
                                             format_func=ui.WEIGHT_NAMES.get,
                                             key=f"weights_{family}") or w_opts[0]
        with qmore.popover("More settings", use_container_width=True):
            params = tuple(sorted((k, st.selectbox(k.replace("_", " ").capitalize(), v,
                                                   key=f"{family}_{k}"))
                                  for k, v in fams[family]["params"].items()))
            rep = st.number_input("Random seed", 0, 99, 0)
        g = ui.generated_graph(family, n, params, weights, int(rep))
        graph_key = (family, n, params, weights, int(rep))
        source_note = f"{ui.FAMILY_NAMES[family].split(' (')[-1].rstrip(')')} graph"

    elif mode == "Ready-made or upload":
        u1, u2 = body.columns([1.4, 1], gap="large")
        with u1:
            ready_row = st.container(horizontal=True, gap="small")
            up = st.file_uploader("Or upload your own graph file", type=FILE_TYPES,
                                  help="Edge list (u v [distance]), distance matrix (CSV), "
                                       "GraphML, GML or node-link JSON.")
            cup = st.file_uploader("Coordinates (optional): one 'node x y' per line",
                                   type=["txt", "csv", "coords"])
        with u2:
            st.markdown("**Accepted formats**\n\n"
                        "- **Edge list**: `A B 4` per line (distance optional)\n"
                        "- **Distance matrix**: square CSV table, 0 = no road\n"
                        "- **GraphML / GML / JSON**: from Gephi, networkx, yEd...")
            ui.md(FITS)
        # Ready-made inputs: the example files, one click instead of an upload.
        chosen = st.session_state.get("ready_file", READY[0][0])
        ready_row.markdown("<span class='sub'>Ready-made inputs:</span>", unsafe_allow_html=True)
        for fname, label in READY:
            ready_row.button(label, key=f"ready_{fname}", type="primary" if fname == chosen
                             and up is None else "secondary",
                             on_click=lambda f=fname: st.session_state.update(ready_file=f))
        if up is not None:
            fname, data = up.name, up.getvalue()
        else:
            fname, data = chosen, (EXAMPLES / chosen).read_bytes()
        coords_text = cup.getvalue().decode("utf-8-sig") if cup else None
        try:
            parsed = ui.parsed_upload(fname, data, coords_text)
        except UnsupportedInput as e:
            qtext.error(f"**This input can't be used.** {e}")
            st.stop()
        g, labels = parsed.graph, parsed.labels
        graph_key = ("upload", fname, hash(data), hash(coords_text))
        source_note = f"{fname} ({parsed.fmt})"
        for note in parsed.notes:
            body.warning(note)

    else:
        m1, m2 = body.columns([1.4, 1], gap="large")
        text = m1.text_area("Maze", SAMPLE_MAZE, height=230, label_visibility="collapsed",
                            help="'#' = wall, any other character = open, S = start, T = target")
        m2.markdown("**How to draw**\n\n`#` wall, space or `.` open, `S` start, `T` target. "
                    "Moves go up, down, left and right.")
        with m2:
            ui.md(FITS)
        try:
            parsed = ui.parsed_maze(text)
        except UnsupportedInput as e:
            qtext.error(f"**This maze can't be used.** {e}")
            st.stop()
        g, labels = parsed.graph, parsed.labels
        fixed_query = (parsed.source, parsed.target)
        graph_key = ("maze", hash(text))
        source_note = "Your maze"

    # ---- query
    if fixed_query:
        s, t = fixed_query
    else:
        if qbtn.button("New random query", type="primary", use_container_width=True):
            ui.new_query("try")
        query = ui.random_query(g, "try")
        if query is None:
            qtext.error("No two nodes of this graph are connected, so there is no path to find.")
            st.stop()
        s, t = query
        if labels is not None:
            with qmore.popover("Start and end", use_container_width=True):
                index = {name: i for i, name in enumerate(labels)}
                a_ = st.text_input("Start node", labels[s])
                b_ = st.text_input("End node", labels[t])
                if a_ in index and b_ in index and a_ != b_:
                    s, t = index[a_], index[b_]
                else:
                    st.caption("Type two different node names from the file.")
        if g.component_labels()[s] != g.component_labels()[t]:
            qtext.error("These two nodes are not connected: there is no path between them.")
            st.stop()
    name = (lambda i: labels[i]) if labels is not None else (lambda i: f"node {i}")
    qtext.markdown(f"<p style='margin:0;color:var(--ink2)'><b style='color:var(--ink)'>Query: "
                   f"{ui.esc(name(s))} to {ui.esc(name(t))}.</b> {ui.esc(source_note)} with "
                   f"{g.n:,} nodes and {g.m:,} edges, "
                   f"{'weighted' if g.weighted else 'unweighted'}"
                   f"{', with map coordinates' if g.has_coordinates else ''}.</p>",
                   unsafe_allow_html=True)

r = ui.analyse(graph_key, g, s, t)
fp, rec, runs = r["fp"], r["rec"], r["runs"]
f = fp.features
choice = rec["choice"]
st.session_state["your_graph"] = f  # shown on the algorithm map
st.session_state["your_choice"] = choice

# ------------------------------------------------------------------ prediction, why, fingerprint
p_col, w_col, f_col = st.columns(3, gap="small")

p_col.markdown(ui.panel("Prediction", (
    "<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:12px'>"
    f"<div style='display:flex;flex-direction:column;gap:6px'><div style='display:flex;"
    f"align-items:center;gap:10px'>{ui.glyph(choice, 15)}<span class='big'>{ui.esc(NAMES[choice])}"
    f"</span></div><p style='font-size:13.5px;color:var(--ink2);margin:0;max-width:260px'>"
    f"{ui.esc(reason(choice, f))}</p></div>"
    "<div style='text-align:right'><div class='sub'>Confidence</div><div style='font-size:28px;"
    f"line-height:1.1;font-weight:600'>{rec['confidence']:.0%}</div></div></div>"
    + ui.prob_rows(rec["probs"])), "Made before any search runs", min_height=300),
    unsafe_allow_html=True)

shap = rec["shap"].reindex(rec["shap"].abs().sort_values(ascending=False).index)
w_col.markdown(ui.panel("Why this prediction", ui.hbars(
    [{"label": ui.esc(ui.FEATURE_NAMES.get(k, k)), "value": float(v),
      "color": ui.COLORS[choice] if v >= 0 else ui.NEG} for k, v in shap.items()],
    label_w=116, fmt="{:+.2f}"), f"SHAP value toward {NAMES[choice]}", min_height=300),
    unsafe_allow_html=True)

with f_col.container(key="panel_fp", height="stretch"):
    head = ui.panel_head("fp", "Fingerprint")
    with head.popover("All features"):
        st.dataframe([{"tier": f"T{tier}", "feature": nm, "value": f[nm]}
                      for tier in [0, 1, 2] for nm in CFG["features"]["tiers"][tier]],
                     hide_index=True)
    rows = [("Nodes", f"{int(f['n_nodes']):,}", 0), ("Edges", f"{int(f['n_edges']):,}", 0),
            ("Density", f"{f['density']:.3g}", 0), ("Weighted", "Yes" if f["is_weighted"] else "No", 0),
            ("Has coordinates", "Yes" if f["has_coordinates"] else "No", 0),
            ("Mean degree", f"{f['deg_mean']:.2f}", 1)]
    cost = ", ".join(f"T{k} {v / 1e3:,.1f} µs" for k, v in fp.tier_cost_ns.items())
    used = "+".join(f"T{t}" for t in CFG["model"]["feature_tiers"])
    ui.md("<table class='tb'><thead><tr><th>Feature</th><th class='n'>Value</th>"
          "<th class='n'>Tier</th></tr></thead><tbody>"
          + "".join(f"<tr><td>{k}</td><td class='n'>{v}</td><td class='n'>T{t}</td></tr>"
                    for k, v, t in rows)
          + f"</tbody></table><p class='sub' style='margin:8px 0 0'>The selector uses {used} only. "
            f"Extraction cost: {cost}.</p>")

# ------------------------------------------------------------------ Figure 1: the race
fastest = runs.iloc[0]["algorithm"]
right_pick = choice == fastest
regret = r["chosen_us"] / runs.iloc[0]["median_us"] - 1
net = r["chosen_us"] + r["overhead_us"]
saving = 1 - net / r["fixed_us"]
table = runs.assign(algorithm=runs["algorithm"].map(NAMES)).round(2)

with st.container(key="panel_fig1"):
    head = ui.panel_head("fig1", "Figure 1", "Median search time per algorithm, in microseconds")
    with head.popover("View as table"):
        st.dataframe(table, hide_index=True)
    head.download_button("CSV", table.to_csv(index=False), "figure1_search_time.csv",
                         "text/csv", type="tertiary")
    bars_col, sum_col = st.columns([1.8, 1], gap="large")

    def chip(a: str) -> str:
        text = {(True, True): "Predicted and fastest", (True, False): "Predicted",
                (False, True): "Fastest"}.get((a == choice, a == fastest))
        return f"<span class='chip{'' if a == fastest else ' warn'}'>{text}</span>" if text else ""

    bars_col.markdown(ui.hbars([{"label": ui.algo_label(a, 11), "value": v,
                                 "color": ui.COLORS[a], "extra": chip(a)}
                                for a, v in zip(runs["algorithm"], runs["median_us"])]),
                      unsafe_allow_html=True)
    sum_col.markdown(ui.dl([
        ("Prediction", ui.verdict(right_pick, "Correct" if right_pick
                                  else f"Wrong, {NAMES[fastest]} was faster")),
        ("Regret", f"<span class='num'>{regret:.1%}</span>"),
        ("Total with prediction", f"<span class='num'>{ui.us(net)}</span>"),
        (f"Always {ui.esc(NAMES[r['best_fixed']])}", f"<span class='num'>{ui.us(r['fixed_us'])}</span>"),
        ("Time saved" if saving >= 0 else "Time lost", f"<span class='num'>{abs(saving):.1%}</span>"),
    ]), unsafe_allow_html=True)
    b = CFG["benchmark"]
    ui.md(ui.caption("Figure 1.", (
        f"Median over {b['repeats']} timed runs after {b['warmup_runs']} warm-up runs. All "
        f"{len(runs)} algorithms return the same path cost ({runs['path_cost'].iloc[0]:,.6g}). "
        f"Total with prediction adds quick-feature extraction and model inference "
        f"({ui.us(r['overhead_us'])}). Timings on a shared server are noisier than the benchmark's.")))

export = {"query": {"source": name(s), "target": name(t), "graph": source_note, "nodes": g.n,
                    "edges": g.m, "weighted": g.weighted, "coordinates": g.has_coordinates},
          "features": {k: float(v) for k, v in f.items()},
          "feature_cost_us": {f"T{k}": v / 1e3 for k, v in fp.tier_cost_ns.items()},
          "prediction": {"algorithm": choice, "confidence": rec["confidence"], "probs": rec["probs"],
                         "shap": {k: float(v) for k, v in rec["shap"].items()}},
          "search_time_us": dict(zip(runs["algorithm"], runs["median_us"])),
          "fastest": fastest, "regret": regret, "total_with_prediction_us": net,
          "best_fixed": r["best_fixed"], "best_fixed_us": r["fixed_us"]}
export_slot.download_button("Export this run", json.dumps(export, indent=2), "run.json",
                            "application/json")
