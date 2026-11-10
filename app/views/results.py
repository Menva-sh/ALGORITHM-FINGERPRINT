"""Page 4: report tables and figures, each read from a file in results/ and ready to export."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import streamlit as st

from app import ui
from src.config import PROJECT_ROOT
from src.demo_logic import NAMES

CFG = ui.CFG
SETS = {"T0": "Quick (T0)", "T0+1": "Quick and scan (T0+T1)",
        "T0+probe64": "Quick and probe (T0+probe64)"}
MODELS = {"plain_tree": "Decision tree", "random_forest": "Random forest", "svm": "SVM (RBF)",
          "mlp": "MLP", "best_fixed": "Best single fixed algorithm",
          "always_dijkstra": "Always Dijkstra", "random": "Random choice",
          "oracle": "Oracle (always the fastest)"}


def bold_best(values: pd.Series, best: float, fmt: str) -> list[str]:
    return [f"<b>{fmt.format(v)}</b>" if abs(v - best) < 1e-12 else fmt.format(v) for v in values]


def slot(key: str, title: str, sub: str, what: str, command: str) -> None:
    """Placeholder for a table or figure whose results file does not exist yet."""
    with st.container(key=f"slot_{key}"):
        ui.md(f"<div class='slot'><div style='display:flex;justify-content:space-between;"
              f"align-items:center;gap:8px'><div class='ttl'><h2 style='font-size:14px;margin:0;"
              f"padding:0'>{ui.esc(title)}</h2><span class='sub'>{ui.esc(sub)}</span></div>"
              f"<span class='chip'>Not generated yet</span></div><p style='font-size:13.5px;"
              f"color:var(--ink2);margin:0'>{ui.esc(what)}</p></div>")
        st.code(command, language="bash")


def figure(col, key: str, title: str, sub: str, png: str, note: str, command: str) -> None:
    path = ui.RESULTS / png
    with col:
        if not path.exists():
            slot(key, title, sub, note, command)
            return
        with st.container(key=f"panel_{key}", height="stretch"):
            head = ui.panel_head(key, title, sub)
            head.download_button("PNG", path.read_bytes(), path.name, "image/png", type="tertiary")
            st.image(str(path), use_container_width=True)
            ui.md(ui.caption(f"{title}.", note))


# ------------------------------------------------------------------ data
road_all = ui.result("shift_real_by_graph.csv")
road = None if road_all is None else road_all[road_all["graph_id"].str.contains("road")]
models = ui.result("model_results.csv")
amap = ui.algorithm_map()

latex = []
if road is not None and not road.empty:
    t1 = road[["feature_set", "model", "accuracy", "regret_mean", "search_rel_best_fixed"]].copy()
    latex.append(t1.to_latex(index=False, float_format="%.3f",
                             caption="Selection quality on the Piedmont road network.",
                             label="tab:road"))
if models is not None:
    t2 = models.rename(columns={models.columns[0]: "method"}).query("method != 'run_all'")
    t2 = t2[["method", "accuracy", "regret_mean", "regret_p95", "saved_vs_best_fixed"]]
    t2 = t2.sort_values("method", key=lambda m: m.map({k: i for i, k in enumerate(MODELS)}))
    latex.append(t2.to_latex(index=False, float_format="%.3f",
                             caption="Selectors against baselines on held-out graphs.",
                             label="tab:baselines"))

figs = sorted(ui.RESULTS.glob("fig_*.png"))
zbuf = io.BytesIO()
with zipfile.ZipFile(zbuf, "w") as z:
    for p in figs:
        z.write(p, p.name)

top = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom")
with top:
    ui.header("Results", "Tables and figures for the report, each generated from a file in "
              "results/ and ready to export.")
buttons = top.container(horizontal=True, gap="small", width="content")
buttons.download_button("Export tables as LaTeX", "\n\n".join(latex), "tables.tex", "text/x-tex",
                        disabled=not latex)
buttons.download_button("Download all figures", zbuf.getvalue(), "figures.zip", "application/zip",
                        type="primary", disabled=not figs)

# ------------------------------------------------------------------ Table 1 and Figure 4
c1, c2 = st.columns([1.5, 1], gap="small")
with c1:
    if road is None or road.empty:
        slot("t1", "Table 1", "Selection quality on the Piedmont road network",
             "Accuracy, regret and search time of each selector on 100 random road queries.",
             "python -m src.experiments.real_graphs")
    else:
        with st.container(key="panel_t1", height="stretch"):
            head = ui.panel_head("t1", "Table 1", "Selection quality on the Piedmont road network")
            head.download_button("LaTeX", latex[0], "table1_road.tex", "text/x-tex", type="tertiary")
            head.download_button("CSV", road.to_csv(index=False), "table1_road.csv", "text/csv",
                                 type="tertiary")
            acc = bold_best(road["accuracy"], road["accuracy"].max(), "{:.2f}")
            reg = bold_best(road["regret_mean"], road["regret_mean"].min(), "{:.3f}")
            rel = bold_best(road["search_rel_best_fixed"], road["search_rel_best_fixed"].min(), "{:.3f}")
            rows, seen = [], set()
            for i, (_, x) in enumerate(road.iterrows()):
                fs = x["feature_set"]
                span = int((road["feature_set"] == fs).sum())
                first = fs not in seen
                seen.add(fs)
                lead = (f"<td rowspan='{span}'>{ui.esc(SETS.get(fs, fs))}</td>" if first else "")
                rows.append(f"<tr class='{'grp' if first and i else ''}'>{lead}"
                            f"<td>{ui.esc(MODELS.get(x['model'], x['model']))}</td>"
                            f"<td class='n'>{acc[i]}</td><td class='n'>{reg[i]}</td>"
                            f"<td class='n'>{rel[i]}</td></tr>")
            orc = road["oracle_rel_best_fixed"].iloc[0]
            rows.append(f"<tr class='orc'><td colspan='2'>Oracle (always the fastest)</td>"
                        f"<td class='n'>1.00</td><td class='n'>0.000</td><td class='n'>{orc:.3f}</td></tr>")
            misses = road.drop_duplicates("feature_set").query("chosen_most != best_most")
            extra = "".join(f" With {SETS.get(x['feature_set'], x['feature_set'])} features the "
                            f"selector chose {NAMES[x['chosen_most']]} on most queries, while "
                            f"{NAMES[x['best_most']]} was usually fastest."
                            for _, x in misses.iterrows())
            ui.md("<table class='bk'><thead><tr><th>Features</th><th>Model</th><th class='n'>Accuracy"
                  "</th><th class='n'>Mean regret</th><th class='n'>Search time vs best fixed</th></tr>"
                  "</thead><tbody>" + "".join(rows) + "</tbody></table>")
            ui.md(ui.caption("Table 1.", f"{int(road['queries'].iloc[0])} random queries on one real "
                             "network. Search time is relative to the best single fixed algorithm, "
                             "so lower is better; bold marks the best selector result in each "
                             f"column.{extra}"))

with c2:
    if amap is None:
        slot("f4", "Figure 4", "Fastest algorithm per graph", "How often each algorithm wins.",
             "python -m src.experiments.algorithm_map")
    else:
        counts = amap["winner"].value_counts()
        with st.container(key="panel_f4", height="stretch"):
            head = ui.panel_head("f4", "Figure 4", "Fastest algorithm per graph")
            head.download_button("CSV", counts.rename("graphs").rename_axis("algorithm").to_csv(),
                                 "figure4_winners.csv", "text/csv", type="tertiary")
            ui.md(ui.win_rows(counts))
            ui.md(ui.caption("Figure 4.", "Number of benchmark graphs on which each algorithm was "
                             f"fastest on most queries ({len(amap):,} graphs in total)."))

st.write("")
# ------------------------------------------------------------------ Table 2, Figures 5 and 6
c3, c4, c5 = st.columns([1.35, 1, 1], gap="small")
with c3:
    if models is None:
        slot("t2", "Table 2", "Baseline comparison", "The selector against always-Dijkstra, the "
             "best single fixed algorithm, random choice and the oracle: accuracy, regret and net "
             "time saved.", "python -m src.evaluate")
    else:
        with st.container(key="panel_t2", height="stretch"):
            head = ui.panel_head("t2", "Table 2", "Baseline comparison")
            head.download_button("LaTeX", latex[-1], "table2_baselines.tex", "text/x-tex",
                                 type="tertiary")
            head.download_button("CSV", t2.to_csv(index=False), "table2_baselines.csv", "text/csv",
                                 type="tertiary")
            body = []
            for i, (_, x) in enumerate(t2.iterrows()):
                cls = "orc" if x["method"] == "oracle" else ("grp" if x["method"] == "best_fixed" else "")
                body.append(f"<tr class='{cls}'><td>{ui.esc(MODELS.get(x['method'], x['method']))}</td>"
                            f"<td class='n'>{x['accuracy']:.2f}</td><td class='n'>{x['regret_mean']:.3f}"
                            f"</td><td class='n'>{format(x['saved_vs_best_fixed'], '+,.1%')
                                                  .replace('-', '−')}</td></tr>")
            ui.md("<table class='bk'><thead><tr><th>Method</th><th class='n'>Accuracy</th>"
                  "<th class='n'>Mean regret</th><th class='n'>Net time saved</th></tr></thead>"
                  "<tbody>" + "".join(body) + "</tbody></table>")
            ui.md(ui.caption("Table 2.", "Held-out graphs (group split). Net time saved is against "
                             "the best single fixed algorithm and includes feature extraction and "
                             "model inference, so a negative value means slower overall."))

figure(c4, "f5", "Figure 5", "Feature cost against quality", "fig_tiers_pareto.png",
       "Regret against extraction time for feature tiers T0 to T3, with the Pareto frontier "
       "marked and extraction time counted in the total.", "python -m src.experiments.tiers")
figure(c5, "f6", "Figure 6", "Value of probing", "fig_probing_tradeoff.png",
       "Accuracy gained per millisecond of probing, static features against static plus probe "
       "features, with probe time counted.", "python -m src.experiments.probing")


# ------------------------------------------------------------------ reproducibility
@st.cache_data(show_spinner=False)
def hardware() -> str | None:
    path = PROJECT_ROOT / CFG["paths"]["dataset"]
    if not path.exists():
        return None
    d = pd.read_csv(path, usecols=["machine", "cpu"])
    cpus = ", ".join(str(c) for c in sorted(d["cpu"].dropna().unique()))
    n = d["machine"].nunique()
    return f"{n} machine{'s' if n != 1 else ''}, pinned CPUs {cpus}"


st.write("")
b = CFG["benchmark"]
hw = hardware()
with st.container(key="panel_repro"):
    ui.panel_head("repro", "Reproducibility", "Report these in the methods section")
    ui.md(ui.stats([("Random seed", str(CFG["seed"]), ""), ("Warm-up runs", str(b["warmup_runs"]), ""),
                    ("Timed runs", f"{b['repeats']}, median", ""), ("Timer", "perf_counter_ns", ""),
                    ("Training set", f"{ui.training_graphs():,} synthetic graphs", ""),
                    ("Hardware", hw or "<span style='color:var(--mute)'>Not recorded yet</span>", "")],
                   small=True))
    st.code("python -m src.reproduce", language="bash")
