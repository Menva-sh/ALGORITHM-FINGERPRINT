"""Home: what the project is, how it works, the algorithms, the data, and the headline results."""
from __future__ import annotations

import streamlit as st

from app import ui
from src.config import PROJECT_ROOT
from src.demo_logic import NAMES

CFG = ui.CFG
WHEN = {"bidirectional_dijkstra": "Weighted graphs without coordinates: the most common winner.",
        "bidirectional_bfs": "Larger unweighted graphs.",
        "astar": "Graphs with map coordinates and distance weights.",
        "bfs": "Small unweighted graphs.",
        "dijkstra": "Rarely: when the other searches' extra bookkeeping does not pay off."}


def section(title: str, accent: str) -> None:
    """Two-tone section heading: grey lead-in, then the point in black."""
    ui.md(f"<p class='sec'>{ui.esc(title)} <b>{ui.esc(accent)}</b></p>")


ui.md("<div class='hero'><span class='tag'>Shortest-path algorithm selection</span>"
      "<h1>Algorithm <b>Fingerprint</b></h1>"
      "<p class='lead'>Which shortest-path algorithm will be fastest for this graph? A small "
      "model reads the graph's fingerprint and decides before any search runs.</p></div>")
_, cta, _ = st.columns([2, 1, 2])
with cta.container(key="home_cta"):
    st.page_link("views/try_graph.py", label="Try a graph", icon=":material/play_arrow:",
                 width="stretch")

# ------------------------------------------------------------------ headline numbers
model = ui.result("model_results.csv")
n_train = ui.training_graphs()
if model is not None and n_train:
    tree = model.set_index(model.columns[0]).loc["plain_tree"]
    ui.md("<div class='cards' style='margin-top:18px'>"
          f"<div class='mc cr'><div class='k'>Picks the fastest algorithm</div>"
          f"<div class='v'>{tree['accuracy']:.1%}</div>"
          f"<div class='s'>on {ui.holdout_graphs():,} held-out graphs</div></div>"
          f"<div class='mc'><div class='k'>Total time saved</div>"
          f"<div class='v'>{tree['saved_vs_best_fixed']:.0%}</div>"
          "<div class='s'>vs the best single algorithm, deciding cost included</div></div>"
          f"<div class='mc'><div class='k'>Candidate algorithms</div><div class='v'>{len(NAMES)}</div>"
          "<div class='s'>all written for this project</div></div>"
          f"<div class='mc bk'><div class='k'>Graphs benchmarked</div>"
          f"<div class='v'>{n_train:,}</div><div class='s'>five families, 100 to 5,000 nodes</div>"
          "</div></div>")
else:
    st.info("Run `python -m src.reproduce` to generate the results shown here.")

# ------------------------------------------------------------------ how it works
section("How it works.", "Four steps, one algorithm.")
ui.md("<div class='flow'>"
      "<div class='st'><span class='n'>01</span><b>Graph and query</b>"
      "<span class='d'>A network plus a start and an end.</span></div>"
      "<div class='st'><span class='n'>02</span><b>Fingerprint</b>"
      "<span class='d'>A few cheap facts: size, density, weighted, coordinates.</span></div>"
      "<div class='st'><span class='n'>03</span><b>Selector</b>"
      "<span class='d'>A small decision tree trained on benchmark timings.</span></div>"
      "<div class='st hi'><span class='n'>04</span><b>One algorithm</b>"
      "<span class='d'>Only the predicted one runs.</span></div></div>"
      "<p class='sub' style='margin:12px 0 0'>The time spent building the fingerprint and running "
      "the model is counted in the total, so a prediction only wins if the whole thing is "
      "still faster.</p>")

# ------------------------------------------------------------------ the algorithms
section("The algorithms.", "When each one wins.")
wins = ui.result("labels_overall.csv")
share = wins.set_index(wins.columns[0]).loc["all queries"] if wins is not None else None
rows = "".join(
    f"<tr><td>{ui.algo_label(a, 11)}</td><td>{ui.esc(WHEN[a])}</td>"
    f"<td class='cr n'>{'' if share is None else f'{share[a]:.0%}'}</td></tr>" for a in WHEN)
ui.md("<table class='tb ctb'><thead><tr><th>Algorithm</th><th>Usually fastest on</th>"
      f"<th class='cr n'>Queries won</th></tr></thead><tbody>{rows}</tbody></table>"
      "<p class='sub' style='margin:10px 0 0'>BFS variants only apply to unweighted graphs and "
      "A* only to graphs with coordinates; the selector chooses among the ones that apply. "
      "Source: results/labels_overall.csv.</p>")

# ------------------------------------------------------------------ the data
section("The data.", "Where it comes from.")
real = sorted(p.stem for p in (PROJECT_ROOT / CFG["paths"]["real_graphs"]).glob("*.edges")
              if not p.stem.startswith("road_"))
fams = [ui.FAMILY_SHORT[f] for f in CFG["generation"]["families"]]
for col, (title, kind, text) in zip(st.columns(3, gap="small"), [
        ("Synthetic graphs", "Training and testing",
         f"{n_train:,} generated graphs from five families: {', '.join(fams).lower()}. Timed "
         "once, offline, on one machine."),
        ("SNAP networks", "Testing only",
         f"{len(real)} real networks the model never saw: {', '.join(real)}."),
        ("OpenStreetMap", "Testing only",
         "The road network of Piedmont, California, read once with OSMnx and stored as an "
         "edge list with coordinates.")]):
    with col.container(key=f"panel_data_{kind[:4]}_{title[:3]}", height="stretch"):
        ui.md(f"<p class='exh'>{ui.esc(title)}</p><span class='pill'>{ui.esc(kind)}</span>"
              f"<p class='sub' style='margin:8px 0 0'>{ui.esc(text)}</p>")
b = CFG["benchmark"]
ui.md("<p class='sub' style='margin:14px 0 0'>Graphs are split into training and test sets by "
      "graph, never by row, so the model is always tested on graphs it has not seen. Selector: "
      f"decision tree on quick features. Random seed {CFG['seed']}. Timing: median of "
      f"{b['repeats']} runs.</p>")
