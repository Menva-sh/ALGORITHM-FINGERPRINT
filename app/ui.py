"""Shared look and data for the demo pages: theme-aware CSS, colours, cached resources."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.config import PROJECT_ROOT, load_config
from src.demo_logic import NAMES, train_selector
from src.generators import InstanceSpec, build_instance, sample_queries
from src.graph import Graph
from src.real_graphs import parse_edge_list, read_edge_list

CFG = load_config()
C = CFG["algorithms"]["candidates"]
STATIC = Path(__file__).resolve().parent / "static"
# Per-theme accents (custom HTML cannot read Streamlit's theme, so the CSS is chosen per run).
PALETTE = {"light": {"accent": "#0f766e", "good": "#15803d", "bad": "#b42318", "warn": "#92400e"},
           "dark": {"accent": "#2bb3a3", "good": "#4ade80", "bad": "#f87171", "warn": "#fbbf24"}}
COLORS = {"dijkstra": "#3b74c4", "bidirectional_dijkstra": "#e07a3a", "astar": "#2e9a62",
          "bfs": "#d4a12f", "bidirectional_bfs": "#c95f88"}
SHORT = {**NAMES, "bidirectional_dijkstra": "Bi-Dijkstra", "bidirectional_bfs": "Bi-BFS"}
FAMILY_NAMES = {"erdos_renyi": "Random (Erdős–Rényi)", "barabasi_albert": "Scale-free (Barabási–Albert)",
                "watts_strogatz": "Small-world (Watts–Strogatz)", "grid_2d": "Grid",
                "random_geometric": "Geometric (random points)"}
FAMILY_SHORT = {"erdos_renyi": "Random", "barabasi_albert": "Scale-free",
                "watts_strogatz": "Small-world", "grid_2d": "Grid", "random_geometric": "Geometric"}
WEIGHT_NAMES = {"unit": "Unit", "uniform": "Uniform", "heavy_tailed": "Heavy-tailed",
                "euclidean": "Distance"}

CSS = """
<style>
:root { --accent: ACCENT; --good: GOOD; --bad: BAD; --warn: WARN;
  --line: rgba(128,128,128,.28); --soft: rgba(128,128,128,.09); --track: rgba(128,128,128,.18); }
.block-container { max-width: 1200px; padding-top: 4.2rem; padding-bottom: 3rem; }
h1 { letter-spacing: -0.02em; }
.accent { color: var(--accent); }
.lead { font-size: 1.02rem; opacity: .78; max-width: 780px; line-height: 1.55; margin-bottom: .4rem; }
.muted { opacity: .7; } .small { font-size: .88rem; }
.stVerticalBlock[class*="st-key-card"] { border: 1px solid var(--line); border-radius: 10px;
  padding: 18px 20px; }
.stVerticalBlock[class*="st-key-rec"] { border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 10px; padding: 18px 22px; background: var(--soft); }
.eyebrow { font-size: .74rem; letter-spacing: .08em; font-weight: 700; opacity: .65; text-transform: uppercase; }
.big-algo { font-size: 1.9rem; font-weight: 700; display: flex; align-items: center; gap: .55rem;
  margin: .15rem 0 .35rem; letter-spacing: -0.01em; }
.dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex: none; }
.tag { display: inline-block; padding: .12rem .5rem; border-radius: 4px; font-weight: 600;
  font-size: .78rem; border: 1px solid var(--line); }
.tag.good { color: var(--good); } .tag.warn { color: var(--warn); }
.bar { height: 8px; border-radius: 4px; background: var(--track); overflow: hidden; }
.bar.thick { height: 16px; }
.bar > span { display: block; height: 100%; border-radius: inherit; }
.why-row { margin: 8px 0 12px; } .why-row div:first-child { margin-bottom: 5px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; margin-top: 8px; }
.tile { background: var(--soft); border-radius: 8px; padding: 10px 12px; }
.tile .k { font-size: .8rem; opacity: .7; } .tile .v { font-size: 1.3rem; font-weight: 700; }
.stat { border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; height: 100%; }
.stat .k { font-weight: 600; font-size: .86rem; opacity: .75; }
.stat .v { font-size: 1.6rem; font-weight: 700; margin: .15rem 0; }
.stat .v.good { color: var(--good); } .stat .v.bad { color: var(--bad); } .stat .v.teal { color: var(--accent); }
.race-row { display: grid; grid-template-columns: minmax(70px, 170px) 1fr auto; gap: 12px;
  align-items: center; margin: 8px 0; }
.race-row .t { text-align: right; font-variant-numeric: tabular-nums; min-width: 64px; }
.callout { border-radius: 10px; padding: 14px 18px; border: 1px solid var(--line); }
.callout.warn { border-left: 4px solid var(--warn); } .callout.warn b { color: var(--warn); }
.callout.good { border-left: 4px solid var(--good); } .callout.good b { color: var(--good); }
.wins { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-bottom: 6px; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
button[data-variant="segmented_control"][aria-checked="true"] { background: var(--accent) !important;
  color: #fff !important; border-color: var(--accent) !important; }
.step { display: flex; align-items: center; gap: 12px; font-size: 1.25rem; font-weight: 700;
  margin: 26px 0 8px; }
.step .num { width: 26px; height: 26px; border-radius: 50%; background: var(--accent); color: #fff;
  font-size: .9rem; display: inline-flex; align-items: center; justify-content: center; flex: none; }
.fit { font-size: .92rem; line-height: 1.6; } .fit b { font-weight: 600; }
.stTextArea textarea { font-family: ui-monospace, Consolas, "Courier New", monospace; line-height: 1.25; }
.footer { border-top: 1px solid var(--line); margin-top: 26px; padding-top: 12px; opacity: .7; font-size: .86rem; }
@media (max-width: 640px) { .block-container { padding-left: 1rem; padding-right: 1rem; }
  .big-algo { font-size: 1.5rem; } .stat .v { font-size: 1.35rem; } }
</style>
"""


def theme() -> str:
    """'light' or 'dark' (the viewer's current Streamlit theme).

    On a session's first run Streamlit reports the system preference rather than the
    theme actually shown; the app opens in light, so light is assumed until a rerun.
    """
    if not st.session_state.get("_theme_known"):
        st.session_state["_theme_known"] = True
        return "light"
    t = getattr(getattr(st.context, "theme", None), "type", None)
    return t if t in PALETTE else "light"


def inject_css() -> None:
    """Theme CSS and logo; decides the theme once per run (see `theme`)."""
    mode = st.session_state["_theme_now"] = theme()
    p = PALETTE[mode]
    css = (CSS.replace("ACCENT", p["accent"]).replace("GOOD", p["good"])
           .replace("BAD", p["bad"]).replace("WARN", p["warn"]))
    st.markdown(css, unsafe_allow_html=True)
    st.logo(str(STATIC / f"logo_{mode}.svg"), size="large")


def accent() -> str:
    return PALETTE[st.session_state.get("_theme_now", "light")]["accent"]


def esc(text: Any) -> str:
    return html.escape(str(text))


def dot(algo: str, size: int = 16) -> str:
    return (f"<span class='dot' style='background:{COLORS[algo]};width:{size}px;"
            f"height:{size}px'></span>")


def bar(frac: float, color: str, thick: bool = False) -> str:
    frac = max(0.0, min(1.0, frac))
    return (f"<div class='bar{' thick' if thick else ''}'><span style='width:{frac * 100:.1f}%;"
            f"background:{color}'></span></div>")


def stat(key: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f"<div class='stat'><div class='k'>{esc(key)}</div><div class='v {cls}'>{value}</div>"
            f"<div class='muted'>{esc(sub)}</div></div>")


def race_rows(runs: pd.DataFrame, names: dict[str, str] = NAMES) -> str:
    """Bar rows: one per algorithm, fastest first, scaled to the slowest."""
    slowest = runs["median_us"].max()
    rows = []
    for i, r in runs.iterrows():
        tag = " <span class='tag good'>Fastest</span>" if i == 0 else ""
        label = f"<b>{esc(names[r['algorithm']])}</b>{tag}" if i == 0 else esc(names[r["algorithm"]])
        rows.append(f"<div class='race-row'><div>{label}</div>"
                    f"{bar(r['median_us'] / slowest, COLORS[r['algorithm']], thick=True)}"
                    f"<div class='t'>{r['median_us']:,.1f} µs</div></div>")
    return "".join(rows)


def us(ns: float) -> str:
    """Microseconds, compact."""
    v = ns / 1e3
    return f"{v:,.0f} µs" if v >= 100 else f"{v:,.1f} µs"


# ------------------------------------------------------------------ cached resources

@st.cache_resource(show_spinner="Training the selector on the benchmark dataset...")
def selector():
    """(plain-tree selector on Tier-0 features, best fixed algorithm on the whole dataset)."""
    df = pd.read_csv(PROJECT_ROOT / CFG["paths"]["dataset"])
    cols = [f"time_{a}" for a in ["dijkstra", "bidirectional_dijkstra"]]
    return train_selector(df, CFG), df[cols].sum().idxmin().removeprefix("time_")


@st.cache_resource(show_spinner="Building graph...")
def generated_graph(family: str, n: int, params: tuple, weights: str, rep: int) -> Graph:
    return build_instance(InstanceSpec(family, n, params, weights, rep), CFG).graph


@st.cache_resource(show_spinner="Reading edge list...")
def uploaded_graph(edges_text: str, coords_text: str | None) -> Graph:
    return parse_edge_list(edges_text, coords_text)[0]


@st.cache_resource(show_spinner="Loading road network...")
def road_graph() -> Graph | None:
    edges = PROJECT_ROOT / CFG["paths"]["real_graphs"] / "road_piedmont.edges"
    if not edges.exists():
        return None
    return read_edge_list(edges, edges.with_suffix(".coords"))[0]


@st.cache_data(show_spinner=False)
def algorithm_map() -> pd.DataFrame | None:
    path = PROJECT_ROOT / CFG["paths"]["results"] / "algorithm_map.csv"
    return pd.read_csv(path) if path.exists() else None


def random_query(g: Graph, key: str) -> tuple[int, int] | None:
    """Current random (s, t) for this widget key; `new_query(key)` draws another."""
    seed = st.session_state.get(f"{key}_seed", 0)
    pairs = sample_queries(g, 1, np.random.default_rng(seed))
    return pairs[0] if pairs else None


def new_query(key: str) -> None:
    st.session_state[f"{key}_seed"] = st.session_state.get(f"{key}_seed", 0) + 1


def analyse(graph_key: Any, g: Graph, s: int, t: int, race: int = 0) -> dict[str, Any]:
    """Fingerprint, recommendation and timed race for one query (kept per session)."""
    from src.demo_logic import recommend, run_all
    from src.fingerprint import extract

    cache = st.session_state.setdefault("analysis", {})
    key = (graph_key, s, t, race)
    if key not in cache:
        model, best_fixed = selector()
        with st.spinner("Fingerprinting and racing the algorithms..."):
            fp = extract(g, tiers=[0, 1, 2], cfg=CFG)
            rec = recommend(model, fp.features, g, CFG)
            runs = run_all(g, s, t, CFG)
        t_of = runs.set_index("algorithm")["median_us"]
        overhead_us = (fp.tier_cost_ns[0] + rec["inference_ns"]) / 1e3
        cache.clear()  # keep only the latest query per session
        cache[key] = {"fp": fp, "rec": rec, "runs": runs, "best_fixed": best_fixed,
                      "chosen_us": float(t_of[rec["choice"]]), "fixed_us": float(t_of[best_fixed]),
                      "overhead_us": overhead_us}
    return cache[key]


# ------------------------------------------------------------------ user inputs

@st.cache_resource(show_spinner="Reading the file...")
def parsed_upload(name: str, data: bytes, coords_text: str | None):
    from src.inputs import parse_file
    return parse_file(name, data, coords_text)


@st.cache_resource(show_spinner="Reading the maze...")
def parsed_maze(text: str):
    from src.inputs import parse_maze
    return parse_maze(text)


def step(number: int, title: str, sub: str = "") -> None:
    """Numbered section heading."""
    st.markdown(f"<div class='step'><span class='num'>{number}</span><span>{esc(title)}</span></div>"
                + (f"<div class='muted small' style='margin:-4px 0 10px 38px'>{esc(sub)}</div>"
                   if sub else ""), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def headline() -> dict[str, float] | None:
    """Held-out accuracy and net saving of the deployed selector, from results/."""
    path = PROJECT_ROOT / CFG["paths"]["results"] / "model_results.csv"
    if not path.exists():
        return None
    row = pd.read_csv(path, index_col=0).loc["plain_tree"]
    return {"accuracy": float(row["accuracy"]), "saved": float(row["saved_vs_best_fixed"])}
