"""Shared look and data for the demo pages: theme CSS, header, colours, cached resources."""
from __future__ import annotations

import html
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
TEAL = "#1f6f68"
COLORS = {"dijkstra": "#2f6bb3", "bidirectional_dijkstra": "#dd7230", "astar": "#2e8b57",
          "bfs": "#cf9a2e", "bidirectional_bfs": "#c4547a"}
SHORT = {**NAMES, "bidirectional_dijkstra": "Bi-Dijkstra", "bidirectional_bfs": "Bi-BFS"}
FAMILY_NAMES = {"erdos_renyi": "Random (Erdős–Rényi)", "barabasi_albert": "Scale-free (Barabási–Albert)",
                "watts_strogatz": "Small-world (Watts–Strogatz)", "grid_2d": "Grid",
                "random_geometric": "Geometric (random points)"}
FAMILY_SHORT = {"erdos_renyi": "Random", "barabasi_albert": "Scale-free",
                "watts_strogatz": "Small-world", "grid_2d": "Grid", "random_geometric": "Geometric"}
WEIGHT_NAMES = {"unit": "Unit", "uniform": "Uniform", "heavy_tailed": "Heavy-tailed",
                "euclidean": "Distance"}

LOGO = (f"<svg width='34' height='34' viewBox='0 0 34 34'><circle cx='17' cy='17' r='17' "
        f"fill='{TEAL}'/><path d='M9 22a8 8 0 0 1 16 0' stroke='white' stroke-width='2.4' "
        f"fill='none' stroke-linecap='round'/><path d='M13 22a4 4 0 0 1 8 0' stroke='white' "
        f"stroke-width='2.4' fill='none' stroke-linecap='round'/></svg>")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=Fraunces:opsz,wght@9..144,600&display=swap');
html, body, .stApp, .stMarkdown, p, label, li, input, button { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .serif { font-family: 'Fraunces', Georgia, serif !important; color: #1d2a2a;
  font-weight: 600 !important; letter-spacing: -0.01em; }
h1 { font-size: 2.7rem !important; line-height: 1.15 !important; padding-bottom: .3rem !important; }
h2 { font-size: 1.9rem !important; } h3 { font-size: 1.45rem !important; }
.stApp { background: #f8f4ee; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.4rem; }
.accent { color: #1f6f68; }
.lead { font-size: 1.12rem; color: #4b5557; max-width: 760px; line-height: 1.55; }
.muted { color: #5d6466; } .small { font-size: .9rem; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div) { border-radius: 16px; }
.stVerticalBlock[class*="st-key-card"] { background: #fff; border: 1px solid #ece5da;
  border-radius: 16px; padding: 22px 26px; }
.stVerticalBlock[class*="st-key-rec"] { background: #e6f0ec; border: 1px solid #cfe1da;
  border-radius: 16px; padding: 24px 30px; }
.eyebrow { font-size: .78rem; letter-spacing: .12em; font-weight: 600; color: #1f6f68; }
.big-algo { font-family: 'Fraunces', Georgia, serif; font-size: 2.5rem; font-weight: 600;
  color: #1d2a2a; display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .4rem; }
.dot { width: 16px; height: 16px; border-radius: 50%; display: inline-block; flex: none; }
.pill { display: inline-block; padding: .25rem .75rem; border-radius: 999px; font-weight: 600;
  font-size: .85rem; background: #d8ecdf; color: #1d5e3c; }
.pill.warn { background: #f8e9c8; color: #7a5410; }
.pill.tiny { font-size: .72rem; padding: .1rem .5rem; }
.bar { height: 12px; border-radius: 999px; background: #efe8dc; overflow: hidden; }
.bar.thick { height: 22px; border-radius: 8px; }
.bar > span { display: block; height: 100%; border-radius: inherit; }
.why-row { margin: 10px 0 14px; } .why-row div:first-child { margin-bottom: 6px; color: #1d2a2a; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 10px; }
.tile { background: #f6f0e6; border-radius: 12px; padding: 12px 14px; }
.tile .k { font-size: .82rem; color: #5d6466; } .tile .v { font-size: 1.55rem; font-weight: 600; color: #1d2a2a; }
.stat { background: #fff; border: 1px solid #ece5da; border-radius: 16px; padding: 18px 22px; height: 100%; }
.stat .k { font-weight: 600; color: #4b5557; font-size: .92rem; }
.stat .v { font-size: 2rem; font-weight: 700; color: #1d2a2a; margin: .2rem 0; }
.stat .v.good { color: #1d6b3d; } .stat .v.bad { color: #a33b2b; } .stat .v.teal { color: #1f5e58; }
.race-row { display: grid; grid-template-columns: 190px 1fr 90px; gap: 14px; align-items: center;
  margin: 10px 0; }
.race-row .t { text-align: right; font-weight: 500; }
.callout { border-radius: 16px; padding: 16px 20px; }
.callout.warn { background: #fdf3dc; border: 1px solid #f0d9a4; }
.callout.warn b { color: #6b4a0e; }
.callout.good { background: #e8f3ec; border: 1px solid #c8e2d1; }
.callout.good b { color: #1d5e3c; }
.wins { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.wins .stat:nth-child(5) { grid-column: 1 / -1; }
.legend { display: flex; flex-wrap: wrap; gap: 22px; margin-bottom: 6px; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.brand { display: flex; align-items: center; gap: 12px; font-family: 'Fraunces', Georgia, serif;
  font-size: 1.45rem; font-weight: 600; color: #1d2a2a; }
.nav-active { background: #e1efea; color: #1f6f68; padding: .5rem 1rem; border-radius: 999px;
  font-weight: 600; text-align: center; }
.stButton button[kind="primary"] { background: #1f6f68; border-color: #1f6f68; border-radius: 10px; }
.stButton button[kind="secondary"] { border-radius: 10px; border-color: #1f6f68; color: #1f6f68; }
.footer { border-top: 1px solid #e3dccf; margin-top: 28px; padding-top: 14px; color: #5d6466; font-size: .9rem; }
@media (max-width: 760px) { .tiles { grid-template-columns: repeat(2, 1fr); }
  .wins { grid-template-columns: repeat(2, 1fr); } .race-row { grid-template-columns: 110px 1fr 70px; } }
</style>
"""
CSS = CSS.replace("</style>", """
button[data-variant="segmented_control"][aria-checked="true"] { background: #1f6f68 !important;
  color: #fff !important; border-color: #1f6f68 !important; }
[data-baseweb="select"] > div { background: #fff !important; }
</style>""")


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(pages: list[Any], current: Any) -> None:
    """Brand on the left, page links on the right (the current page as a highlighted pill)."""
    cols = st.columns([4.2, 1.2, 1.3, 1.3], vertical_alignment="center")
    cols[0].markdown(f"<div class='brand'>{LOGO}Algorithm Fingerprint</div>", unsafe_allow_html=True)
    for col, page in zip(cols[1:], pages):
        if page.url_path == current.url_path:
            col.markdown(f"<div class='nav-active'>{page.title}</div>", unsafe_allow_html=True)
        else:
            col.page_link(page, label=page.title)
    st.markdown("<div style='border-bottom:1px solid #e3dccf;margin:4px 0 18px'></div>",
                unsafe_allow_html=True)


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


def race_rows(runs: pd.DataFrame, names: dict[str, str] = NAMES, label_px: int = 190) -> str:
    """Bar rows: one per algorithm, fastest first, scaled to the slowest."""
    slowest = runs["median_us"].max()
    grid = f"grid-template-columns:{label_px}px 1fr {80 if label_px < 150 else 90}px"
    rows = []
    for i, r in runs.iterrows():
        tag = " <span class='pill tiny'>Fastest</span>" if i == 0 else ""
        label = f"<b>{esc(names[r['algorithm']])}</b>{tag}" if i == 0 else esc(names[r["algorithm"]])
        rows.append(f"<div class='race-row' style='{grid}'><div>{label}</div>"
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
