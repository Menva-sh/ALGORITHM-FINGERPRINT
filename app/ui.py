"""Shared look and data for the demo pages: CSS, algorithm glyphs, HTML chart pieces, cache."""
from __future__ import annotations

import html
import math
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
RESULTS = PROJECT_ROOT / CFG["paths"]["results"]

# Okabe-Ito colours plus a distinct shape per algorithm, so charts survive colour blindness.
COLORS = {"dijkstra": "#0072B2", "bidirectional_dijkstra": "#56B4E9", "bfs": "#E69F00",
          "bidirectional_bfs": "#D55E00", "astar": "#009E73"}
GLYPH = {"dijkstra": "dij", "bidirectional_dijkstra": "bdij", "bfs": "bfs",
         "bidirectional_bfs": "bbfs", "astar": "astar"}
SHAPES = {"dijkstra": "circle", "bidirectional_dijkstra": "square", "bfs": "triangle-up",
          "bidirectional_bfs": "diamond", "astar": "cross"}
CLIP = {"dij": "none", "bdij": "none", "bfs": "polygon(50% 0,100% 100%,0 100%)",
        "bbfs": "polygon(50% 0,100% 50%,50% 100%,0 50%)",
        "astar": "polygon(35% 0,65% 0,65% 35%,100% 35%,100% 65%,65% 65%,65% 100%,35% 100%,"
                 "35% 65%,0 65%,0 35%,35% 35%)"}
NEG = "#6B7587"  # bars that push away from the choice
FAMILY_NAMES = {"erdos_renyi": "Random (Erdős–Rényi)", "barabasi_albert": "Scale-free (Barabási–Albert)",
                "watts_strogatz": "Small-world (Watts–Strogatz)", "grid_2d": "Grid",
                "random_geometric": "Random geometric"}
FAMILY_SHORT = {"erdos_renyi": "Random", "barabasi_albert": "Scale-free",
                "watts_strogatz": "Small-world", "grid_2d": "Grid", "random_geometric": "Geometric"}
WEIGHT_NAMES = {"unit": "Unit", "uniform": "Uniform", "heavy_tailed": "Heavy-tailed",
                "euclidean": "Distance"}
FEATURE_NAMES = {"n_nodes": "Number of nodes", "n_edges": "Number of edges", "density": "Density",
                 "is_weighted": "Weighted edges", "has_coordinates": "Has coordinates"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
:root{--ink:#080808;--ink2:#444444;--mute:#666666;--line:#E8E8E8;--grid:#EDEDED;--axis:#C8C8C8;
  --panel:#FFFFFF;--side:#080808;--ok:#4E5A36;--bad:#B42318;--cream:#EAEAC6;--moss:#4E5A36;
  --ring:0 0 0 1px #E8E8E8;--mono:"JetBrains Mono",ui-monospace,Consolas,"Courier New",monospace;
  --c-dij:#0072B2;--c-bdij:#56B4E9;--c-bfs:#E69F00;--c-bbfs:#D55E00;--c-astar:#009E73}
.block-container{max-width:1240px;padding:4.4rem 28px 2.5rem}
[data-testid="stHeader"]{background:#F8F8F8;border-bottom:1px solid var(--line)}
h1{font-size:30px!important;line-height:1.15!important;font-weight:500!important;
  letter-spacing:-.03em!important;padding:0!important}
.lead{font-size:15px;color:var(--ink2);margin:6px 0 0;max-width:820px}
.mono{font-family:var(--mono);font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--mute)}
/* cards */
.panel,[class*="st-key-panel"]{background:var(--panel);border:0;border-radius:22px;box-shadow:var(--ring)}
[class*="st-key-panel"]{padding:4px 20px 16px;gap:12px}
.st-key-panel_input{padding-top:16px}
[class*="st-key-panel"] .stats{margin:0 -20px -16px}
.panel{display:flex;flex-direction:column;height:100%;overflow:hidden}
.ph{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 20px;
  border-bottom:1px solid var(--grid);min-height:48px}
.ph h2,[class*="st-key-ph"] h2{font-family:var(--mono);font-size:11.5px!important;font-weight:500;
  letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);margin:0;padding:0!important}
[class*="st-key-ph"]{margin:0 -20px;padding:6px 20px;border-bottom:1px solid var(--grid);min-height:48px}
[class*="st-key-ph"] [data-testid="stMarkdownContainer"] p{margin:0}
.pb{padding:16px 20px 20px;display:flex;flex-direction:column;gap:12px;flex:1}
.sub{font-size:12.5px;color:var(--mute)}
.ttl{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.cap{font-size:12.5px;color:var(--mute);border-top:1px solid var(--grid);padding-top:8px;margin:0}
.cap b{font-weight:500;color:var(--ink2)}
[class*="st-key-ph"] button[kind="tertiary"],[class*="st-key-ph"] [data-testid="stPopover"] button{
  font-size:12.5px;color:var(--ink2);text-decoration:underline;text-underline-offset:2px;
  padding:2px 6px;min-height:28px;border:0;background:none}
[class*="st-key-ph"] [data-testid="stPopover"] button svg{display:none}
[data-testid="stExpander"]{border-radius:16px;background:#fff}
/* glyphs */
.g{display:inline-block;width:11px;height:11px;flex:none}
.g.dij{background:var(--c-dij);border-radius:50%}.g.bdij{background:var(--c-bdij)}
.g.bfs{background:var(--c-bfs);clip-path:polygon(50% 0,100% 100%,0 100%)}
.g.bbfs{background:var(--c-bbfs);clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%)}
.g.astar{background:var(--c-astar);clip-path:polygon(35% 0,65% 0,65% 35%,100% 35%,100% 65%,65% 65%,65% 100%,35% 100%,35% 65%,0 65%,0 35%,35% 35%)}
.al{display:flex;align-items:center;gap:8px;font-size:13px;min-width:0}
.num{font-variant-numeric:tabular-nums;font-size:13px;font-weight:500}
.big{font-size:24px;line-height:1.15;font-weight:500;letter-spacing:-.02em}
.chip{font-size:12px;font-weight:500;padding:1px 9px;border:1px solid var(--ink2);border-radius:999px;
  color:var(--ink);white-space:nowrap}
.chip.warn{border-color:var(--bad);color:var(--bad)}
/* bar charts */
.prow{display:grid;grid-template-columns:150px minmax(0,1fr) 36px;align-items:center;gap:10px}
.track{height:8px;background:var(--grid);border-radius:4px}
.track>div{height:100%;border-radius:4px;min-width:3px}
.hb{position:relative;padding-bottom:22px}
.hb .gridl{position:absolute;right:0;top:0;bottom:22px}
.hb .gridl div{position:absolute;top:0;bottom:0;border-left:1px solid var(--grid)}
.hb .gridl div.zero{border-left-color:var(--axis)}
.hb .row{display:grid;gap:10px;align-items:center;height:30px;position:relative}
.hb .row .lbl{font-size:13px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hb .row .area{position:relative;height:100%}
.hb .bar{position:absolute;top:8px;height:14px}
.hb .val{position:absolute;top:4px;display:flex;align-items:center;gap:8px;white-space:nowrap}
.hb .ticks{position:absolute;right:0;bottom:0;height:18px;font-size:12px;color:var(--mute);
  font-variant-numeric:tabular-nums}
.hb .ticks span{position:absolute;transform:translateX(-50%)}
.hb .ticks span:last-child{transform:translateX(-100%)}
.wins{display:flex;flex-direction:column;gap:11px}
.wins .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.wins .track{height:10px;margin-top:5px}
/* tables and lists */
.tb{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
.tb th{font-family:var(--mono);font-weight:400;color:var(--mute);text-align:left;padding:6px 0;
  border-bottom:1px solid var(--line);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase}
.tb td{padding:6px 0;border-bottom:1px solid var(--grid)}
.tb .n,.bk .n{text-align:right}
.ctb{background:#fff;border-radius:16px;overflow:hidden;box-shadow:var(--ring);border-collapse:separate;
  border-spacing:0}
.ctb th{padding:10px 14px;background:#fff}.ctb td{padding:10px 14px;background:#fff}
.ctb th.cr,.ctb td.cr{background:var(--cream);font-weight:500}
.bk{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13.5px}
.bk thead th{border-top:2px solid var(--ink);border-bottom:1px solid var(--ink);padding:7px 10px;
  font-weight:500;text-align:left;font-size:13px;vertical-align:bottom}
.bk td{padding:6px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
[data-testid="stMarkdownContainer"]:has(>.bk),[data-testid="stMarkdownContainer"]:has(.bk){overflow-x:auto}
.bk .grp td{border-top:1px solid var(--line)}
.bk .orc td{border-top:1px solid var(--ink);border-bottom:2px solid var(--ink);font-style:italic;color:var(--ink2)}
.bk b{font-weight:500}
.dl{display:flex;flex-direction:column;margin:0}
.dl>div{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:7px 0;
  border-bottom:1px solid var(--grid)}
.dl>div:last-child{border-bottom:0}.dl dt{color:var(--ink2)}.dl dd{margin:0}
.verdict{display:flex;align-items:center;gap:6px;font-weight:500}
.verdict.ok{color:var(--ok)}.verdict.bad{color:var(--bad)}
.stats{display:flex;flex-wrap:wrap}
.stats>div{flex:1 1 150px;padding:14px 20px 16px;border-right:1px solid var(--grid)}
.stats>div:last-child{border-right:0}
.stats .k{font-size:12.5px;color:var(--mute)}.stats .v{font-size:26px;line-height:1.2;font-weight:500;letter-spacing:-.02em}
.stats.small .v{font-size:14px;line-height:1.45}
.slot{border:1px dashed #A9A9A9;border-radius:22px;padding:16px 20px;display:flex;flex-direction:column;
  gap:10px;background:rgba(255,255,255,.55);height:100%}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:8px}
/* widgets */
.stButton button,.stDownloadButton button,[data-testid="stPageLink"] a{border-radius:999px;font-weight:500}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:var(--ink);
  color:#F8F8F8;border-color:var(--ink)}
.stButton button[kind="secondary"],.stDownloadButton button[kind="secondary"]{background:#fff;
  border-color:var(--axis);color:var(--ink)}
[data-testid="stWidgetLabel"] p{font-size:12.5px;font-weight:500;color:var(--ink2)}
[data-testid="stFileUploaderDropzone"]{border-radius:16px;border:1.5px dashed var(--axis);background:#fff}
[role="radiogroup"]:has(>button[data-variant="segmented_control"]){background:var(--line);
  padding:3px;border-radius:999px;gap:2px}
button[data-variant="segmented_control"]{border:0!important;background:transparent;color:var(--ink2);
  font-weight:500;border-radius:999px!important}
button[data-variant="segmented_control"][data-selected="true"]{background:#fff!important;
  color:var(--ink)!important;box-shadow:var(--ring)}
button[data-variant="pills"]{background:#fff;font-weight:500;border-radius:999px!important;
  border-color:var(--axis)!important;color:var(--ink2)}
button[data-variant="pills"][data-selected="true"]{background:var(--ink)!important;color:#fff!important;
  border-color:var(--ink)!important}
.stTextArea textarea{font-family:var(--mono);line-height:1.25}
/* home */
.hero{background:radial-gradient(ellipse at 50% 0,#EAEAC6 0,rgba(234,234,198,0) 72%);border-radius:22px;
  padding:38px 24px 30px;text-align:center}
.tag{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--mute);background:#fff;box-shadow:inset 0 0 0 1px var(--line);
  border-radius:999px;padding:5px 12px}
.hero h1{font-size:46px!important;letter-spacing:-.035em!important;margin:14px 0 8px!important}
.hero h1 b{color:var(--moss);font-weight:500}
.hero .lead{margin:0 auto;font-size:16px;max-width:640px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.mc{background:#fff;border-radius:22px;box-shadow:var(--ring);padding:18px 20px}
.mc.cr{background:var(--cream);box-shadow:0 0 0 1px #D9D9A8}
.mc.bk{background:var(--ink);color:#F8F8F8;box-shadow:none}
.mc .k{font-size:12.5px;color:var(--mute)}.mc .v{font-size:32px;font-weight:500;letter-spacing:-.025em;line-height:1.2}
.mc .s{font-size:12.5px;color:var(--mute)}.mc.cr .k,.mc.cr .s{color:#55552E}
.mc.bk .k,.mc.bk .s{color:#BDBDBD}
.sec{font-size:22px;font-weight:500;letter-spacing:-.025em;color:#767676;margin:0;padding:32px 0 4px}
.st-key-home_cta a{background:var(--ink);color:#F8F8F8!important;justify-content:center;border-radius:999px;padding:6px 16px}
.st-key-home_cta a *{color:#F8F8F8!important}
.sec b{color:var(--ink);font-weight:500}
.exh{font-size:15px;font-weight:500;margin:2px 0 0}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.flow .st{background:#fff;border-radius:22px;box-shadow:var(--ring);padding:16px 18px}
.flow .st.hi{background:var(--cream);box-shadow:0 0 0 1px #D9D9A8}
.flow .st .n{font-family:var(--mono);font-size:11px;letter-spacing:.07em;color:var(--mute)}
.flow .st b{display:block;font-size:15px;font-weight:500;margin-top:2px}
.flow .st span.d{font-size:12.5px;color:var(--mute)}
.pill{display:inline-block;font-size:12px;font-weight:500;padding:1px 9px;border:1px solid var(--axis);
  border-radius:999px;color:var(--ink2);white-space:nowrap}
@media (max-width:900px){.cards,.flow{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.block-container{padding:1.4rem 16px 2rem}.prow{grid-template-columns:120px minmax(0,1fr) 36px}
  .hero h1{font-size:34px!important}}
</style>
"""


def inject_css() -> None:
    """Page CSS and the sidebar logo."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.logo(str(STATIC / "logo.svg"), size="large", icon_image=str(STATIC / "logo_icon.svg"))


def esc(text: Any) -> str:
    return html.escape(str(text))


def md(html_text: str) -> None:
    st.markdown(html_text, unsafe_allow_html=True)


# ------------------------------------------------------------------ HTML pieces

def glyph(algo: str, size: int = 11) -> str:
    """Coloured shape that identifies an algorithm."""
    return f"<span class='g {GLYPH[algo]}' style='width:{size}px;height:{size}px'></span>"


def algo_label(algo: str, size: int = 11) -> str:
    return f"<span class='al'>{glyph(algo, size)}{esc(NAMES[algo])}</span>"


def header(title: str, lead: str) -> None:
    md(f"<h1>{esc(title)}</h1><p class='lead'>{lead}</p>")


def panel(title: str, body: str, sub: str = "", min_height: int = 0) -> str:
    """A white panel with a title row; `body` is HTML."""
    style = f" style='min-height:{min_height}px'" if min_height else ""
    right = f"<span class='sub'>{esc(sub)}</span>" if sub else ""
    head = f"<div class='ph'><h2>{esc(title)}</h2>{right}</div>" if title else ""
    return f"<section class='panel'{style}>{head}<div class='pb'>{body}</div></section>"


def panel_head(key: str, title: str, sub: str = "") -> st.delta_generator.DeltaGenerator:
    """Title row of a widget panel; returns the row so links/buttons can be added to it."""
    row = st.container(key=f"ph_{key}", horizontal=True, vertical_alignment="center",
                       horizontal_alignment="distribute", gap="small")
    row.markdown(f"<div class='ttl'><h2>{esc(title)}</h2><span class='sub'>{esc(sub)}</span></div>",
                 unsafe_allow_html=True)
    return row.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                         gap="small", width="content")


def caption(label: str, text: str) -> str:
    return f"<p class='cap'><b>{esc(label)}</b> {text}</p>"


def nice_ticks(hi: float, lo: float = 0.0, n: int = 4) -> list[float]:
    """Round axis ticks covering [lo, hi]."""
    span = max(hi - lo, 1e-12)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    return [round(start + i * step, 10) for i in range(int(round((end - start) / step)) + 1)]


def fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    s = f"{abs(v):,.10g}" if abs(v) >= 1 else f"{abs(v):.10g}"
    return ("−" if v < 0 else "") + s


def hbars(rows: list[dict[str, Any]], label_w: int = 170, fmt: str = "{:,.1f}") -> str:
    """Horizontal bar chart with a value axis. Each row: label (HTML), value, color, extra (HTML).

    Negative values extend left of zero.
    """
    vals = [r["value"] for r in rows]
    ticks = nice_ticks(max(max(vals), 0.0), min(min(vals), 0.0))
    lo, hi = ticks[0], ticks[-1]
    pos = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731
    z = pos(0.0)
    left = label_w + 10
    grid = "".join(f"<div class='{'zero' if t == 0 else ''}' style='left:{pos(t):.3f}%'></div>"
                   for t in ticks)
    out = [f"<div class='hb'><div class='gridl' style='left:{left}px'>{grid}</div>"]
    for r in rows:
        v = r["value"]
        a, b = sorted((z, pos(v)))
        radius = "0 4px 4px 0" if v >= 0 else "4px 0 0 4px"
        txt = fmt.format(v).replace("-", "−")
        out.append(
            f"<div class='row' style='grid-template-columns:{label_w}px minmax(0,1fr)'>"
            f"<span class='lbl'>{r['label']}</span><div class='area'>"
            f"<div class='bar' style='left:{a:.3f}%;width:{max(b - a, 0.4):.3f}%;"
            f"background:{r['color']};border-radius:{radius}'></div>"
            f"<div class='val' style='left:calc({max(a, b) if v >= 0 else max(z, a):.3f}% + 8px)'>"
            f"<span class='num'>{txt}</span>{r.get('extra', '')}</div></div></div>")
    labels = "".join(f"<span style='left:{pos(t):.3f}%'>{fmt_tick(t)}</span>" for t in ticks)
    out.append(f"<div class='ticks' style='left:{left}px'>{labels}</div></div>")
    return "".join(out)


def prob_rows(probs: dict[str, float]) -> str:
    """Predicted probability per applicable algorithm, highest first."""
    rows = []
    for a, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        rows.append(f"<div class='prow'>{algo_label(a)}<div class='track'>"
                    f"<div style='width:{p * 100:.1f}%;background:{COLORS[a]}'></div></div>"
                    f"<span class='num'>{p:.2f}</span></div>")
    return "<div style='display:flex;flex-direction:column;gap:9px'>" + "".join(rows) + "</div>"


def win_rows(counts: pd.Series) -> str:
    """Count and share of graphs won per algorithm, bars scaled to the largest."""
    total, top = counts.sum(), counts.max()
    rows = [f"<div><div class='top'>{algo_label(a)}<span class='num'>{int(c):,} "
            f"({c / total:.1%})</span></div><div class='track'><div style='width:{c / top * 100:.1f}%;"
            f"background:{COLORS[a]}'></div></div></div>" for a, c in counts.items()]
    return "<div class='wins'>" + "".join(rows) + "</div>"


def dl(items: list[tuple[str, str]]) -> str:
    return "<dl class='dl'>" + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items) + "</dl>"


def verdict(ok: bool, text: str) -> str:
    icon = ("<svg width='16' height='16' viewBox='0 0 16 16' aria-hidden='true'><circle cx='8' cy='8' "
            f"r='8' fill='{'#4E5A36' if ok else '#B42318'}'/>"
            + ("<path d='M4.5 8.3l2.3 2.3 4.7-5' stroke='#fff' stroke-width='1.8' fill='none' "
               "stroke-linecap='round' stroke-linejoin='round'/>" if ok else
               "<path d='M5.2 5.2l5.6 5.6M10.8 5.2l-5.6 5.6' stroke='#fff' stroke-width='1.8' "
               "stroke-linecap='round'/>") + "</svg>")
    return f"<span class='verdict {'ok' if ok else 'bad'}'>{icon}{esc(text)}</span>"


def stats(items: list[tuple[str, str, str]], small: bool = False) -> str:
    return (f"<div class='stats{' small' if small else ''}'>" + "".join(
        f"<div><div class='k'>{esc(k)}</div><div class='v'>{v}</div>"
        + (f"<div class='k'>{esc(s)}</div>" if s else "") + "</div>" for k, v, s in items) + "</div>")


def us(v: float) -> str:
    """Microseconds, compact."""
    return f"{v:,.0f} µs" if v >= 100 else f"{v:,.1f} µs"


# ------------------------------------------------------------------ cached resources

@st.cache_data(show_spinner=False)
def training_graphs() -> int:
    path = PROJECT_ROOT / CFG["paths"]["dataset"]
    return int(pd.read_csv(path, usecols=["graph_id"])["graph_id"].nunique()) if path.exists() else 0


@st.cache_data(show_spinner=False)
def holdout_graphs() -> int:
    """Number of held-out test graphs under the configured group split."""
    from src.evaluate import split_by_graph
    path = PROJECT_ROOT / CFG["paths"]["dataset"]
    df = pd.read_csv(path, usecols=["graph_id"])
    _, test = split_by_graph(df, CFG["model"]["test_fraction"], CFG["seed"])
    return int(df.iloc[test]["graph_id"].nunique())


@st.cache_data(show_spinner=False)
def result(name: str) -> pd.DataFrame | None:
    """A CSV from results/, or None when it has not been generated."""
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


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


def algorithm_map() -> pd.DataFrame | None:
    return result("algorithm_map.csv")


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
