"""Small SVG pictures of a graph for the demo, built without Streamlit so they can be tested.

Graphs with coordinates are drawn where they are; others get a fixed-seed spring layout.
"""
from __future__ import annotations

import html

from src.graph import Graph

BG, EDGE, INK, PATH = "#F4F4F0", "#B9B9B2", "#080808", "#0072B2"
Point = tuple[float, float]


def layout(g: Graph, seed: int = 42) -> list[Point]:
    """Node positions scaled to the unit square (y up for coordinates, as on a map)."""
    if g.coords is not None:
        raw = [(float(x), float(y)) for x, y in g.coords]
    else:
        import networkx as nx

        nxg = nx.Graph()
        nxg.add_nodes_from(range(g.n))
        nxg.add_edges_from((u, v) for u, v, _ in g.edges())
        # spring layout is slow on big graphs; use fewer iterations, then a circle, above these sizes
        pos = (nx.spring_layout(nxg, seed=seed, iterations=50 if g.n <= 600 else 20)
               if g.n <= 3000 else nx.circular_layout(nxg))
        raw = [(float(pos[i][0]), float(pos[i][1])) for i in range(g.n)]
    xs, ys = [p[0] for p in raw], [p[1] for p in raw]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0) or 1.0  # one scale for both axes keeps shapes true
    cx, cy = (max(xs) - x0) / span / 2, (max(ys) - y0) / span / 2
    return [((x - x0) / span - cx + 0.5, (y - y0) / span - cy + 0.5) for x, y in raw]


def graph_svg(g: Graph, pos: list[Point], s: int | None = None, t: int | None = None,
              path: list[int] | None = None, w: int = 640, h: int = 380, pad: int = 18,
              label: str = "Graph") -> str:
    """The graph as an SVG string: edges, optional shortest path, start (filled), end (open)."""
    side = min(w - 2 * pad, h - 2 * pad)
    ox, oy = (w - side) / 2, (h - side) / 2
    p = [(ox + x * side, oy + (1 - y) * side) for x, y in pos]  # flip y: up is up
    edges = "".join(f"M{p[u][0]:.1f} {p[u][1]:.1f}L{p[v][0]:.1f} {p[v][1]:.1f}"
                    for u, v, _ in g.edges())
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' role='img' "
             f"aria-label='{html.escape(label)}' style='width:100%;height:auto;display:block;"
             f"background:{BG};border-radius:14px'>",
             f"<path d='{edges}' stroke='{EDGE}' stroke-width='1' fill='none' "
             "stroke-linecap='round'/>"]
    if g.n <= 80:
        parts += [f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='#FFFFFF' stroke='{EDGE}' "
                  "stroke-width='1.2'/>" for x, y in p]
    if path:
        line = " ".join(f"{p[i][0]:.1f},{p[i][1]:.1f}" for i in path)
        parts.append(f"<polyline points='{line}' fill='none' stroke='#FFFFFF' stroke-width='7' "
                     "stroke-linecap='round' stroke-linejoin='round'/>"
                     f"<polyline points='{line}' fill='none' stroke='{PATH}' stroke-width='3.5' "
                     "stroke-linecap='round' stroke-linejoin='round'/>")
    if s is not None:
        parts.append(f"<circle cx='{p[s][0]:.1f}' cy='{p[s][1]:.1f}' r='6.5' fill='{INK}' "
                     "stroke='#FFFFFF' stroke-width='2'/>")
    if t is not None:
        parts.append(f"<circle cx='{p[t][0]:.1f}' cy='{p[t][1]:.1f}' r='6.5' fill='#FFFFFF' "
                     f"stroke='{INK}' stroke-width='2.5'/>")
    parts.append("</svg>")
    return "".join(parts)
