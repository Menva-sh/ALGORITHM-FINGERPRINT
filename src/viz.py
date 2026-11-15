"""Small SVG pictures of a graph for the demo, built without Streamlit so they can be tested.

Graphs with coordinates are drawn where they are; others get a fixed-seed spring layout.
"""
from __future__ import annotations

import html
import math

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


# ------------------------------------------------------------------ readable view around the path

CAP = 45  # nodes drawn in the "around the path" view


def neighbourhood(g: Graph, path: list[int], cap: int = CAP) -> list[int]:
    """The path's nodes, then their nearest neighbours layer by layer, up to about `cap` nodes.

    Neighbours are added round-robin across the frontier, so no single hub fills the picture.
    """
    from collections import deque

    cap = max(cap, len(path) + 12)
    keep = list(dict.fromkeys(path))
    seen = set(keep)
    frontier = list(keep)
    while frontier and len(keep) < cap:
        queues = [deque(sorted(v for v, _ in g.adj[u] if v not in seen)) for u in frontier]
        added: list[int] = []
        while len(keep) < cap and any(queues):
            for q in queues:
                while q and q[0] in seen:
                    q.popleft()
                if q and len(keep) < cap:
                    v = q.popleft()
                    seen.add(v)
                    keep.append(v)
                    added.append(v)
        frontier = added
    return keep


def _unit(raw: list[Point], keep_aspect: bool) -> list[Point]:
    """Scale points to the unit square; one scale for both axes when shapes must stay true."""
    xs, ys = [p[0] for p in raw], [p[1] for p in raw]
    x0, y0 = min(xs), min(ys)
    sx, sy = (max(xs) - x0) or 1.0, (max(ys) - y0) or 1.0
    if keep_aspect:
        span = max(sx, sy)
        return [((x - x0) / span + (1 - sx / span) / 2, (y - y0) / span + (1 - sy / span) / 2)
                for x, y in raw]
    return [((x - x0) / sx, (y - y0) / sy) for x, y in raw]


def _num(v: float) -> str:
    s = f"{v:.3g}"
    return f"{v:,.0f}" if "e" in s else s


def _coord(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}"


def path_view(g: Graph, path: list[int], labels: list[str] | None = None, cap: int = CAP,
              seed: int = 42, w: int = 760, h: int = 420, pad: int = 40) -> tuple[str, int, bool]:
    """Circles for nodes, lines for edges, distances on edges, the chosen path highlighted.

    Draws the path and its nearest neighbours (about `cap` nodes) so every node and number
    stays readable. Graphs with coordinates keep their real positions; others are laid out
    with the path as a fixed line from left to right. Long node ids (like OpenStreetMap's)
    are replaced by short numbers in the circles; hover a node for its real id.
    Returns (svg, nodes drawn, whether nodes were numbered).
    """
    nodes = neighbourhood(g, path, cap)
    if g.coords is not None:  # real positions: leave out neighbours far from the path
        cx = sum(g.coords[v][0] for v in path) / len(path)
        cy = sum(g.coords[v][1] for v in path) / len(path)
        dist = {v: math.hypot(g.coords[v][0] - cx, g.coords[v][1] - cy) for v in nodes}
        ds = sorted(dist.values())
        q1, q3 = ds[len(ds) // 4], ds[(3 * len(ds)) // 4]
        limit = q3 + 1.5 * (q3 - q1)
        nodes = [v for v in nodes if v in set(path) or dist[v] <= limit]
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(u, v, wt) for u in nodes for v, wt in g.adj[u]
             if v in idx and (g.directed or u < v)]
    if g.coords is not None:  # real positions, same scale on both axes, filling the picture
        raw = [(float(g.coords[v][0]), float(g.coords[v][1])) for v in nodes]
        x0, x1 = min(p[0] for p in raw), max(p[0] for p in raw)
        y0, y1 = min(p[1] for p in raw), max(p[1] for p in raw)
        sx, sy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
        scale = min((w - 2 * pad) / sx, (h - 2 * pad) / sy)
        ox, oy = (w - sx * scale) / 2, (h - sy * scale) / 2
        px = [(ox + (x - x0) * scale, oy + (y1 - y) * scale) for x, y in raw]  # y up, as on a map
    else:
        import random

        import networkx as nx

        sub = nx.Graph()
        sub.add_nodes_from(range(len(nodes)))
        sub.add_edges_from((idx[u], idx[v]) for u, v, _ in edges)
        rng = random.Random(seed)
        init = {i: (rng.random(), rng.random()) for i in range(len(nodes))}
        fixed = []
        for k, v in enumerate(path):
            init[idx[v]] = (k / (len(path) - 1), 0.5)
            fixed.append(idx[v])
        pos = nx.spring_layout(sub, pos=init, fixed=fixed, seed=seed, iterations=120)
        unit = _unit([(float(pos[i][0]), float(pos[i][1])) for i in range(len(nodes))], False)
        px = [(pad + x * (w - 2 * pad), pad + (1 - y) * (h - 2 * pad)) for x, y in unit]

    on_path = set(path)
    path_edges = {frozenset(e) for e in zip(path, path[1:])}
    lengths = sorted(math.hypot(px[idx[u]][0] - px[idx[v]][0], px[idx[u]][1] - px[idx[v]][1])
                     for u, v, _ in edges)
    edge_px = lengths[len(lengths) // 2] if lengths else 60.0   # typical edge length on screen
    base_r = min(11.0, max(3.5, 0.34 * edge_px))               # circles shrink when crowded
    show_all_weights = len(edges) <= 70 and edge_px >= 44
    real = (lambda v: labels[v]) if labels is not None else (lambda v: str(v))
    numbered = max(len(real(v)) for v in nodes) > 4
    short = {v: str(i + 1) for i, v in enumerate(nodes)}
    name = (lambda v: short[v]) if numbered else real
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' role='img' "
           f"aria-label='{html.escape(f'Shortest path from {name(path[0])} to {name(path[-1])} '
                                       f'among {len(nodes)} nearby nodes')}' "
           f"style='width:100%;height:auto;display:block;background:{BG};border-radius:14px'>"]
    quiet = [(u, v, wt) for u, v, wt in edges if frozenset((u, v)) not in path_edges]
    hot = [(u, v, wt) for u, v, wt in edges if frozenset((u, v)) in path_edges]
    for group, color, width in ((quiet, EDGE, 1.6), (hot, PATH, 4.5)):
        for u, v, _ in group:
            (x1, y1), (x2, y2) = px[idx[u]], px[idx[v]]
            out.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                       f"stroke='{color}' stroke-width='{width}' stroke-linecap='round'/>")
    halo = "paint-order='stroke' stroke='#FFFFFF' stroke-width='3.5' stroke-linejoin='round'"
    if g.weighted:
        for u, v, wt in edges:
            hot_edge = frozenset((u, v)) in path_edges
            if not (show_all_weights or hot_edge):
                continue
            (x1, y1), (x2, y2) = px[idx[u]], px[idx[v]]
            out.append(f"<text x='{(x1 + x2) / 2:.1f}' y='{(y1 + y2) / 2 + 3.5:.1f}' "
                       f"text-anchor='middle' font-size='{11 if hot_edge else 9.5}' "
                       f"font-weight='{600 if hot_edge else 400}' "
                       f"fill='{PATH if hot_edge else '#5A5A55'}' {halo}>{_num(wt)}</text>")
    for v in nodes:
        x, y = px[idx[v]]
        label = name(v)
        r = base_r if len(label) <= 3 else min(24, max(base_r, 4.4 * len(label) + 4))
        show_label = base_r >= 8
        is_s, is_t, hot_node = v == path[0], v == path[-1], v in on_path
        fill = INK if is_s else "#FFFFFF"
        stroke = INK if (is_s or is_t) else (PATH if hot_node else "#8C8C85")
        out.append(f"<g><title>{html.escape(f'node {real(v)}')}"
                   + (f" ({_coord(g.coords[v][0])}, {_coord(g.coords[v][1])})"
                      if g.coords is not None else "") + "</title>"
                   f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='{fill}' stroke='{stroke}' "
                   f"stroke-width='{3 if (is_s or is_t) else 2.2 if hot_node else 1.4}'/>"
                   + (f"<text x='{x:.1f}' y='{y + 3.6:.1f}' text-anchor='middle' font-size='10.5' "
                      f"font-weight='500' fill='{'#FFFFFF' if is_s else INK}'>{html.escape(label)}</text>"
                      if show_label else "") + "</g>")
        if is_s or is_t:
            tag = "Start" if is_s else "End"
            sub_line = (f"<tspan x='{x:.1f}' dy='13' font-size='9.5' font-weight='400' fill='#5A5A55'>"
                        f"({_coord(g.coords[v][0])}, {_coord(g.coords[v][1])})</tspan>"
                        if g.coords is not None else "")
            out.append(f"<text x='{x:.1f}' y='{y + r + 14:.1f}' text-anchor='middle' font-size='11.5' "
                       f"font-weight='600' fill='{INK}' {halo}>{tag}{sub_line}</text>")
    out.append("</svg>")
    return "".join(out), len(nodes), numbered
