"""Turn user inputs into the shared Graph: files of several formats and text mazes.

Every input must describe the problem the selector was built for: a shortest path between
two points, with distances that are never negative. Anything else raises UnsupportedInput
with a message that says why.

Supported:
  edge list    .txt/.edges/.csv, one edge per line: "u v" or "u v weight" (spaces or commas)
  matrix       .csv/.txt, square table of distances (0 or empty = no road), optional header
  GraphML/GML  .graphml/.gml, weight from edge attribute "weight" or "length",
               coordinates from node attributes x/y
  JSON         .json in networkx node-link format
  maze         text grid: '#' wall, any other character open, one 'S' start and one 'T' target
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field

from src.graph import Graph

MAX_NODES = 60_000
MAX_EDGES = 400_000
FILE_TYPES = ["txt", "edges", "csv", "graphml", "gml", "json"]


class UnsupportedInput(ValueError):
    """The input cannot be turned into a valid shortest-path problem."""


@dataclass
class ParsedGraph:
    graph: Graph
    fmt: str
    labels: list[str]                     # original node name per node id
    notes: list[str] = field(default_factory=list)
    source: int | None = None             # fixed query (mazes)
    target: int | None = None


# ------------------------------------------------------------------ helpers

def _check_size(n: int, m: int) -> None:
    if n < 2:
        raise UnsupportedInput("The graph needs at least two nodes.")
    if m == 0:
        raise UnsupportedInput("The graph has no edges, so there is no path to find.")
    if n > MAX_NODES or m > MAX_EDGES:
        raise UnsupportedInput(f"Too large for this demo ({n:,} nodes, {m:,} edges; limit "
                               f"{MAX_NODES:,} nodes / {MAX_EDGES:,} edges).")


def _build(names: list[str], edges: list[tuple[int, int, float]], fmt: str,
           coords: list[tuple[float, float]] | None = None,
           notes: list[str] | None = None) -> ParsedGraph:
    """Validate weights, decide weighted/unweighted, build the Graph."""
    notes = list(notes or [])
    for _, _, w in edges:
        if math.isnan(w) or math.isinf(w):
            raise UnsupportedInput("Some distances are not numbers.")
        if w < 0:
            raise UnsupportedInput("Some distances are negative. These algorithms (and the "
                                   "selector) only handle distances of 0 or more; negative "
                                   "distances need Bellman-Ford instead.")
    weighted = any(w != 1.0 for _, _, w in edges)
    payload = edges if weighted else [(u, v) for u, v, _ in edges]
    g = Graph.from_edges(len(names), payload, directed=False, weighted=weighted, coords=coords)
    _check_size(g.n, g.m)
    return ParsedGraph(g, fmt, names, notes)


def _number(tok: str) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return None


# ------------------------------------------------------------------ text formats

def _rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%")):
            continue
        rows.append([t for t in line.replace(",", " ").replace(";", " ").split() if t])
    return rows


def _as_matrix(rows: list[list[str]]) -> tuple[list[str], list[list[float]]] | None:
    """(names, square matrix) if the rows form a distance matrix, else None."""
    if len(rows) < 2:
        return None
    header = None
    if any(_number(t) is None for t in rows[0]):
        header, rows = rows[0], rows[1:]
    n = len(rows)
    has_index = all(len(r) == n + 1 and _number(r[0]) is None for r in rows)
    body = [r[1:] if has_index else r for r in rows]
    if any(len(r) != n for r in body) or n < 2:
        return None
    vals = [[_number(t) for t in r] for r in body]
    if any(v is None for r in vals for v in r):
        return None
    names = ([r[0] for r in rows] if has_index else
             header[-n:] if header and len(header) >= n else [str(i) for i in range(n)])
    return names, vals  # type: ignore[return-value]


def parse_text(text: str, coords_text: str | None = None) -> ParsedGraph:
    """Edge list or distance matrix (detected automatically)."""
    rows = _rows(text)
    if not rows:
        raise UnsupportedInput("The file is empty.")
    matrix = _as_matrix(rows)
    if matrix is not None and len(matrix[0]) > 3:  # 2-3 node "matrices" read as edge lists
        names, vals = matrix
        n = len(names)
        edges, asym = [], False
        for i in range(n):
            for j in range(n):
                if i != j and vals[i][j] != vals[j][i]:
                    asym = True
                if i < j:
                    w = min(x for x in (vals[i][j], vals[j][i]) if x != 0) \
                        if (vals[i][j] or vals[j][i]) else 0.0
                    if w:
                        edges.append((i, j, w))
        notes = ["Matrix is not symmetric: each road is used in both directions with the "
                 "smaller distance."] if asym else []
        return _build(names, edges, "distance matrix", notes=notes)

    ids: dict[str, int] = {}
    edges = []
    for r in rows:
        if len(r) < 2:
            raise UnsupportedInput(f"Could not read the line '{' '.join(r)}': an edge needs "
                                   f"two node names (and optionally a distance).")
        w = 1.0
        if len(r) >= 3:
            w = _number(r[2])
            if w is None:
                raise UnsupportedInput(f"'{r[2]}' is not a number (third column = distance).")
        u, v = (ids.setdefault(t, len(ids)) for t in r[:2])
        edges.append((u, v, w))
    names = list(ids)
    coords = _coords(coords_text, ids) if coords_text else None
    return _build(names, edges, "edge list", coords)


def _coords(text: str, ids: dict[str, int]) -> list[tuple[float, float]]:
    xy: dict[int, tuple[float, float]] = {}
    for r in _rows(text):
        if len(r) >= 3 and r[0] in ids and _number(r[1]) is not None and _number(r[2]) is not None:
            xy[ids[r[0]]] = (float(r[1]), float(r[2]))
    if len(xy) != len(ids):
        raise UnsupportedInput(f"Coordinates are missing for {len(ids) - len(xy)} nodes.")
    return [xy[i] for i in range(len(ids))]


# ------------------------------------------------------------------ graph file formats

def _from_networkx(nxg, fmt: str) -> ParsedGraph:
    notes = []
    if nxg.is_directed():
        notes.append("The file describes one-way edges; here every edge is used in both "
                     "directions.")
        nxg = nxg.to_undirected()
    names = [str(v) for v in nxg.nodes]
    index = {v: i for i, v in enumerate(nxg.nodes)}
    edges = []
    for u, v, d in nxg.edges(data=True):
        w = d.get("weight", d.get("length", 1.0))
        num = _number(str(w))
        if num is None:
            raise UnsupportedInput(f"Edge weight '{w}' is not a number.")
        edges.append((index[u], index[v], num))
    coords = None
    data = [nxg.nodes[v] for v in nxg.nodes]
    if data and all("x" in d and "y" in d for d in data):
        coords = [(float(d["x"]), float(d["y"])) for d in data]
    elif data and all("pos" in d for d in data):
        coords = [tuple(map(float, d["pos"]))[:2] for d in data]
    return _build(names, edges, fmt, coords, notes)


def parse_file(name: str, data: bytes, coords_text: str | None = None) -> ParsedGraph:
    """Parse an uploaded file by its extension (text formats are auto-detected)."""
    import networkx as nx

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise UnsupportedInput("The file is not a text file.") from e
    try:
        if ext == "graphml":
            return _from_networkx(nx.read_graphml(io.BytesIO(data)), "GraphML")
        if ext == "gml":
            return _from_networkx(nx.parse_gml(text.splitlines(), label="label"), "GML")
        if ext == "json":
            return _from_networkx(nx.node_link_graph(json.loads(text), edges="links"), "JSON")
    except UnsupportedInput:
        raise
    except Exception as e:  # malformed file in a structured format
        raise UnsupportedInput(f"Could not read the {ext.upper()} file: {e}") from e
    return parse_text(text, coords_text)


# ------------------------------------------------------------------ maze

def parse_maze(text: str) -> ParsedGraph:
    """Grid maze: '#' walls, other characters open; 4-neighbour moves; S start, T target."""
    lines = [ln.rstrip("\n") for ln in text.strip("\n").splitlines()]
    cells: dict[tuple[int, int], int] = {}
    s = t = None
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#":
                continue
            cells[(r, c)] = len(cells)
            if ch in "Ss":
                if s is not None:
                    raise UnsupportedInput("The maze has more than one S.")
                s = (r, c)
            elif ch in "Tt":
                if t is not None:
                    raise UnsupportedInput("The maze has more than one T.")
                t = (r, c)
    if s is None or t is None:
        raise UnsupportedInput("Mark one start cell with S and one target cell with T.")
    edges = [(i, cells[(r + dr, c + dc)], 1.0) for (r, c), i in cells.items()
             for dr, dc in ((0, 1), (1, 0)) if (r + dr, c + dc) in cells]
    if not edges:
        raise UnsupportedInput("There is no way from S to T: the walls block every path.")
    names = [f"row {r + 1}, col {c + 1}" for (r, c) in cells]
    coords = [(float(c), float(-r)) for (r, c) in cells]
    parsed = _build(names, edges, "maze", coords)
    parsed.source, parsed.target = cells[s], cells[t]
    if parsed.graph.component_labels()[parsed.source] != parsed.graph.component_labels()[parsed.target]:
        raise UnsupportedInput("There is no way from S to T: the walls block every path.")
    return parsed


SAMPLE_MAZE = """#####################
#S    #       #     #
# ### # ##### # ### #
#   #   #   #   #   #
### ##### # ##### # #
#   #     #     # # #
# ### ######### # # #
#     #       #   #T#
#####################"""
