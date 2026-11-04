"""Streamlit demo: which shortest-path algorithm is fastest for this graph and query?

Run:  streamlit run app/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ui import header, inject_css  # noqa: E402

st.set_page_config(page_title="Algorithm Fingerprint", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
inject_css()

PAGES = [st.Page("views/try_graph.py", title="Try a graph", url_path="try", default=True),
         st.Page("views/algorithm_map.py", title="Algorithm map", url_path="map"),
         st.Page("views/road_network.py", title="Road network", url_path="road")]
current = st.navigation(PAGES, position="hidden")
header(PAGES, current)
current.run()
