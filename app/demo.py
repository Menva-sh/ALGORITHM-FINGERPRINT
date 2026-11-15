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

from app.ui import inject_css  # noqa: E402

st.set_page_config(page_title="Algorithm Fingerprint", page_icon="🧭", layout="wide",
                   initial_sidebar_state="auto")  # open on desktop, collapsed on phones
inject_css()

PAGES = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("views/try_graph.py", title="Try a graph", icon=":material/hub:"),
    st.Page("views/algorithm_map.py", title="Algorithm map", url_path="map",
            icon=":material/scatter_plot:"),
    st.Page("views/road_network.py", title="Road network", url_path="road",
            icon=":material/route:"),
    st.Page("views/results.py", title="Results", url_path="results",
            icon=":material/table_chart:"),
]
page = st.navigation(PAGES, position="top")
page.run()
