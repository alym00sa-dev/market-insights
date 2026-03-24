"""
Market Intelligence Agent — Streamlit entry point.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Market Intelligence Agent",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/1_agent_chat.py", title="Market Intelligence Agent", icon="💬"),
    st.Page("pages/2_dashboard.py", title="Dashboard", icon="📊"),
])
pg.run()
