"""Offline browser preview using synthetic outcomes in the real app shell."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit_app
import ui_forward_test

rows = runpy.run_path(str(Path(__file__).with_name("forward_summary_rows.py")))["summary_rows"]()
st.session_state.setdefault("_tab_last", "forward")
st.caption("Preview · synthetic test data")
with (
    patch.object(streamlit_app, "get_credentials", lambda: ("", "")),
    patch.object(ui_forward_test, "load_snapshot", lambda: (rows, [], [])),
):
    streamlit_app.main()
