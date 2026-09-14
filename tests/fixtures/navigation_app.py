"""Real app navigation with isolated, deterministic market/study adapters.

Used by AppTest and local browser checks. No providers or databases are called.
Set GL_NAV_TEST_DELAY to exercise tab switches during slow market-data work.
"""
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import streamlit_app as app
import ui_forward_test
import ui_history
import ui_spread_finder
import ui_tv_export

NOW = datetime(2026, 9, 14, 11, tzinfo=ZoneInfo("America/New_York"))
LEVELS = {"zero_gamma": 6500, "call_wall": 6550, "put_wall": 6450}
DATA = SimpleNamespace(
    gex_df=pd.DataFrame({"strike": [6450, 6500, 6550]}), spot=6505,
    levels=LEVELS, regime_info={"regime": "positive"}, prev_close=6500,
    market_open=True, stats={}, target_exps=["2026-09-14"], dte0_calls=[],
    dte0_puts=[], dte0_exp="2026-09-14", calendar_snapshot={},
    confidence_info={}, staleness_info={}, chain_cache={},
    avail=["2026-09-14", "2026-09-18"], wall_cred={}, run_time="11:00 ET",
)


def market_data(*args, **kwargs):
    st.session_state["fixture_market_calls"] = st.session_state.get("fixture_market_calls", 0) + 1
    time.sleep(float(os.environ.get("GL_NAV_TEST_DELAY", "0")))
    return DATA


def export_band(*args, **kwargs):
    time.sleep(float(os.environ.get("GL_NAV_TEST_DELAY", "0")))
    return None


with (
    patch.object(app, "get_credentials", lambda: ("fixture", "")),
    patch.object(app, "now_ny", lambda: NOW),
    patch.object(app, "get_expirations_cached", lambda *a: DATA.avail),
    patch.object(app, "fetch_all_data", market_data),
    patch.object(app, "archive_computed_gamma_levels", lambda **k: None),
    patch.object(app, "build_expected_move_analysis", lambda **k: {"market_context": "live"}),
    patch.object(app, "TradierDataClient", lambda **k: SimpleNamespace(chain_cache={})),
    patch.object(app, "compute_em_for_expiration", lambda *a: {}),
    patch.object(app, "render_gex_html", lambda *a, **k: '<div id="fixture-gex">GEX chart fixture</div>'),
    patch.object(app, "render_key_levels", lambda *a, **k: ""),
    patch.object(app, "render_gex_stream", lambda *a, **k: ""),
    patch.object(app, "render_expected_move_panel", lambda *a, **k: ""),
    patch.object(app, "render_quality_card", lambda *a, **k: ""),
    patch.object(ui_history, "_apply_em_snapshot", lambda value, *a, **k: value),
    patch.object(ui_history, "_apply_typed_em_snapshot", lambda value, *a, **k: value),
    patch.object(ui_forward_test, "load_snapshot", lambda: ([], [], [{"study_id": "FIXTURE", "start_week": "2026-09-14"}])),
    patch.object(ui_tv_export, "_resolve_har_pi", export_band),
    patch.object(ui_spread_finder, "_render_spread_finder_tab", lambda *a, **k: st.title("Spread Finder fixture")),
):
    app.main()
