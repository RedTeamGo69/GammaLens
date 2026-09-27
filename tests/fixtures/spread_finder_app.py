"""The REAL Spread Finder tab rendered against deterministic fakes.

No provider, database or network is touched: every cached loader the tab
calls is replaced with synthetic data and a real (in-memory) HAR fit.
GL_SF_MODE selects the scenario: ready | missing | spike.
"""
from datetime import datetime
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import phase1.market_clock as market_clock
import ui_spread_finder as usf
from range_finder.recommendations import fit_specs
from range_finder.tests.spread_finder_fixtures import spread_finder_features

MODE = os.environ.get("GL_SF_MODE", "ready")
NOW = datetime(2026, 6, 17, 11, tzinfo=ZoneInfo("America/New_York"))   # Wed
FEATURES = spread_finder_features(
    week="2026-06-08" if MODE == "missing" else "2026-06-15",
    vix=15.0,
)
_FIT = fit_specs(FEATURES, specs=["M1_baseline"]).fits["M1_baseline"]
PAYLOAD = {"result": _FIT.result, "feature_cols": _FIT.feature_cols,
           "metrics": _FIT.metrics}
LEVELS = {"zero_gamma": 5990.0, "call_wall": 6100.0, "put_wall": 5900.0}
DATA = SimpleNamespace(chain_cache={}, avail=["2026-06-18"], market_open=True)


class _Conn:
    def cursor(self):
        raise AssertionError("fixture: the tab reached the database directly")


with (
    patch.object(market_clock, "now_ny", lambda: NOW),
    patch.object(usf, "_get_rf_conn", lambda: _Conn()),
    patch.object(usf, "_live_vol_close", lambda proxy: 31.0 if MODE == "spike" else 15.5),
    patch.object(usf, "_cached_rf_get_features", lambda conn, ticker="SPX": FEATURES),
    patch.object(usf, "_cached_rf_load_model", lambda model, ticker: PAYLOAD),
    patch.object(usf, "_cached_weekly_setup", lambda conn, week, ticker: (6000.0, 15.2)),
    patch.object(usf, "_cached_prior_week_close", lambda week, ticker: 5990.0),
    patch.object(usf, "_cached_side_share_q", lambda ticker: None),
    patch.object(usf, "_cached_pi_coverage", lambda ticker: {"n": 0}),
    patch.object(usf, "_cached_nonactive_week_bands", lambda *a: []),
):
    usf._render_spread_finder_tab(
        6005.0, LEVELS, {"regime": "positive"}, DATA, ticker="SPX", weekly_em={},
    )
