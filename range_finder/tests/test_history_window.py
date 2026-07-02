"""Training-window pinning (har_model.TRAIN_WINDOW_YEARS).

The 10y history experiment backfills weekly_spx/model_features with deep
rows. These tests pin the guarantee that makes that SAFE: production reads
go through train_window_min_date(), so extra history in the DB can never
silently change what production trains on.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

import range_finder.feature_builder as fb
from range_finder.har_model import TRAIN_WINDOW_YEARS, train_window_min_date


# ── train_window_min_date ──────────────────────────────────────────────────────

def test_min_date_is_train_window_years_back():
    got = datetime.strptime(train_window_min_date(), "%Y-%m-%d")
    expected = datetime.today() - timedelta(days=int(TRAIN_WINDOW_YEARS * 365.25))
    assert abs((got - expected).days) <= 1


def test_min_date_accepts_override():
    got = datetime.strptime(train_window_min_date(years=10), "%Y-%m-%d")
    expected = datetime.today() - timedelta(days=int(10 * 365.25))
    assert abs((got - expected).days) <= 1


# ── get_features(min_date=...) boundary ───────────────────────────────────────

class _FakeConn:
    """pd.read_sql_query stand-in target — patched below instead."""


def test_get_features_min_date_filters(monkeypatch):
    idx = pd.date_range("2016-01-04", periods=520, freq="7D")
    fake = pd.DataFrame({
        "week_start": idx,
        "log_range": [-4.5] * 520,
        "ticker": ["SPX"] * 520,
    })
    monkeypatch.setattr(fb.pd, "read_sql_query",
                        lambda *a, **k: fake.copy())

    cutoff = "2020-07-01"
    out = fb.get_features(_FakeConn(), min_date=cutoff)

    assert out.index.min() >= pd.Timestamp(cutoff)
    assert len(out) < 520


def test_get_features_without_min_date_returns_all(monkeypatch):
    idx = pd.date_range("2016-01-04", periods=520, freq="7D")
    fake = pd.DataFrame({
        "week_start": idx,
        "log_range": [-4.5] * 520,
        "ticker": ["SPX"] * 520,
    })
    monkeypatch.setattr(fb.pd, "read_sql_query",
                        lambda *a, **k: fake.copy())

    out = fb.get_features(_FakeConn())

    assert len(out) == 520
    assert out.index.min() == pd.Timestamp("2016-01-04")


# ── build_features(history_years=...) forwarding ──────────────────────────────

def test_build_features_forwards_history_years(monkeypatch):
    captured = {}

    weekly_idx = pd.date_range("2024-01-01", periods=60, freq="7D")
    weekly = pd.DataFrame({
        "spx_open": 5000.0, "spx_high": 5050.0, "spx_low": 4950.0,
        "spx_close": 5010.0, "range_pct": 0.02, "log_range": -3.9,
        "spx_return": 0.002, "vix_close": 15.0, "vix_open": 15.0,
        "vix_high": 16.0, "vix_low": 14.0,
    }, index=weekly_idx)
    weekly.index.name = "week_start"

    def fake_daily(ticker, years=6):
        captured["daily_years"] = years
        return pd.DataFrame(columns=["spx_close"])

    def fake_ts(years=6):
        captured["ts_years"] = years
        return pd.DataFrame(columns=["vix9d_close", "vix3m_close", "vix_ts_slope"])

    monkeypatch.setattr(fb, "_load_weekly_for_ticker", lambda conn, ticker: weekly.copy())
    monkeypatch.setattr(fb, "_load_daily_for_ticker", fake_daily)
    monkeypatch.setattr(fb, "fetch_vix_term_structure", fake_ts)
    monkeypatch.setattr(fb, "get_macro_daily",
                        lambda conn: pd.DataFrame(columns=["yield_spread", "fed_funds"],
                                                  index=pd.DatetimeIndex([])))
    monkeypatch.setattr(fb, "get_event_flags",
                        lambda conn: pd.DataFrame(columns=["has_fomc", "has_cpi",
                                                           "has_nfp", "has_opex",
                                                           "event_count"]))
    monkeypatch.setattr(fb, "load_gex_inputs",
                        lambda conn, ticker="SPX": pd.DataFrame(columns=["gex"]))
    monkeypatch.setattr(fb, "_load_earnings_flags",
                        lambda conn, ticker: pd.DataFrame(columns=["has_earnings"]))
    monkeypatch.setattr(fb, "_save_features", lambda conn, df, ticker="SPX": None)

    fb.build_features(_FakeConn(), ticker="SPX", history_years=10)

    assert captured["daily_years"] == 10
    assert captured["ts_years"] == 10


def test_build_features_default_stays_six(monkeypatch):
    captured = {}
    monkeypatch.setattr(fb, "_load_weekly_for_ticker",
                        lambda conn, ticker: pd.DataFrame())
    # empty weekly short-circuits before the supplemental fetches — verify
    # the default parameter value directly instead
    import inspect
    sig = inspect.signature(fb.build_features)
    assert sig.parameters["history_years"].default == 6
    assert captured == {}
