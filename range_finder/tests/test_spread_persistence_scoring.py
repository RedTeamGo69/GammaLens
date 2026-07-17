"""Outcome scoring integrity (spread_persistence).

Two audit regressions pinned here:

  * _fetch_week_ohlc must NOT trust a weekly_spx row whose updated_at is
    inside the scored week — that row is the Monday-morning partial bar,
    and scoring against it records a tiny "range" for a week that may have
    breached a strike (permanently corrupting outcome + coverage stats).
  * "max_loss" (both strikes touched intraweek) books -width, never
    -2*width: a cash-settled European condor expires at ONE index print, so
    at most one side can finish ITM.
"""
import sqlite3

import pandas as pd
import pytest

import range_finder.spread_persistence as sp


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE weekly_spx (
            week_start TEXT PRIMARY KEY, week_end TEXT,
            spx_open REAL, spx_high REAL, spx_low REAL, spx_close REAL,
            updated_at TEXT)
    """)
    conn.execute("""
        CREATE TABLE spread_log (
            week_start TEXT, ticker TEXT DEFAULT 'SPX',
            call_short REAL, put_short REAL, spx_ref_close REAL,
            wing_width_used REAL, actual_high REAL, actual_low REAL,
            actual_range_pct REAL, call_breached INTEGER,
            put_breached INTEGER, outcome TEXT, pnl_pts REAL,
            updated_at TEXT,
            PRIMARY KEY (week_start, ticker))
    """)
    return conn


WEEK = "2026-07-06"
FRIDAY = "2026-07-10"


def test_fresh_weekly_spx_row_is_trusted_no_network(monkeypatch):
    conn = _conn()
    conn.execute(
        "INSERT INTO weekly_spx (week_start, spx_open, spx_high, spx_low, "
        "updated_at) VALUES (?, 6000, 6180, 5995, '2026-07-13T13:31:00+00:00')",
        (WEEK,))

    def _no_network(*a, **kw):
        raise AssertionError("fresh row must not trigger a network fetch")

    monkeypatch.setattr(sp.yf, "download", _no_network)
    high, low, week_open = sp._fetch_week_ohlc(conn, WEEK, FRIDAY)
    assert (high, low, week_open) == (6180.0, 5995.0, 6000.0)


def test_stale_partial_row_falls_through_to_network(monkeypatch):
    """Row written Monday 9:31 OF THE SCORED WEEK (refresh failed next
    Monday): must be rejected and the completed bars fetched instead."""
    conn = _conn()
    conn.execute(
        "INSERT INTO weekly_spx (week_start, spx_open, spx_high, spx_low, "
        "updated_at) VALUES (?, 6000, 6003, 5999, '2026-07-06T13:31:00+00:00')",
        (WEEK,))                                   # 4-pt "range": partial bar

    monkeypatch.setattr("range_finder.data_collector._tradier_token",
                        lambda: None)              # skip Tradier arm
    days = pd.date_range(WEEK, FRIDAY, freq="B")
    frame = pd.DataFrame(
        {"Open": [6000, 6010, 6050, 6120, 6150],
         "High": [6020, 6060, 6110, 6170, 6180],
         "Low": [5995, 6005, 6040, 6100, 6140]},
        index=days,
    )
    monkeypatch.setattr(sp.yf, "download", lambda *a, **kw: frame)

    high, low, week_open = sp._fetch_week_ohlc(conn, WEEK, FRIDAY)
    assert high == 6180.0 and low == 5995.0        # REAL week, not 4 pts
    assert week_open == 6000.0


def test_both_sides_touched_books_single_width_loss(monkeypatch):
    conn = _conn()
    conn.execute(
        "INSERT INTO spread_log (week_start, ticker, call_short, put_short, "
        "spx_ref_close, wing_width_used) VALUES (?, 'SPX', 6100, 5900, 6000, 50)",
        (WEEK,))
    conn.commit()
    # Violent week: both strikes touched (H >= call_short, L <= put_short).
    monkeypatch.setattr(sp, "_fetch_week_ohlc",
                        lambda conn_, ws, fs: (6150.0, 5850.0, 6000.0))

    outcome = sp.update_expiration_outcome(WEEK, conn)
    assert outcome == "max_loss"
    row = conn.execute(
        "SELECT pnl_pts, call_breached, put_breached FROM spread_log "
        "WHERE week_start = ?", (WEEK,)).fetchone()
    assert row[0] == pytest.approx(-50.0)          # ONE wing width, not -100
    assert (row[1], row[2]) == (1, 1)
