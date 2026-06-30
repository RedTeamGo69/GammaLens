"""Monday-anchor capture + persistence (the weekly_setup freeze).

These pin the shared helper that backs BOTH the Monday cron and the UI's
mid-week self-heal:

  * ``_daily_open_on`` resolves a given session date out of a 5-day yfinance
    history frame (the Monday bar is 1-3 sessions back when the UI self-heals
    on Tue-Thu),
  * ``capture_and_save_monday_anchor`` writes a ``weekly_setup`` row carrying the
    SCALED underlying Open (XSP rides ^GSPC at /10) and the vol-proxy Open,
  * the self-heal contract: with no spot fallback and a missing Monday bar it
    raises and persists NOTHING — so a mid-week live price can never be locked
    in as the weekly anchor.
"""
import datetime as dt
import sqlite3

import pandas as pd
import pytest

import range_finder.data_collector as dc


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _make_conn() -> sqlite3.Connection:
    """In-memory DB whose weekly_setup matches range_finder.db's DDL.

    The helper uses ``?`` placeholders + ``ON CONFLICT … DO UPDATE`` upsert,
    which sqlite supports natively, so a plain sqlite connection is a faithful
    stand-in for the production psycopg2 wrapper here.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE weekly_setup (
            week_start  TEXT NOT NULL,
            ticker      TEXT NOT NULL DEFAULT 'SPX',
            monday_open REAL,
            monday_vix  REAL,
            captured_at TEXT,
            PRIMARY KEY (week_start, ticker)
        )
        """
    )
    return conn


def _hist(open_by_date: dict) -> pd.DataFrame:
    """A yfinance-shaped daily history: tz-aware index + an 'Open' column."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d).tz_localize("America/New_York") for d in open_by_date]
    )
    return pd.DataFrame({"Open": list(open_by_date.values())}, index=idx)


class _FakeTicker:
    def __init__(self, frame):
        self._frame = frame

    def history(self, period="5d"):
        return self._frame


def _patch_opens(monkeypatch, opens: dict) -> None:
    """Stub the module-level _daily_open_on to return per-symbol opens."""
    monkeypatch.setattr(dc, "_daily_open_on", lambda sym, d: opens.get(sym))


# ── _daily_open_on ─────────────────────────────────────────────────────────────

def test_daily_open_on_resolves_target_date(monkeypatch):
    target = dt.date(2026, 6, 29)  # a Monday
    frame = _hist({
        dt.date(2026, 6, 26): 6750.0,
        target:               6802.5,
        dt.date(2026, 6, 30): 6810.0,
    })
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeTicker(frame))
    assert dc._daily_open_on("^GSPC", target) == pytest.approx(6802.5)


def test_daily_open_on_missing_date_returns_none(monkeypatch):
    frame = _hist({dt.date(2026, 6, 26): 6750.0})
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeTicker(frame))
    assert dc._daily_open_on("^GSPC", dt.date(2026, 6, 29)) is None


def test_daily_open_on_empty_history_returns_none(monkeypatch):
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeTicker(pd.DataFrame()))
    assert dc._daily_open_on("^GSPC", dt.date(2026, 6, 29)) is None


# ── capture_and_save_monday_anchor ─────────────────────────────────────────────

def test_capture_writes_weekly_setup_row(monkeypatch):
    conn = _make_conn()
    _patch_opens(monkeypatch, {"^GSPC": 6800.0, "^VIX": 17.5})

    open_, vix, _src = dc.capture_and_save_monday_anchor(
        conn, "SPX", "2026-06-29", dt.date(2026, 6, 29),
    )

    assert open_ == pytest.approx(6800.0)
    assert vix == pytest.approx(17.5)
    row = conn.execute(
        "SELECT monday_open, monday_vix FROM weekly_setup "
        "WHERE week_start = ? AND ticker = ?",
        ("2026-06-29", "SPX"),
    ).fetchone()
    assert row[0] == pytest.approx(6800.0)
    assert row[1] == pytest.approx(17.5)


def test_capture_scales_mini_underlying(monkeypatch):
    # XSP rides ^GSPC at /10 — the persisted anchor must be the scaled price.
    conn = _make_conn()
    _patch_opens(monkeypatch, {"^GSPC": 6800.0, "^VIX": 17.5})

    open_, _vix, _src = dc.capture_and_save_monday_anchor(
        conn, "XSP", "2026-06-29", dt.date(2026, 6, 29),
    )

    assert open_ == pytest.approx(680.0)   # 6800 / 10
    row = conn.execute(
        "SELECT monday_open FROM weekly_setup WHERE ticker = 'XSP'"
    ).fetchone()
    assert row[0] == pytest.approx(680.0)


def test_capture_uses_spot_fallback_when_bar_missing(monkeypatch):
    conn = _make_conn()
    _patch_opens(monkeypatch, {"^VIX": 17.5})  # underlying bar absent

    open_, _vix, src = dc.capture_and_save_monday_anchor(
        conn, "SPX", "2026-06-29", dt.date(2026, 6, 29),
        spot_fallback=6790.0, live_vix_fallback=18.2,
    )

    assert open_ == pytest.approx(6790.0)
    assert "live spot" in src


def test_selfheal_without_spot_fallback_raises_and_persists_nothing(monkeypatch):
    # The UI self-heal passes spot_fallback=None precisely so a missing Monday
    # bar can NEVER persist a mid-week live price as the weekly anchor.
    conn = _make_conn()
    _patch_opens(monkeypatch, {"^VIX": 17.5})  # underlying bar absent

    with pytest.raises(RuntimeError):
        dc.capture_and_save_monday_anchor(
            conn, "SPX", "2026-06-29", dt.date(2026, 6, 29),
            spot_fallback=None, live_vix_fallback=17.5,
        )

    assert conn.execute("SELECT COUNT(*) FROM weekly_setup").fetchone()[0] == 0
