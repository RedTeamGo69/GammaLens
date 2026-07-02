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


# ── anchor-open source chain (Tradier -> Cboe -> yfinance) ─────────────────────
# A 2026-07-02 yfinance flake made the anchor fall back to live spot mid-week
# (7483.23 instead of Monday's 7391.88). These pin the Tradier-first chain and
# the exact fallback ordering that prevents a repeat.

class _FakeTradierHistory:
    def __init__(self, days):
        self._days = days

    def get_history(self, symbol, interval="daily", start=None, end=None):
        if isinstance(self._days, Exception):
            raise self._days
        return self._days


def _patch_tradier_history(monkeypatch, days, token="tok"):
    monkeypatch.setattr(dc, "_tradier_token", lambda: token)
    import phase1.data_client as pdc
    monkeypatch.setattr(pdc, "TradierDataClient",
                        lambda tok: _FakeTradierHistory(days))


def _cboe_frame(opens_by_date: dict) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in opens_by_date]).normalize()
    vals = list(opens_by_date.values())
    df = pd.DataFrame({"open": vals, "high": vals, "low": vals, "close": vals},
                      index=idx)
    df.index.name = "session_date"
    return df


def test_open_from_tradier_resolves_target_date(monkeypatch):
    _patch_tradier_history(monkeypatch, [
        {"date": "2026-06-29", "open": 7391.88, "high": 7400.0,
         "low": 7350.0, "close": 7380.0, "volume": 0},
    ])
    assert dc._open_from_tradier("^GSPC", dt.date(2026, 6, 29)) \
        == pytest.approx(7391.88)


def test_open_from_tradier_wrong_date_returns_none(monkeypatch):
    _patch_tradier_history(monkeypatch, [
        {"date": "2026-06-26", "open": 7350.0, "high": 7360.0,
         "low": 7340.0, "close": 7355.0, "volume": 0},
    ])
    assert dc._open_from_tradier("^GSPC", dt.date(2026, 6, 29)) is None


def test_open_from_tradier_no_token_returns_none(monkeypatch):
    _patch_tradier_history(monkeypatch, [], token="")
    assert dc._open_from_tradier("^GSPC", dt.date(2026, 6, 29)) is None


def test_open_from_tradier_vxn_unmappable(monkeypatch):
    # ^VXN doesn't quote on Tradier — must decline instantly, no API call.
    _patch_tradier_history(monkeypatch,
                           ConnectionError("must never be called"))
    assert dc._open_from_tradier("^VXN", dt.date(2026, 6, 29)) is None


def test_open_from_tradier_plain_symbol_maps_to_itself(monkeypatch):
    _patch_tradier_history(monkeypatch, [
        {"date": "2026-06-29", "open": 550.25, "high": 555.0,
         "low": 548.0, "close": 552.0, "volume": 100},
    ])
    assert dc._open_from_tradier("QQQ", dt.date(2026, 6, 29)) \
        == pytest.approx(550.25)


def test_open_from_cboe_serves_vxn(monkeypatch):
    import range_finder.cboe_data as cd
    monkeypatch.setattr(cd, "fetch_cboe_index_history",
                        lambda index, timeout=30: _cboe_frame(
                            {"2026-06-29": 21.44}))
    assert dc._open_from_cboe("^VXN", dt.date(2026, 6, 29)) \
        == pytest.approx(21.44)


def test_open_from_cboe_declines_non_vol_symbols(monkeypatch):
    import range_finder.cboe_data as cd
    def _boom(index, timeout=30):
        raise ConnectionError("must never be called")
    monkeypatch.setattr(cd, "fetch_cboe_index_history", _boom)
    assert dc._open_from_cboe("^GSPC", dt.date(2026, 6, 29)) is None


def test_source_chain_prefers_tradier(monkeypatch):
    calls = []

    def tradier(sym, d):
        calls.append("tradier")
        return 7391.88

    def yfin(sym, d):
        calls.append("yf")
        return 9999.0

    monkeypatch.setattr(dc, "_ANCHOR_OPEN_SOURCES", [tradier, dc._open_from_cboe, yfin])

    assert dc._daily_open_on("^GSPC", dt.date(2026, 6, 29)) == pytest.approx(7391.88)
    assert calls == ["tradier"]          # later sources never consulted


def test_source_chain_falls_through_in_order(monkeypatch):
    calls = []

    def tradier(sym, d):
        calls.append("tradier")
        return None                       # Tradier outage

    def cboe(sym, d):
        calls.append("cboe")
        return None                       # not a vol index

    def yfin(sym, d):
        calls.append("yf")
        return 7391.88

    monkeypatch.setattr(dc, "_ANCHOR_OPEN_SOURCES", [tradier, cboe, yfin])

    assert dc._daily_open_on("^GSPC", dt.date(2026, 6, 29)) == pytest.approx(7391.88)
    assert calls == ["tradier", "cboe", "yf"]


def test_source_chain_all_dry_returns_none(monkeypatch):
    monkeypatch.setattr(dc, "_ANCHOR_OPEN_SOURCES",
                        [lambda s, d: None, lambda s, d: None])
    assert dc._daily_open_on("^GSPC", dt.date(2026, 6, 29)) is None
