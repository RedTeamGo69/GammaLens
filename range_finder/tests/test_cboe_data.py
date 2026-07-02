"""Cboe index-history fetch + VIX1D backfill (range_finder.cboe_data).

Pins the contracts that make the Cboe layer safe to run unattended:

  * CSV parsing tolerates both date formats Cboe has shipped and coerces
    (rather than crashes on) bad rows,
  * ``backfill_vix1d`` is UPDATE-only (never invents daily_spx rows), fills
    only NULLs by default (rerun-idempotent), and replaces values only with
    ``overwrite=True``,
  * ``merge_cboe_closes`` overlays Cboe values without mutating its input and
    degrades to the input frame when Cboe is unavailable.
"""
import sqlite3

import pandas as pd
import pytest

import range_finder.cboe_data as cd


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _make_conn() -> sqlite3.Connection:
    """In-memory DB whose daily_spx matches range_finder.db's DDL (subset)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE daily_spx (
            session_date TEXT NOT NULL,
            ticker       TEXT NOT NULL DEFAULT 'SPX',
            spx_open     REAL,
            spx_close    REAL,
            vix1d_close  REAL,
            updated_at   TEXT,
            PRIMARY KEY (session_date, ticker)
        )
        """
    )
    return conn


def _seed(conn, rows):
    """rows: iterable of (session_date, vix1d_close_or_None)."""
    conn.executemany(
        "INSERT INTO daily_spx (session_date, ticker, spx_open, spx_close, vix1d_close) "
        "VALUES (?, 'SPX', 5000.0, 5010.0, ?)",
        rows,
    )
    conn.commit()


def _cboe_frame(closes_by_date: dict) -> pd.DataFrame:
    """A fetch_cboe_index_history-shaped frame: normalized index, ohlc floats."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in closes_by_date]).normalize()
    vals = list(closes_by_date.values())
    df = pd.DataFrame(
        {"open": vals, "high": vals, "low": vals, "close": vals}, index=idx
    )
    df.index.name = "session_date"
    return df


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patch_csv(monkeypatch, csv_text: str) -> None:
    monkeypatch.setattr(cd.requests, "get",
                        lambda url, timeout=30: _FakeResponse(csv_text))


def _patch_history(monkeypatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(cd, "fetch_cboe_index_history", lambda index, timeout=30: frame)


# ── fetch_cboe_index_history parsing ──────────────────────────────────────────

def test_fetch_parses_mdy_dates(monkeypatch):
    _patch_csv(monkeypatch,
               "DATE,OPEN,HIGH,LOW,CLOSE\n"
               "05/13/2022,33.63,33.63,33.63,33.63\n"
               "05/16/2022,30.10,31.00,29.90,30.55\n")
    df = cd.fetch_cboe_index_history("VIX1D")
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2022-05-13")
    assert df["close"].iloc[1] == pytest.approx(30.55)


def test_fetch_parses_iso_dates(monkeypatch):
    _patch_csv(monkeypatch,
               "DATE,OPEN,HIGH,LOW,CLOSE\n"
               "2022-05-13,33.63,33.63,33.63,33.63\n"
               "2022-05-16,30.10,31.00,29.90,30.55\n")
    df = cd.fetch_cboe_index_history("VIX1D")
    assert list(df.index) == [pd.Timestamp("2022-05-13"), pd.Timestamp("2022-05-16")]


def test_fetch_drops_unparseable_rows(monkeypatch):
    _patch_csv(monkeypatch,
               "DATE,OPEN,HIGH,LOW,CLOSE\n"
               "05/13/2022,33.63,33.63,33.63,33.63\n"
               "05/16/2022,30.10,31.00,29.90,30.55\n"
               "05/17/2022,29.00,29.50,28.80,29.20\n"
               "05/18/2022,28.00,28.50,27.80,28.20\n"
               "05/19/2022,27.00,27.50,26.80,27.20\n"
               "05/20/2022,26.00,26.50,25.80,26.20\n"
               "05/23/2022,25.00,25.50,24.80,25.20\n"
               "05/24/2022,24.00,24.50,23.80,24.20\n"
               "05/25/2022,23.00,23.50,22.80,23.20\n"
               "not-a-date,22.00,22.50,21.80,22.20\n")
    df = cd.fetch_cboe_index_history("VIX1D")
    assert len(df) == 9  # bad row dropped, good rows kept
    assert not df.index.isna().any()


def test_fetch_rejects_unexpected_columns(monkeypatch):
    _patch_csv(monkeypatch, "WHAT,EVEN,IS,THIS\n1,2,3,4\n")
    with pytest.raises(ValueError):
        cd.fetch_cboe_index_history("VIX1D")


def test_fetch_sorts_ascending(monkeypatch):
    _patch_csv(monkeypatch,
               "DATE,OPEN,HIGH,LOW,CLOSE\n"
               "05/16/2022,30.10,31.00,29.90,30.55\n"
               "05/13/2022,33.63,33.63,33.63,33.63\n")
    df = cd.fetch_cboe_index_history("VIX1D")
    assert df.index.is_monotonic_increasing


# ── backfill_vix1d ─────────────────────────────────────────────────────────────

def test_backfill_fills_only_nulls(monkeypatch):
    conn = _make_conn()
    _seed(conn, [("2022-05-13", None), ("2022-05-16", 30.55), ("2022-05-17", None)])
    _patch_history(monkeypatch, _cboe_frame({
        "2022-05-13": 33.63, "2022-05-16": 99.99, "2022-05-17": 29.20,
    }))

    updated = cd.backfill_vix1d(conn)

    assert updated == 2
    rows = dict(conn.execute(
        "SELECT session_date, vix1d_close FROM daily_spx ORDER BY session_date"
    ).fetchall())
    assert rows["2022-05-13"] == pytest.approx(33.63)
    assert rows["2022-05-16"] == pytest.approx(30.55)   # existing value untouched
    assert rows["2022-05-17"] == pytest.approx(29.20)


def test_backfill_is_idempotent(monkeypatch):
    conn = _make_conn()
    _seed(conn, [("2022-05-13", None)])
    _patch_history(monkeypatch, _cboe_frame({"2022-05-13": 33.63}))

    assert cd.backfill_vix1d(conn) == 1
    assert cd.backfill_vix1d(conn) == 0  # second run finds nothing to fill


def test_backfill_overwrite_replaces_values(monkeypatch):
    conn = _make_conn()
    _seed(conn, [("2022-05-16", 30.55)])
    _patch_history(monkeypatch, _cboe_frame({"2022-05-16": 99.99}))

    assert cd.backfill_vix1d(conn, overwrite=True) == 1
    val = conn.execute(
        "SELECT vix1d_close FROM daily_spx WHERE session_date = '2022-05-16'"
    ).fetchone()[0]
    assert val == pytest.approx(99.99)


def test_backfill_never_inserts_rows(monkeypatch):
    conn = _make_conn()
    _seed(conn, [("2022-05-13", None)])
    # Cboe has a session daily_spx doesn't — it must NOT become a new row.
    _patch_history(monkeypatch, _cboe_frame({
        "2022-05-13": 33.63, "2022-05-14": 11.11,
    }))

    cd.backfill_vix1d(conn)

    assert conn.execute("SELECT COUNT(*) FROM daily_spx").fetchone()[0] == 1


def test_backfill_respects_start(monkeypatch):
    conn = _make_conn()
    _seed(conn, [("2022-05-13", None), ("2023-01-03", None)])
    _patch_history(monkeypatch, _cboe_frame({
        "2022-05-13": 33.63, "2023-01-03": 21.50,
    }))

    assert cd.backfill_vix1d(conn, start="2023-01-01") == 1
    val = conn.execute(
        "SELECT vix1d_close FROM daily_spx WHERE session_date = '2022-05-13'"
    ).fetchone()[0]
    assert val is None


# ── vix1d_coverage ─────────────────────────────────────────────────────────────

def test_coverage_math():
    conn = _make_conn()
    _seed(conn, [
        ("2022-04-25", None),          # before Cboe window — not backfillable
        ("2022-05-13", None),          # in window, NULL — backfillable
        ("2023-04-24", 17.5),          # covered
    ])

    cov = cd.vix1d_coverage(conn)

    assert cov == {
        "min_date": "2022-04-25",
        "max_date": "2023-04-24",
        "non_null": 1,
        "total": 3,
        "null_in_cboe_window": 1,
    }


# ── merge_cboe_closes / merge_cboe_vix1d ───────────────────────────────────────

def _daily_frame():
    idx = pd.DatetimeIndex(["2022-05-13", "2022-05-16", "2022-05-17"])
    return pd.DataFrame({
        "spx_close":   [4000.0, 4010.0, 4020.0],
        "vix1d_close": [None, 30.55, None],
    }, index=idx)


def test_merge_cboe_wins_and_fills(monkeypatch):
    df = _daily_frame()
    _patch_history(monkeypatch, _cboe_frame({
        "2022-05-13": 33.63, "2022-05-16": 99.99,
    }))

    out = cd.merge_cboe_vix1d(df)

    assert out["vix1d_close"].iloc[0] == pytest.approx(33.63)  # Cboe fills NULL
    assert out["vix1d_close"].iloc[1] == pytest.approx(99.99)  # Cboe wins over existing
    assert out["vix1d_close"].iloc[2] is None or pd.isna(out["vix1d_close"].iloc[2])
    assert len(out) == len(df)                                  # no rows added/removed


def test_merge_degrades_on_cboe_outage(monkeypatch):
    df = _daily_frame()

    def _boom(index, timeout=30):
        raise ConnectionError("cdn down")
    monkeypatch.setattr(cd, "fetch_cboe_index_history", _boom)

    out = cd.merge_cboe_vix1d(df)

    pd.testing.assert_frame_equal(out, df)   # values degrade to input
    assert out is not df                     # ...but still a new frame


def test_merge_does_not_mutate_input(monkeypatch):
    df = _daily_frame()
    original = df.copy()
    _patch_history(monkeypatch, _cboe_frame({"2022-05-13": 33.63}))

    cd.merge_cboe_vix1d(df)

    pd.testing.assert_frame_equal(df, original)


# ── resample_cboe_weekly ───────────────────────────────────────────────────────

def _cboe_daily_two_weeks() -> pd.DataFrame:
    """Two trading weeks of daily bars: 2026-06-22..26 and a holiday-short
    2026-06-29..07-02 (no Friday session)."""
    days = {
        # week of Mon 2026-06-22
        "2026-06-22": (10.0, 12.0,  9.0, 11.0),
        "2026-06-23": (11.0, 13.0, 10.0, 12.0),
        "2026-06-24": (12.0, 14.0, 11.0, 13.0),
        "2026-06-25": (13.0, 15.0, 12.0, 14.0),
        "2026-06-26": (14.0, 16.0, 13.0, 15.0),
        # week of Mon 2026-06-29 (short week)
        "2026-06-29": (15.0, 17.0, 14.0, 16.0),
        "2026-06-30": (16.0, 18.0, 15.0, 17.0),
        "2026-07-01": (17.0, 19.0, 16.0, 18.0),
        "2026-07-02": (18.0, 20.0, 17.0, 19.0),
    }
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    df = pd.DataFrame(
        [list(v) for v in days.values()],
        columns=["open", "high", "low", "close"], index=idx,
    )
    df.index.name = "session_date"
    return df


def test_resample_weekly_monday_labels_and_ohlc():
    weekly = cd.resample_cboe_weekly(_cboe_daily_two_weeks())

    # Labeled by each week's MONDAY (yfinance 1wk convention)
    assert list(weekly.index) == [pd.Timestamp("2026-06-22"),
                                  pd.Timestamp("2026-06-29")]
    wk1 = weekly.loc["2026-06-22"]
    assert wk1["open"] == pytest.approx(10.0)   # Monday's open
    assert wk1["high"] == pytest.approx(16.0)   # week max
    assert wk1["low"] == pytest.approx(9.0)     # week min
    assert wk1["close"] == pytest.approx(15.0)  # Friday's close


def test_resample_weekly_handles_short_week():
    weekly = cd.resample_cboe_weekly(_cboe_daily_two_weeks())

    wk2 = weekly.loc["2026-06-29"]
    assert wk2["open"] == pytest.approx(15.0)
    assert wk2["close"] == pytest.approx(19.0)  # Thursday close (no Friday bar)


def test_resample_weekly_drops_empty_weeks():
    daily = _cboe_daily_two_weeks()
    # Remove the whole second week — the resampler must not emit an all-NaN row
    daily = daily[daily.index < "2026-06-29"]

    weekly = cd.resample_cboe_weekly(daily)

    assert list(weekly.index) == [pd.Timestamp("2026-06-22")]


# ── merge_cboe_weekly_ohlc / fetch_cboe_weekly_closes ─────────────────────────

_VIX_COL_MAP = {"open": "vix_open", "high": "vix_high",
                "low": "vix_low", "close": "vix_close"}


def _weekly_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2026-06-22", "2026-06-29"])
    idx.name = "week_start"
    return pd.DataFrame({
        "spx_open":  [6000.0, 6050.0],
        "vix_open":  [17.0, 18.0],
        "vix_high":  [19.0, 20.0],
        "vix_low":   [16.0, 17.0],
        "vix_close": [18.0, None],   # in-progress week: yfinance has no close yet
    }, index=idx)


def test_merge_weekly_cboe_wins_and_seam_fills(monkeypatch):
    _patch_history(monkeypatch, _cboe_daily_two_weeks())
    df = _weekly_frame()

    out = cd.merge_cboe_weekly_ohlc(df, "VIX", _VIX_COL_MAP)

    # Completed week: Cboe's resampled bar wins over yfinance
    assert out.loc["2026-06-22", "vix_open"] == pytest.approx(10.0)
    assert out.loc["2026-06-22", "vix_close"] == pytest.approx(15.0)
    # In-progress week: Cboe (which has partial sessions) still provides the bar
    assert out.loc["2026-06-29", "vix_close"] == pytest.approx(19.0)
    # Non-mapped columns untouched, no rows added/removed
    assert out.loc["2026-06-22", "spx_open"] == pytest.approx(6000.0)
    assert len(out) == 2


def test_merge_weekly_degrades_on_outage(monkeypatch):
    def _boom(index, timeout=30):
        raise ConnectionError("cdn down")
    monkeypatch.setattr(cd, "fetch_cboe_index_history", _boom)
    df = _weekly_frame()

    out = cd.merge_cboe_weekly_ohlc(df, "VIX", _VIX_COL_MAP)

    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_merge_weekly_does_not_mutate_input(monkeypatch):
    _patch_history(monkeypatch, _cboe_daily_two_weeks())
    df = _weekly_frame()
    original = df.copy()

    cd.merge_cboe_weekly_ohlc(df, "VIX", _VIX_COL_MAP)

    pd.testing.assert_frame_equal(df, original)


def test_fetch_weekly_closes_series(monkeypatch):
    _patch_history(monkeypatch, _cboe_daily_two_weeks())

    s = cd.fetch_cboe_weekly_closes("VIX9D", "vix9d_close")

    assert s.name == "vix9d_close"
    assert s.index.name == "week_start"
    assert s.loc[pd.Timestamp("2026-06-22")] == pytest.approx(15.0)


def test_yf_to_cboe_mapping_covers_app_vol_proxies():
    # Every vol proxy configured in ticker_config must map to a Cboe index
    # (or explicitly stay yfinance-only). ^VIX and ^VXN are the two in use.
    assert cd.YF_TO_CBOE_INDEX["^VIX"] == "VIX"
    assert cd.YF_TO_CBOE_INDEX["^VXN"] == "VXN"
