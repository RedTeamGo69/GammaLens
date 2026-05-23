# =============================================================================
# data_collector.py
# Weekly SPX Range Prediction Model — Data Collection Module
#
# Pulls and stores historical SPX OHLC, VIX, and FRED macro data to Postgres
# (via range_finder.db.get_connection()) for use by downstream modules.
#
# Data Sources:
#   - yfinance  : SPX weekly OHLC, VIX weekly close
#   - FRED API  : 10Y Treasury yield, 2Y Treasury yield, Fed Funds rate
# =============================================================================

import math
import os
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

# =============================================================================
# CONFIG
# =============================================================================

# Read FRED key from Streamlit secrets or environment
FRED_API_KEY = ""
try:
    import streamlit as st
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
except Exception:
    pass
if not FRED_API_KEY:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def fred_key_status() -> str:
    """Human-readable description of where the FRED key was (or wasn't)
    resolved from, without leaking the key itself. Use in UI captions /
    log lines so "is my key actually loaded?" is diagnosable without
    having to crack open the secrets panel."""
    if not FRED_API_KEY:
        return "not set (neither st.secrets['FRED_API_KEY'] nor $FRED_API_KEY)"
    return f"configured ({len(FRED_API_KEY)} chars; FRED keys are normally 32)"

# How many years of history to pull on initial load
HISTORY_YEARS = 6

# FRED series used as macro features
FRED_SERIES = {
    "treasury_10y": "DGS10",       # 10-Year Treasury Constant Maturity Rate
    "treasury_2y":  "DGS2",        # 2-Year Treasury Constant Maturity Rate
    "fed_funds":    "DFF",         # Federal Funds Effective Rate
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_db():
    """
    Connect to Postgres (via DATABASE_URL) and ensure all range-finder tables
    exist.

    Tables:
        weekly_spx  — SPX and VIX weekly OHLC + derived range metrics
        macro_daily — Daily FRED macro series (joined to weekly during feature build)
        event_flags — Manual or scraped FOMC/CPI/NFP week flags
    """
    from range_finder.db import get_connection, init_all_tables
    conn = get_connection()
    init_all_tables(conn)
    return conn


# =============================================================================
# SPX + VIX DATA
# =============================================================================

def fetch_spx_vix(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Pull weekly SPX and VIX OHLC from yfinance.

    yfinance weekly bars run Monday open → Friday close.
    VIX close is aligned to the same week as SPX — in the feature builder
    this gets lagged by one week (you observe Friday's VIX before next week opens).

    Returns a single DataFrame with prefixed columns: spx_*, vix_*
    """
    end   = datetime.today()
    start = end - timedelta(weeks=years * 52 + 4)   # small buffer for alignment

    log.info(f"Fetching SPX weekly OHLC from {start.date()} to {end.date()}")
    spx_raw = yf.download("^GSPC", start=start, end=end, interval="1wk", progress=False, timeout=60)

    if spx_raw.empty:
        raise RuntimeError("yfinance returned empty SPX data — market may be closed or network issue")

    log.info(f"Fetching VIX weekly OHLC from {start.date()} to {end.date()}")
    vix_raw = yf.download("^VIX", start=start, end=end, interval="1wk", progress=False, timeout=60)

    # yfinance sometimes returns a MultiIndex — flatten it
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    # Rename and prefix
    spx = spx_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close", "spx_volume"]

    vix = vix_raw[["Open", "High", "Low", "Close"]].copy()
    vix.columns = ["vix_open", "vix_high", "vix_low", "vix_close"]

    # Merge on date index
    df = spx.join(vix, how="inner")
    df.index.name = "week_start"
    df.index = pd.to_datetime(df.index).normalize()   # strip time component

    # Derived range metrics — BUG FIX: use math.log directly instead of inline __import__
    df["range_pts"]   = df["spx_high"] - df["spx_low"]
    df["range_pct"]   = df["range_pts"] / df["spx_open"]
    df["log_range"]   = df["range_pct"].apply(lambda x: pd.NA if x <= 0 else math.log(x))
    df["spx_return"]  = (df["spx_close"] - df["spx_open"]) / df["spx_open"]

    # Approximate week_end (Friday = Monday + 4 days)
    df["week_end"] = df.index + timedelta(days=4)

    df.dropna(subset=["spx_open", "spx_close", "range_pct"], inplace=True)

    log.info(f"SPX/VIX: {len(df)} weekly rows collected")
    return df


def save_spx_vix(conn, df: pd.DataFrame) -> int:
    """
    Upsert weekly SPX/VIX rows into the database.
    Returns the number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0

    cur = conn.cursor()
    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO weekly_spx (
                week_start, week_end,
                spx_open, spx_high, spx_low, spx_close, spx_volume,
                vix_open, vix_high, vix_low, vix_close,
                range_pts, range_pct, log_range, spx_return,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(week_start) DO UPDATE SET
                week_end    = excluded.week_end,
                spx_open    = excluded.spx_open,
                spx_high    = excluded.spx_high,
                spx_low     = excluded.spx_low,
                spx_close   = excluded.spx_close,
                spx_volume  = excluded.spx_volume,
                vix_open    = excluded.vix_open,
                vix_high    = excluded.vix_high,
                vix_low     = excluded.vix_low,
                vix_close   = excluded.vix_close,
                range_pts   = excluded.range_pts,
                range_pct   = excluded.range_pct,
                log_range   = excluded.log_range,
                spx_return  = excluded.spx_return,
                updated_at  = excluded.updated_at
        """, (
            week_start.strftime("%Y-%m-%d"),
            row["week_end"].strftime("%Y-%m-%d") if pd.notna(row["week_end"]) else None,
            _safe(row, "spx_open"),   _safe(row, "spx_high"),
            _safe(row, "spx_low"),    _safe(row, "spx_close"),
            _safe(row, "spx_volume"),
            _safe(row, "vix_open"),   _safe(row, "vix_high"),
            _safe(row, "vix_low"),    _safe(row, "vix_close"),
            _safe(row, "range_pts"),  _safe(row, "range_pct"),
            _safe(row, "log_range"),  _safe(row, "spx_return"),
            now,
        ))
        rows_written += 1

    conn.commit()
    log.info(f"SPX/VIX: {rows_written} rows upserted into weekly_spx")
    return rows_written


# =============================================================================
# PER-TICKER UNDERLYING + VOL-PROXY DATA
# =============================================================================
# QQQ / AMZN / AMD don't piggyback on SPX history — each gets its own weekly
# OHLC + per-ticker vol-proxy series (VXN for QQQ, VIX for stocks). SPX/XSP
# continue to use weekly_spx above; the schema split keeps the SPX pipeline
# unchanged.

def fetch_underlying_weekly(
    ticker: str,
    yf_symbol: str,
    vol_proxy_yf: str,
    years: int = HISTORY_YEARS,
) -> pd.DataFrame:
    """Pull weekly OHLC + per-ticker vol-proxy OHLC from yfinance.

    Returns one DataFrame indexed by week_start with columns:
        open, high, low, close, volume,
        vol_proxy_open, vol_proxy_high, vol_proxy_low, vol_proxy_close,
        range_pts, range_pct, log_range, return_pct, week_end

    Mirrors the schema of ``fetch_spx_vix`` so the downstream feature builder
    can treat the two outputs uniformly.
    """
    end   = datetime.today()
    start = end - timedelta(weeks=years * 52 + 4)

    log.info(f"[{ticker}] Fetching weekly OHLC ({yf_symbol}) {start.date()} → {end.date()}")
    raw = yf.download(yf_symbol, start=start, end=end, interval="1wk",
                      progress=False, timeout=60)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {yf_symbol} ({ticker}) — "
            "market closed, network issue, or unsupported symbol."
        )

    log.info(f"[{ticker}] Fetching weekly vol proxy ({vol_proxy_yf})")
    vp_raw = yf.download(vol_proxy_yf, start=start, end=end, interval="1wk",
                         progress=False, timeout=60)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if isinstance(vp_raw.columns, pd.MultiIndex):
        vp_raw.columns = vp_raw.columns.get_level_values(0)

    base = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    base.columns = ["open", "high", "low", "close", "volume"]

    if not vp_raw.empty:
        vp = vp_raw[["Open", "High", "Low", "Close"]].copy()
        vp.columns = ["vol_proxy_open", "vol_proxy_high", "vol_proxy_low", "vol_proxy_close"]
        df = base.join(vp, how="left")
    else:
        log.warning(f"[{ticker}] vol proxy {vol_proxy_yf} returned empty — vol_proxy_* will be NaN")
        for col in ("vol_proxy_open", "vol_proxy_high", "vol_proxy_low", "vol_proxy_close"):
            base[col] = pd.NA
        df = base

    df.index.name = "week_start"
    df.index = pd.to_datetime(df.index).normalize()

    df["range_pts"]   = df["high"] - df["low"]
    df["range_pct"]   = df["range_pts"] / df["open"]
    df["log_range"]   = df["range_pct"].apply(lambda x: pd.NA if (x is None or pd.isna(x) or x <= 0) else math.log(x))
    df["return_pct"]  = (df["close"] - df["open"]) / df["open"]
    df["week_end"]    = df.index + timedelta(days=4)

    df.dropna(subset=["open", "close", "range_pct"], inplace=True)

    log.info(f"[{ticker}] weekly_underlying: {len(df)} rows collected")
    return df


def save_underlying_weekly(conn, ticker: str, df: pd.DataFrame) -> int:
    """Upsert weekly_underlying rows for one ticker. Returns number written."""
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0
    cur = conn.cursor()
    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO weekly_underlying (
                ticker, week_start, week_end,
                open, high, low, close, volume,
                vol_proxy_open, vol_proxy_high, vol_proxy_low, vol_proxy_close,
                range_pts, range_pct, log_range, return_pct,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, week_start) DO UPDATE SET
                week_end        = excluded.week_end,
                open            = excluded.open,
                high            = excluded.high,
                low             = excluded.low,
                close           = excluded.close,
                volume          = excluded.volume,
                vol_proxy_open  = excluded.vol_proxy_open,
                vol_proxy_high  = excluded.vol_proxy_high,
                vol_proxy_low   = excluded.vol_proxy_low,
                vol_proxy_close = excluded.vol_proxy_close,
                range_pts       = excluded.range_pts,
                range_pct       = excluded.range_pct,
                log_range       = excluded.log_range,
                return_pct      = excluded.return_pct,
                updated_at      = excluded.updated_at
        """, (
            ticker,
            week_start.strftime("%Y-%m-%d"),
            row["week_end"].strftime("%Y-%m-%d") if pd.notna(row["week_end"]) else None,
            _safe(row, "open"),  _safe(row, "high"),
            _safe(row, "low"),   _safe(row, "close"),
            _safe(row, "volume"),
            _safe(row, "vol_proxy_open"),  _safe(row, "vol_proxy_high"),
            _safe(row, "vol_proxy_low"),   _safe(row, "vol_proxy_close"),
            _safe(row, "range_pts"),       _safe(row, "range_pct"),
            _safe(row, "log_range"),       _safe(row, "return_pct"),
            now,
        ))
        rows_written += 1

    conn.commit()
    log.info(f"[{ticker}] weekly_underlying: {rows_written} rows upserted")
    return rows_written


def get_weekly_underlying(conn, ticker: str, limit: int = None) -> pd.DataFrame:
    """Return weekly_underlying for one ticker as a DataFrame indexed by week_start."""
    query = "SELECT * FROM weekly_underlying WHERE ticker = ? ORDER BY week_start ASC"
    params = (ticker,)
    if limit:
        query += " LIMIT ?"
        params = (ticker, int(limit))
    df = pd.read_sql_query(query, conn, params=params,
                           parse_dates=["week_start", "week_end"])
    df.set_index("week_start", inplace=True)
    return df


# =============================================================================
# EARNINGS FLAGS (single-stock weekly gate)
# =============================================================================

def populate_earnings_flags(conn, ticker: str) -> int:
    """Backfill historical + upcoming earnings dates for a single-stock ticker.

    Pulls dates from yfinance ``Ticker.earnings_dates`` (typically 4-12
    quarters available) and writes a ``has_earnings=1`` row to
    ``earnings_flags`` for the week containing each earnings date.

    Index/ETF tickers (SPX, XSP, QQQ) shouldn't call this — they have no
    single-name earnings event. Returns the number of weeks flagged.
    """
    try:
        edf = yf.Ticker(ticker).earnings_dates
    except Exception as e:
        log.warning(f"[{ticker}] yfinance earnings_dates fetch failed: {e}")
        return 0

    if edf is None or edf.empty:
        log.info(f"[{ticker}] yfinance returned no earnings_dates")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = 0
    cur = conn.cursor()

    # earnings_dates is indexed by tz-aware datetimes. Map each to its Monday-
    # of-week so it joins with model_features.week_start cleanly.
    for ts in edf.index:
        try:
            d = ts.to_pydatetime().date() if hasattr(ts, "to_pydatetime") else ts.date()
        except Exception:
            continue
        # Monday of the week (weekday() == 0)
        monday = d - timedelta(days=d.weekday())
        week_start = monday.strftime("%Y-%m-%d")
        cur.execute("""
            INSERT INTO earnings_flags (ticker, week_start, has_earnings, earnings_date, updated_at)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT (ticker, week_start) DO UPDATE SET
                has_earnings  = 1,
                earnings_date = excluded.earnings_date,
                updated_at    = excluded.updated_at
        """, (ticker, week_start, d.strftime("%Y-%m-%d"), now))
        rows += 1

    conn.commit()
    log.info(f"[{ticker}] earnings_flags: {rows} weeks flagged")
    return rows


def get_earnings_flag(conn, ticker: str, week_start: str) -> bool:
    """Return True if the (ticker, week_start) row has has_earnings=1."""
    cur = conn.cursor()
    cur.execute(
        "SELECT has_earnings FROM earnings_flags WHERE ticker = ? AND week_start = ?",
        (ticker, week_start),
    )
    row = cur.fetchone()
    return bool(row and row[0])


# =============================================================================
# FRED MACRO DATA
# =============================================================================

def fetch_fred_macro(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Pull daily macro series from FRED.

    Series pulled:
        DGS10  — 10-Year Treasury yield
        DGS2   — 2-Year Treasury yield
        DFF    — Federal Funds Effective Rate

    Yield spread (10y - 2y) is computed here. The feature builder will
    resample this to weekly frequency and align it to SPX weeks.
    """
    from fredapi import Fred

    fred  = Fred(api_key=FRED_API_KEY)
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    frames = {}
    for col_name, series_id in FRED_SERIES.items():
        log.info(f"Fetching FRED series: {series_id} ({col_name})")
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        frames[col_name] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).normalize()

    # Forward-fill small gaps (FRED has occasional missing business days)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Derived feature
    df["yield_spread"] = df["treasury_10y"] - df["treasury_2y"]

    log.info(f"FRED macro: {len(df)} daily rows collected")
    return df


def save_fred_macro(conn, df: pd.DataFrame) -> int:
    """
    Upsert daily FRED macro rows into the database.
    Returns the number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0

    cur = conn.cursor()
    for date, row in df.iterrows():
        cur.execute("""
            INSERT INTO macro_daily (
                date, treasury_10y, treasury_2y, yield_spread, fed_funds, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                treasury_10y  = excluded.treasury_10y,
                treasury_2y   = excluded.treasury_2y,
                yield_spread  = excluded.yield_spread,
                fed_funds     = excluded.fed_funds,
                updated_at    = excluded.updated_at
        """, (
            date.strftime("%Y-%m-%d"),
            _safe(row, "treasury_10y"),
            _safe(row, "treasury_2y"),
            _safe(row, "yield_spread"),
            _safe(row, "fed_funds"),
            now,
        ))
        rows_written += 1

    conn.commit()
    log.info(f"FRED macro: {rows_written} rows upserted into macro_daily")
    return rows_written


# =============================================================================
# EVENT FLAGS  (FOMC / CPI / NFP) — re-exported from event_calendars.py
# =============================================================================

from range_finder.event_calendars import (  # noqa: F401
    FOMC_DATES,
    CPI_DATES,
    NFP_DATES,
    _get_week_start,
    build_event_flags,
)


# =============================================================================
# WEEKLY UPDATE  (call this every Friday evening)
# =============================================================================

def update_weekly(conn) -> None:
    """
    Incremental update — pulls the last 8 weeks of data and upserts.
    Use this every Friday after market close instead of the full initial load.
    """
    log.info("Running incremental weekly update...")
    df_spx = fetch_spx_vix(years=0.2)    # ~10 weeks
    save_spx_vix(conn, df_spx)
    df_macro = fetch_fred_macro(years=0.2)
    save_fred_macro(conn, df_macro)
    build_event_flags(conn)
    log.info("Weekly update complete.")


# =============================================================================
# UTILITY
# =============================================================================

def _safe(row: pd.Series, col: str):
    """Return float or None — psycopg2 doesn't like numpy scalars or pd.NA."""
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_weekly_spx(conn, limit: int = None) -> pd.DataFrame:
    """
    Convenience reader — returns weekly_spx as a DataFrame sorted by week_start.
    """
    query = "SELECT * FROM weekly_spx ORDER BY week_start ASC"
    params = None
    if limit:
        query += " LIMIT ?"
        params = (int(limit),)
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["week_start", "week_end"])
    df.set_index("week_start", inplace=True)
    return df


def get_macro_daily(conn) -> pd.DataFrame:
    """Returns macro_daily as a DataFrame indexed by date."""
    df = pd.read_sql_query(
        "SELECT * FROM macro_daily ORDER BY date ASC", conn, parse_dates=["date"]
    )
    df.set_index("date", inplace=True)
    return df


def get_event_flags(conn) -> pd.DataFrame:
    """Returns event_flags as a DataFrame indexed by week_start."""
    df = pd.read_sql_query(
        "SELECT * FROM event_flags ORDER BY week_start ASC", conn, parse_dates=["week_start"]
    )
    df.set_index("week_start", inplace=True)
    return df


# =============================================================================
# DAILY SPX + VIX + VIX1D  (0DTE spread finder data layer)
# =============================================================================
# Separate from the weekly fetchers so the existing weekly_spx pipeline stays
# untouched. VIX1D (^VIX1D on yfinance) only exists from ~2022-04-25 so the
# daily series carries NULL vix1d_close before that — daily HAR specs that
# don't depend on VIX1D (M1_daily_baseline) can still train on the full window.

def fetch_daily_spx_vix(years: int = 4) -> pd.DataFrame:
    """Pull daily SPX OHLC + VIX + VIX1D from yfinance.

    Returns DataFrame indexed by session_date (Monday-Friday calendar dates,
    no weekend rows) with columns:
        spx_open, spx_high, spx_low, spx_close,
        range_pts, range_pct, log_range, spx_return,
        vix_close, vix1d_close

    VIX1D pre-2022 rows are NaN — the feature builder lets daily HAR train
    on a smaller set of specs in that window via feature_has_enough_data.
    """
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365 + 30))

    log.info(f"Fetching daily SPX OHLC from {start.date()} to {end.date()}")
    spx_raw = yf.download("^GSPC", start=start, end=end, interval="1d",
                          progress=False, timeout=60)
    if spx_raw.empty:
        raise RuntimeError("yfinance returned empty daily SPX data — "
                           "market may be closed or network issue")

    log.info("Fetching daily VIX closes")
    vix_raw = yf.download("^VIX", start=start, end=end, interval="1d",
                          progress=False, timeout=60)

    log.info("Fetching daily VIX1D closes")
    vix1d_raw = yf.download("^VIX1D", start=start, end=end, interval="1d",
                            progress=False, timeout=60)

    def _flatten(raw):
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    spx_raw = _flatten(spx_raw)
    vix_raw = _flatten(vix_raw)
    vix1d_raw = _flatten(vix1d_raw)

    spx = spx_raw[["Open", "High", "Low", "Close"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close"]

    vix = vix_raw[["Close"]].copy() if not vix_raw.empty and "Close" in vix_raw.columns \
        else pd.DataFrame(columns=["vix_close"])
    if not vix.empty:
        vix.columns = ["vix_close"]

    if not vix1d_raw.empty and "Close" in vix1d_raw.columns:
        vix1d = vix1d_raw[["Close"]].copy()
        vix1d.columns = ["vix1d_close"]
    else:
        log.warning("VIX1D returned empty — vix1d_close will be NULL "
                    "(expected pre-2022 or on network errors).")
        vix1d = pd.DataFrame(columns=["vix1d_close"])

    df = spx.join(vix, how="left")
    df = df.join(vix1d, how="left")
    df.index.name = "session_date"
    df.index = pd.to_datetime(df.index).normalize()

    df["range_pts"]  = df["spx_high"] - df["spx_low"]
    df["range_pct"]  = df["range_pts"] / df["spx_open"]
    df["log_range"]  = df["range_pct"].apply(
        lambda x: pd.NA if (x is None or pd.isna(x) or x <= 0) else math.log(x)
    )
    df["spx_return"] = (df["spx_close"] - df["spx_open"]) / df["spx_open"]

    df.dropna(subset=["spx_open", "spx_close", "range_pct"], inplace=True)
    log.info(f"Daily SPX/VIX/VIX1D: {len(df)} daily rows collected")
    return df


def save_daily_spx(conn, df: pd.DataFrame, ticker: str = "SPX") -> int:
    """Upsert daily SPX/VIX/VIX1D rows into the daily_spx table."""
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0
    cur = conn.cursor()
    for session_date, row in df.iterrows():
        cur.execute("""
            INSERT INTO daily_spx (
                session_date, ticker,
                spx_open, spx_high, spx_low, spx_close,
                range_pts, range_pct, log_range, spx_return,
                vix_close, vix1d_close,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_date, ticker) DO UPDATE SET
                spx_open    = excluded.spx_open,
                spx_high    = excluded.spx_high,
                spx_low     = excluded.spx_low,
                spx_close   = excluded.spx_close,
                range_pts   = excluded.range_pts,
                range_pct   = excluded.range_pct,
                log_range   = excluded.log_range,
                spx_return  = excluded.spx_return,
                vix_close   = excluded.vix_close,
                vix1d_close = excluded.vix1d_close,
                updated_at  = excluded.updated_at
        """, (
            session_date.strftime("%Y-%m-%d"), ticker,
            _safe(row, "spx_open"),   _safe(row, "spx_high"),
            _safe(row, "spx_low"),    _safe(row, "spx_close"),
            _safe(row, "range_pts"),  _safe(row, "range_pct"),
            _safe(row, "log_range"),  _safe(row, "spx_return"),
            _safe(row, "vix_close"),  _safe(row, "vix1d_close"),
            now,
        ))
        rows_written += 1
    conn.commit()
    log.info(f"daily_spx: {rows_written} rows upserted")
    return rows_written


def get_daily_spx(conn, ticker: str = "SPX", limit: int = None) -> pd.DataFrame:
    """Return daily_spx for one ticker as a DataFrame indexed by session_date."""
    query = "SELECT * FROM daily_spx WHERE ticker = ? ORDER BY session_date ASC"
    params = (ticker,)
    if limit:
        query += " LIMIT ?"
        params = (ticker, int(limit))
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["session_date"])
    df.set_index("session_date", inplace=True)
    return df


def fetch_live_vix1d() -> float | None:
    """Return a single live ^VIX1D quote, or None on failure.

    No DB write — this is the intraday-refresh input for the 0DTE finder UI.
    Cached at the Streamlit fragment level by the caller.
    """
    try:
        h = yf.Ticker("^VIX1D").history(period="5d")
        if h.empty:
            return None
        return float(h["Close"].dropna().iloc[-1])
    except Exception as e:
        log.warning(f"fetch_live_vix1d failed: {e}")
        return None


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(conn) -> None:
    """Print a quick data health summary to console."""
    cur = conn.cursor()

    spx_count = cur.execute("SELECT COUNT(*) FROM weekly_spx").fetchone()[0]
    spx_range = cur.execute(
        "SELECT MIN(week_start), MAX(week_start) FROM weekly_spx"
    ).fetchone()

    macro_count = cur.execute("SELECT COUNT(*) FROM macro_daily").fetchone()[0]
    event_count = cur.execute(
        "SELECT SUM(event_count) FROM event_flags"
    ).fetchone()[0]

    print("\n" + "=" * 55)
    print("  DATA COLLECTOR — DATABASE SUMMARY")
    print("=" * 55)
    print(f"  weekly_spx  : {spx_count:>5} rows  ({spx_range[0]} → {spx_range[1]})")
    print(f"  macro_daily : {macro_count:>5} rows")
    print(f"  event_flags : {event_count:>5} total event-weeks flagged")
    print("=" * 55 + "\n")
