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

    # SPX weekly bars: Tradier primary (authenticated broker API), yfinance
    # fallback — see bar_sources for the validation + fallback logic.
    log.info(f"Fetching SPX weekly OHLC from {start.date()} to {end.date()}")
    from range_finder.bar_sources import fetch_weekly_bars
    spx_bars = fetch_weekly_bars("^GSPC", "SPX", years, "SPX")
    if spx_bars.empty:
        raise RuntimeError("no weekly SPX bars from any source — "
                           "market may be closed or network issue")
    spx = spx_bars[["open", "high", "low", "close", "volume"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close", "spx_volume"]

    # VIX weekly bars: yfinance seeds the frame; the Cboe overlay below is
    # the primary source. An empty yfinance response degrades to all-NaN
    # columns for Cboe to fill rather than killing the whole fetch.
    log.info(f"Fetching VIX weekly OHLC from {start.date()} to {end.date()}")
    vix_raw = yf.download("^VIX", start=start, end=end, interval="1wk", progress=False, timeout=60)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    vix_cols = ["vix_open", "vix_high", "vix_low", "vix_close"]
    if not vix_raw.empty and "Close" in vix_raw.columns:
        vix = vix_raw[["Open", "High", "Low", "Close"]].copy()
        vix.columns = vix_cols
        vix.index = pd.to_datetime(vix.index).normalize()
    else:
        log.warning("yfinance VIX weekly returned empty — relying on Cboe overlay")
        vix = pd.DataFrame(columns=vix_cols)

    # Merge on date index (left join: SPX bars define the weeks; VIX columns
    # may be NaN until the Cboe overlay fills them)
    df = spx.join(vix, how="left")
    df.index.name = "week_start"
    df.index = pd.to_datetime(df.index).normalize()   # strip time component

    # Overlay official Cboe VIX weekly bars (primary source; the publisher's
    # own data). The yfinance values above remain as the seam-filler for the
    # in-progress week and as the fallback when the Cboe CDN is unreachable.
    from range_finder.cboe_data import merge_cboe_weekly_ohlc
    df = merge_cboe_weekly_ohlc(df, "VIX", {
        "open": "vix_open", "high": "vix_high",
        "low": "vix_low",   "close": "vix_close",
    })

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

    # Underlying weekly bars: Tradier primary, yfinance fallback. The app
    # ticker IS the Tradier symbol for everything that reaches this fetcher
    # (QQQ/AMZN/AMD/... — SPX/XSP ride weekly_spx, and dynamic tickers are
    # validated against Tradier before they're added).
    log.info(f"[{ticker}] Fetching weekly OHLC ({yf_symbol}) {start.date()} → {end.date()}")
    from range_finder.bar_sources import fetch_weekly_bars
    base = fetch_weekly_bars(yf_symbol, ticker, years, ticker)
    if base.empty:
        raise RuntimeError(
            f"no weekly bars for {yf_symbol} ({ticker}) from any source — "
            "market closed, network issue, or unsupported symbol."
        )
    base = base[["open", "high", "low", "close", "volume"]].copy()

    log.info(f"[{ticker}] Fetching weekly vol proxy ({vol_proxy_yf})")
    vp_raw = yf.download(vol_proxy_yf, start=start, end=end, interval="1wk",
                         progress=False, timeout=60)
    if isinstance(vp_raw.columns, pd.MultiIndex):
        vp_raw.columns = vp_raw.columns.get_level_values(0)

    if not vp_raw.empty:
        vp = vp_raw[["Open", "High", "Low", "Close"]].copy()
        vp.columns = ["vol_proxy_open", "vol_proxy_high", "vol_proxy_low", "vol_proxy_close"]
        # base comes from bar_sources pre-normalized — align vp before joining
        vp.index = pd.to_datetime(vp.index).normalize()
        df = base.join(vp, how="left")
    else:
        log.warning(f"[{ticker}] vol proxy {vol_proxy_yf} returned empty — vol_proxy_* will be NaN")
        for col in ("vol_proxy_open", "vol_proxy_high", "vol_proxy_low", "vol_proxy_close"):
            base[col] = pd.NA
        df = base

    df.index.name = "week_start"
    df.index = pd.to_datetime(df.index).normalize()

    # Overlay official Cboe weekly bars for the vol proxy where one exists
    # (^VIX -> VIX, ^VXN -> VXN, ...). Note VXN does NOT quote on Tradier, so
    # for QQQ/NDX this Cboe series is the only non-yfinance source. yfinance
    # values above remain the seam-filler / fallback.
    from range_finder.cboe_data import YF_TO_CBOE_INDEX, merge_cboe_weekly_ohlc
    cboe_index = YF_TO_CBOE_INDEX.get(vol_proxy_yf)
    if cboe_index:
        df = merge_cboe_weekly_ohlc(df, cboe_index, {
            "open": "vol_proxy_open", "high": "vol_proxy_high",
            "low": "vol_proxy_low",   "close": "vol_proxy_close",
        })

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
# MONDAY ANCHOR  (weekly_setup freeze — shared by the Monday cron and the
# UI mid-week self-heal)
# =============================================================================
# The weekly Spread Finder anchors every strike to Monday's daily-candle Open
# (and the vol-proxy's Open as the frozen VIX). The Monday 9:30 ET cron captures
# this for `today`; the UI calls the same code to retroactively capture *this
# week's Monday* when the cron didn't run, so mid-week views stay locked instead
# of drifting on live spot.

def _daily_open_on(symbol: str, target_date) -> "float | None":
    """Daily-candle Open for `symbol` on `target_date`, or None.

    Scans a short recent window (yfinance ``period="5d"``) for the bar whose
    date matches `target_date` — matching on ``.date()`` dodges tz/DST edges.
    Returns None when that bar isn't published yet or the fetch fails. On
    Tue-Thu the week's Monday is 1-3 sessions back, well inside the 5-day window.
    """
    try:
        hist = yf.Ticker(symbol).history(period="5d")
    except Exception as e:
        log.warning(f"yfinance daily-open fetch failed for {symbol}: {e}")
        return None
    if hist is None or hist.empty or "Open" not in hist.columns:
        return None
    for ts, row in hist.iterrows():
        if hasattr(ts, "date") and ts.date() == target_date:
            op = row.get("Open")
            if op is not None and not (isinstance(op, float) and op != op):
                return float(op)
    return None


def save_weekly_setup(conn, ticker: str, week_start: str,
                      monday_open: float, monday_vix: float) -> None:
    """Upsert the frozen Monday open + VIX for (week_start, ticker)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO weekly_setup (week_start, ticker, monday_open, monday_vix, captured_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (week_start, ticker) DO UPDATE SET
            monday_open = excluded.monday_open,
            monday_vix  = excluded.monday_vix,
            captured_at = excluded.captured_at
    """, (week_start, ticker, monday_open, monday_vix, now_iso))
    conn.commit()


def capture_and_save_monday_anchor(
    conn,
    ticker: str,
    week_start: str,
    target_date,
    spot_fallback: "float | None" = None,
    live_vix_fallback: "float | None" = None,
    cfg: "dict | None" = None,
) -> "tuple[float, float, str]":
    """Capture `ticker`'s Monday daily-candle Open + vol-proxy Open and persist
    them to ``weekly_setup``. Returns ``(monday_open, monday_vix, open_source)``.

    `target_date` is the session whose daily Open is the weekly anchor — "today"
    when the Monday cron runs, or "this week's Monday" when the UI self-heals a
    missing capture mid-week. The underlying's Open is scaled by
    ``price_scale_divisor(ticker)`` (XSP→SPX rides ^SPX /10); the vol proxy
    (^VIX for most, ^VXN for QQQ/NDX) supplies the frozen VIX.

    `cfg` lets a caller pass an already-resolved config (the UI's chain-derived
    ``resolve_config`` for arbitrary tickers); when omitted we resolve via
    ``get_config``. Falls back to `spot_fallback` / `live_vix_fallback` only when
    yfinance has no daily bar for `target_date`.
    """
    from phase1.ticker_config import get_config, price_scale_divisor

    cfg = cfg or get_config(ticker)
    underlying_symbol = cfg.get("yf_symbol", "^GSPC")
    vol_proxy_symbol = cfg.get("vol_proxy_yf", "^VIX")
    scale = price_scale_divisor(ticker)

    underlying_open = _daily_open_on(underlying_symbol, target_date)
    if underlying_open is not None:
        monday_open = round(underlying_open / scale, 2)
        open_source = f"{underlying_symbol} daily Open ({target_date})"
    elif spot_fallback is not None:
        monday_open = round(float(spot_fallback), 2)
        open_source = "live spot (daily bar unavailable)"
    else:
        raise RuntimeError(
            f"No daily Open for {underlying_symbol} on {target_date} "
            "and no spot fallback supplied"
        )

    vol_proxy_open = _daily_open_on(vol_proxy_symbol, target_date)
    if vol_proxy_open is not None:
        monday_vix = round(vol_proxy_open, 2)
        vix_source = f"{vol_proxy_symbol} daily Open ({target_date})"
    elif live_vix_fallback is not None:
        monday_vix = round(float(live_vix_fallback), 2)
        vix_source = f"live {vol_proxy_symbol} (daily Open unavailable)"
    else:
        monday_vix = 18.0
        vix_source = "default 18.0 (no vol-proxy data)"

    save_weekly_setup(conn, ticker, week_start, monday_open, monday_vix)
    log.info(
        f"[{ticker}] Monday anchor saved for {week_start}: "
        f"open={monday_open:.2f} ({open_source}), vix={monday_vix:.2f} ({vix_source})"
    )
    return monday_open, monday_vix, open_source


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

    # SPX daily bars: Tradier primary, yfinance fallback (bar_sources).
    log.info(f"Fetching daily SPX OHLC from {start.date()} to {end.date()}")
    from range_finder.bar_sources import fetch_daily_bars
    spx_bars = fetch_daily_bars("^GSPC", "SPX", years, "SPX")
    if spx_bars.empty:
        raise RuntimeError("no daily SPX bars from any source — "
                           "market may be closed or network issue")
    spx = spx_bars[["open", "high", "low", "close"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close"]

    # VIX / VIX1D daily closes: yfinance seeds the frame; the Cboe overlay
    # below is the primary source and fills anything yfinance missed.
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

    vix_raw = _flatten(vix_raw)
    vix1d_raw = _flatten(vix1d_raw)

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

    # spx comes from bar_sources pre-normalized — align the yfinance frames
    # before joining so dates actually match.
    for frame in (vix, vix1d):
        if not frame.empty:
            frame.index = pd.to_datetime(frame.index).normalize()
    df = spx.join(vix, how="left")
    df = df.join(vix1d, how="left")
    df.index.name = "session_date"
    df.index = pd.to_datetime(df.index).normalize()

    # Overlay official Cboe closes for VIX and VIX1D (primary source).
    # VIX1D is the one that matters most: yfinance's ^VIX1D series only starts
    # at the 2023-04-24 index launch, but Cboe published reconstructed values
    # back to 2022-05-13 — this fills that gap AND wins over yfinance wherever
    # both have the session. The yfinance values above remain as the
    # seam-filler (e.g. today's row before Cboe posts EOD) and as the fallback
    # when the Cboe CDN is unreachable.
    from range_finder.cboe_data import merge_cboe_closes
    df = merge_cboe_closes(df, {"vix_close": "VIX", "vix1d_close": "VIX1D"})

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


def _tradier_token() -> str:
    """Resolve the Tradier token from Streamlit secrets or environment
    (same pattern as FRED_API_KEY above). Empty string when unset."""
    token = ""
    try:
        import streamlit as st
        token = st.secrets.get("TRADIER_TOKEN", "")
    except Exception:
        pass
    return token or os.environ.get("TRADIER_TOKEN", "")


def fetch_live_vix1d() -> float | None:
    """Return a single live VIX1D quote, or None on failure.

    Tradier /markets/quotes is the primary source (verified: VIX1D quotes
    live there; it's the app's authenticated broker API, unlike the yfinance
    scraper). yfinance remains the fallback for when the token is missing or
    Tradier has an outage.

    No DB write — this is the intraday-refresh input for the 0DTE finder UI.
    Cached at the Streamlit fragment level by the caller.
    """
    token = _tradier_token()
    if token:
        try:
            from phase1.data_client import TradierDataClient
            spot = TradierDataClient(token).get_spot_price("VIX1D")
            if spot and spot > 0:
                return float(spot)
            log.warning("Tradier VIX1D quote empty — falling back to yfinance")
        except Exception as e:
            log.warning(f"Tradier VIX1D quote failed: {e} — falling back to yfinance")

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
    """Print a quick data health summary to console.

    Uses separate execute() + fetchone() calls — psycopg2's cursor.execute()
    returns None (unlike sqlite3's, which returns the cursor), so the old
    chained ``cur.execute(...).fetchone()`` pattern AttributeErrors on
    Postgres. The Postgres migration removed sqlite without auditing this
    summary path, and it stayed latent until the 0DTE bootstrap exercised
    it on a Postgres deploy.
    """
    def _scalar(sql: str, default=0):
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return (row[0] if row and row[0] is not None else default)

    def _pair(sql: str):
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone() or (None, None)
        return (row[0] or "—", row[1] or "—")

    weekly_count           = _scalar("SELECT COUNT(*) FROM weekly_spx")
    weekly_min, weekly_max = _pair("SELECT MIN(week_start), MAX(week_start) FROM weekly_spx")
    macro_count            = _scalar("SELECT COUNT(*) FROM macro_daily")
    event_count            = _scalar("SELECT SUM(event_count) FROM event_flags")

    # 0DTE / daily tables. Wrapped in try/except so an older deploy that
    # hasn't run init_all_tables() against the new schema yet doesn't crash
    # the whole summary — it just shows "—" rows for the new tables.
    try:
        daily_count            = _scalar("SELECT COUNT(*) FROM daily_spx")
        daily_min, daily_max   = _pair("SELECT MIN(session_date), MAX(session_date) FROM daily_spx")
        daily_feat_count       = _scalar("SELECT COUNT(*) FROM daily_model_features")
        daily_event_count      = _scalar("SELECT SUM(event_count) FROM event_flags_daily")
    except Exception:
        daily_count = daily_feat_count = daily_event_count = 0
        daily_min = daily_max = "—"

    print("\n" + "=" * 60)
    print("  DATA COLLECTOR — DATABASE SUMMARY")
    print("=" * 60)
    print(f"  weekly_spx           : {weekly_count:>5} rows  ({weekly_min} → {weekly_max})")
    print(f"  macro_daily          : {macro_count:>5} rows")
    print(f"  event_flags          : {event_count:>5} total event-weeks flagged")
    print(f"  daily_spx            : {daily_count:>5} rows  ({daily_min} → {daily_max})")
    print(f"  daily_model_features : {daily_feat_count:>5} rows")
    print(f"  event_flags_daily    : {daily_event_count:>5} total event-days flagged")
    print("=" * 60 + "\n")
