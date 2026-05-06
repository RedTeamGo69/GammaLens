# =============================================================================
# feature_builder.py
# Weekly SPX Range Prediction Model — Feature Engineering Module
#
# Reads raw data from Postgres (populated by data_collector.py) and produces
# a clean, model-ready feature matrix saved to the `model_features` table.
# =============================================================================

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from range_finder.data_collector import (
    get_weekly_spx,
    get_macro_daily,
    get_event_flags,
    init_db,
)

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
# DATABASE — add model_features table
# =============================================================================

def init_features_table(conn) -> None:
    """
    Ensure the model_features table exists.
    Now handled by db.init_all_tables() — this is kept for backwards compatibility.
    """
    pass  # Tables created in db.init_all_tables()


# =============================================================================
# DAILY SPX — needed for HV calculation
# =============================================================================

def fetch_daily_spx(years: int = 6) -> pd.DataFrame:
    """Pull daily SPX closes from yfinance for HV calculation."""
    return fetch_daily_underlying("^GSPC", years=years, label="SPX",
                                   close_col="spx_close")


def fetch_daily_underlying(yf_symbol: str, years: int = 6,
                            label: str = None,
                            close_col: str = "spx_close") -> pd.DataFrame:
    """Pull daily closes for any underlying from yfinance.

    Returns a DataFrame indexed by date with one column whose name defaults to
    ``spx_close`` so the existing ``compute_hv_windows`` code path works
    unchanged on any ticker. Pass a different ``close_col`` if you need to
    distinguish multiple series in the same DataFrame.
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    label = label or yf_symbol
    log.info(f"Fetching daily {label} closes for HV calculation...")
    raw = yf.download(yf_symbol, start=start, end=end, interval="1d", progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw.empty or "Close" not in raw.columns:
        log.warning(f"Daily {label}: yfinance returned empty data")
        return pd.DataFrame(columns=[close_col])

    df = raw[["Close"]].copy()
    df.columns = [close_col]
    df.index = pd.to_datetime(df.index).normalize()
    df.dropna(inplace=True)

    log.info(f"Daily {label}: {len(df)} rows")
    return df


def compute_hv_windows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling historical volatility (annualized) from daily log returns.
    HV formula: std(log_returns, window) * sqrt(252)

    Operates on the first column of the DataFrame regardless of name, so it
    works for SPX (spx_close), QQQ (close), or any single-ticker daily series.
    """
    df = daily_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["hv5", "hv10", "hv20"])
    close_col = df.columns[0]
    df["log_ret"] = np.log(df[close_col] / df[close_col].shift(1))

    df["hv5"]  = df["log_ret"].rolling(5,  min_periods=4).std()  * math.sqrt(252)
    df["hv10"] = df["log_ret"].rolling(10, min_periods=8).std()  * math.sqrt(252)
    df["hv20"] = df["log_ret"].rolling(20, min_periods=15).std() * math.sqrt(252)

    # Resample to weekly — take the LAST value of each week (Friday close HV)
    weekly_hv = df[["hv5", "hv10", "hv20"]].resample("W-FRI").last()
    weekly_hv.index = weekly_hv.index - pd.offsets.Week(weekday=0)  # shift to Monday
    weekly_hv.index.name = "week_start"

    log.info(f"HV windows computed: {len(weekly_hv)} weekly rows")
    return weekly_hv


# =============================================================================
# VIX TERM STRUCTURE
# =============================================================================

def fetch_vix_term_structure(years: int = 6) -> pd.DataFrame:
    """Pull weekly closes for VIX9D and VIX3M from yfinance."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    log.info("Fetching VIX9D and VIX3M for term structure...")

    vix9d_raw = yf.download("^VIX9D", start=start, end=end, interval="1wk", progress=False)
    vix3m_raw = yf.download("^VIX3M", start=start, end=end, interval="1wk", progress=False)

    def extract_close(raw, name):
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            log.warning(f"{name} returned empty — will be NULL in features")
            return pd.Series(dtype=float, name=name)
        s = raw["Close"].copy()
        s.name = name
        s.index = pd.to_datetime(s.index).normalize()
        return s

    vix9d = extract_close(vix9d_raw, "vix9d_close")
    vix3m = extract_close(vix3m_raw, "vix3m_close")

    df = pd.DataFrame({"vix9d_close": vix9d, "vix3m_close": vix3m})
    df.index.name = "week_start"

    df["vix_ts_slope"] = df["vix3m_close"] - df["vix9d_close"]

    log.info(f"VIX term structure: {len(df)} weekly rows")
    return df


# =============================================================================
# MACRO — resample daily FRED to weekly
# =============================================================================

def resample_macro_to_weekly(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily FRED data to weekly frequency (Friday → Monday index)."""
    weekly = macro_df[["yield_spread", "fed_funds"]].resample("W-FRI").last()
    weekly.index = weekly.index - pd.offsets.Week(weekday=0)
    weekly.index.name = "week_start"
    weekly.ffill(inplace=True)
    return weekly


# =============================================================================
# HAR FEATURE COMPUTATION
# =============================================================================

def compute_har_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three HAR components from the range_pct series.

    Follows the canonical Corsi (2009) HAR-RV structure adapted for weekly
    data: each component ends at lag 1 so the forecast at week t uses only
    information known at the close of week t-1. The components OVERLAP by
    design — this is how standard HAR captures vol cascade across horizons.

    har_d1 — range_pct at lag 1                  (prior week)
    har_w  — mean of lags 1..5   (rolling 5)     (~1 month of weeks)
    har_m  — mean of lags 1..20  (rolling 20)    (~5 months of weeks)
    """
    df = weekly_df[["range_pct"]].copy()

    # shift(1) ensures the lag-1 value is the most recent observation used.
    lagged = df["range_pct"].shift(1)

    df["har_d1"] = lagged

    df["har_w"] = lagged.rolling(5, min_periods=3).mean()

    df["har_m"] = lagged.rolling(20, min_periods=10).mean()

    return df[["har_d1", "har_w", "har_m"]]


# =============================================================================
# GEX PLACEHOLDER
# =============================================================================

def load_gex_inputs(conn, ticker: str = "SPX") -> pd.DataFrame:
    """Load GEX values from the gex_inputs table if it exists.

    Filters by ticker so the HAR feature builder only sees rows that
    match the underlying it was trained on. The feature builder
    normalizes by SPX's spx_open² (see build_features), so mixing XSP
    rows into the input here would produce a ~100× scale mismatch on
    gex_normalized for any week that XSP wrote last.
    """
    try:
        df = pd.read_sql_query(
            "SELECT week_start, gex FROM gex_inputs WHERE ticker = ? "
            "ORDER BY week_start ASC",
            conn,
            params=(ticker,),
            parse_dates=["week_start"],
        )
        df.set_index("week_start", inplace=True)
        log.info(f"GEX inputs loaded: {len(df)} rows (ticker={ticker})")
        return df
    except Exception:
        log.info("gex_inputs table not found — GEX features will be NULL")
        return pd.DataFrame(columns=["gex"])


def create_gex_table(conn) -> None:
    """Ensure the gex_inputs table exists.
    Now handled by db.init_all_tables() — kept for backwards compatibility."""
    pass  # Tables created in db.init_all_tables()


def upsert_gex(conn, week_start: str, gex: float, notes: str = "",
               ticker: str = "SPX") -> None:
    """Insert or update a single GEX value for a given week and ticker."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO gex_inputs (week_start, ticker, gex, notes, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(week_start, ticker) DO UPDATE SET
            gex        = excluded.gex,
            notes      = excluded.notes,
            updated_at = excluded.updated_at
    """, (week_start, ticker, gex, notes, now))
    conn.commit()
    log.info(f"GEX upserted: {week_start} {ticker} → {gex:,.0f}")


# =============================================================================
# MAIN FEATURE BUILD
# =============================================================================

def _load_weekly_for_ticker(conn, ticker: str) -> pd.DataFrame:
    """Return weekly OHLC + vol-proxy data for a ticker, normalised to the
    legacy SPX-style column names (``spx_open``, ``vix_close``, ``spx_return``,
    etc.) so the downstream feature pipeline can stay column-name-stable
    across SPX and the own-HAR tickers (QQQ / AMZN / AMD).

    SPX/XSP read ``weekly_spx`` (the existing table). Own-HAR tickers read
    ``weekly_underlying`` (per-ticker schema). The vol-proxy column is VIX
    for SPX/stocks and VXN for QQQ — see phase1.ticker_config.vol_proxy_yf.
    """
    from phase1.ticker_config import uses_own_har
    if not uses_own_har(ticker):
        return get_weekly_spx(conn)

    # Per-ticker path: read weekly_underlying and rename to legacy column
    # names so the downstream pipeline stays unchanged.
    from range_finder.data_collector import get_weekly_underlying
    raw = get_weekly_underlying(conn, ticker=ticker)
    if raw.empty:
        log.warning(f"weekly_underlying empty for {ticker}")
        return raw
    return raw.rename(columns={
        "open":            "spx_open",
        "high":            "spx_high",
        "low":             "spx_low",
        "close":           "spx_close",
        "volume":          "spx_volume",
        "vol_proxy_open":  "vix_open",
        "vol_proxy_high":  "vix_high",
        "vol_proxy_low":   "vix_low",
        "vol_proxy_close": "vix_close",
        "return_pct":      "spx_return",
    })


def _load_daily_for_ticker(ticker: str, years: int = 6) -> pd.DataFrame:
    """Return daily closes for HV computation for one ticker.

    Returns a DataFrame with column ``spx_close`` regardless of ticker so
    ``compute_hv_windows`` works unchanged.
    """
    from phase1.ticker_config import get_config, uses_own_har
    if not uses_own_har(ticker):
        return fetch_daily_spx(years=years)
    cfg = get_config(ticker)
    return fetch_daily_underlying(cfg["yf_symbol"], years=years, label=ticker,
                                   close_col="spx_close")


def _load_earnings_flags(conn, ticker: str) -> pd.DataFrame:
    """Return earnings_flags rows for a ticker, indexed by week_start.

    Single-stock tickers (AMZN/AMD) have rows here; index/ETF tickers don't,
    so the join produces all-zero ``has_earnings`` columns.
    """
    try:
        df = pd.read_sql_query(
            "SELECT week_start, has_earnings FROM earnings_flags WHERE ticker = ?",
            conn,
            params=(ticker,),
            parse_dates=["week_start"],
        )
    except Exception:
        return pd.DataFrame(columns=["has_earnings"])
    if df.empty:
        return pd.DataFrame(columns=["has_earnings"])
    df.set_index("week_start", inplace=True)
    return df


def build_features(conn, exclude_covid: bool = True,
                    ticker: str = "SPX") -> pd.DataFrame:
    """
    Assemble the full model-ready feature matrix and save to model_features.
    Every feature is lagged so that at row t (target week), you only
    use information observable at Friday close of week t-1.

    For SPX (and XSP, which shares SPX features) this is the original
    SPX/VIX pipeline. For own-HAR tickers (QQQ / AMZN / AMD) the weekly OHLC
    and vol-proxy come from the per-ticker ``weekly_underlying`` table; the
    rest of the pipeline (HAR, HV, macro, events, GEX normalisation) is
    column-name-stable thanks to ``_load_weekly_for_ticker`` renaming on the way in.
    """
    log.info(f"Building feature matrix for {ticker}...")

    # --- Load base data (per-ticker for own-HAR; SPX/XSP share SPX rows) ---
    weekly  = _load_weekly_for_ticker(conn, ticker)
    macro   = get_macro_daily(conn)
    events  = get_event_flags(conn)
    earnings = _load_earnings_flags(conn, ticker)

    if weekly.empty:
        log.warning(f"weekly OHLC empty for {ticker} — skipping feature build")
        return pd.DataFrame()

    # --- Fetch supplemental data ---
    daily_underlying = _load_daily_for_ticker(ticker, years=6)
    # VIX9D/VIX3M term structure stays VIX-anchored across all tickers — it's
    # a macro-vol regime feature that single names also covary with. yfinance
    # has VIX9D/VIX3M for the full feature window.
    vix_ts    = fetch_vix_term_structure(years=6)

    # --- HAR components ---
    har = compute_har_features(weekly)

    # --- HV windows ---
    hv = compute_hv_windows(daily_underlying)

    # --- Macro weekly ---
    macro_wk = resample_macro_to_weekly(macro)

    # --- GEX (per-ticker — see save_gex_to_range_finder writes per ticker) ---
    gex_df = load_gex_inputs(conn, ticker=ticker)

    # --- Vol-proxy implied range ---
    # Vol proxy is annualized 1-SD vol in percent (VIX for SPX/stocks, VXN for
    # QQQ). De-annualize by sqrt(52) for the weekly 1-SD, then scale by the
    # Brownian high-low range factor E[H-L] = 2·sigma·sqrt(2/pi) ≈ 1.5958·sigma
    # (Feller 1951 / Parkinson 1980) so the column is directly comparable to
    # realized range_pct = (H-L)/open. Previously this was just 1-SD weekly vol
    # (biased ~37% low as a range predictor); the fix rescales it so UI
    # comparisons like model_vs_vix and the "trust the model" warning in
    # spread_levels.py stop firing spuriously on nearly every week. OLS is
    # scale-invariant so re-fitting the HAR model just rescales its coefficient
    # on this feature — predictive power is unchanged.
    _BM_RANGE_FACTOR = 2.0 * math.sqrt(2.0 / math.pi)
    weekly["vix_implied_range"] = (
        (weekly["vix_close"] / math.sqrt(52)) / 100 * _BM_RANGE_FACTOR
    )

    # --- Underlying return lags (column kept as spx_return_lag1 for SQL stability) ---
    weekly["spx_return_lag1"] = weekly["spx_return"].shift(1)
    weekly["abs_return_lag1"] = weekly["spx_return_lag1"].abs()

    # --- Assemble ---
    df = weekly[[
        "range_pct", "log_range",
        "vix_close", "vix_implied_range",
        "spx_return_lag1", "abs_return_lag1",
    ]].copy()

    # Vol-proxy close needs to be lagged — we observe prior Friday's value
    df["vix_close"]         = df["vix_close"].shift(1)
    df["vix_implied_range"] = df["vix_implied_range"].shift(1)

    # Join HAR
    df = df.join(har, how="left")

    # Join HV (already on Monday index)
    hv_lagged = hv.shift(1)
    df = df.join(hv_lagged, how="left")
    df["hv_ratio"] = df["hv5"] / df["hv20"]

    # --- High-vol regime detection ---
    # Binary flag: 1 if trailing 4-week average vol-proxy > 20
    df["high_vol_regime"] = (df["vix_close"].rolling(4, min_periods=2).mean() > 20).astype(int)

    # Join VIX term structure (lag 1 week)
    vix_ts_lagged = vix_ts.shift(1)
    df = df.join(vix_ts_lagged, how="left")
    df["vix_wk_ratio"] = df["vix_close"] / df["vix3m_close"]

    # Join macro (lag 1 week)
    macro_lagged = macro_wk.shift(1)
    df = df.join(macro_lagged, how="left")

    # Join GEX (Monday open — same week, no lag needed). Per-ticker filter
    # applied above means each ticker's GEX is normalised by its OWN open²
    # (cancelling the spot² scale embedded in the GEX formula at
    # gex_engine.py:174). Without this cancellation, time-series of
    # gex_normalized would track the level of the underlying rather than the
    # positioning signal — and would also be incomparable across tickers.
    if not gex_df.empty:
        df = df.join(gex_df[["gex"]], how="left")
        df["gex_flag"] = df["gex"].apply(_gex_flag)
        df["gex_normalized"] = df["gex"] / (weekly["spx_open"] ** 2) * 1e4
    else:
        df["gex"]            = np.nan
        df["gex_flag"]       = np.nan
        df["gex_normalized"] = np.nan

    # Join event flags
    df = df.join(
        events[["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]],
        how="left"
    )
    for col in ["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # Join single-stock earnings flag (only AMZN/AMD ever populates it)
    if not earnings.empty:
        df = df.join(earnings[["has_earnings"]], how="left")
    else:
        df["has_earnings"] = 0
    df["has_earnings"] = df["has_earnings"].fillna(0).astype(int)

    # Drop rows with insufficient lag history
    df.dropna(subset=["har_d1", "har_w", "har_m", "vix_close", "log_range"], inplace=True)

    # Exclude COVID crash period if requested (default True)
    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2020-09-30")
        pre_filter = len(df)
        df = df[(df.index < covid_start) | (df.index > covid_end)]
        removed = pre_filter - len(df)
        if removed:
            log.info(f"exclude_covid: removed {removed} rows (2020-03-01 to 2020-09-30)")

    log.info(f"Feature matrix ({ticker}): {len(df)} rows x {len(df.columns)} columns")

    # --- Save to DB ---
    _save_features(conn, df, ticker=ticker)

    return df


def _gex_flag(gex_val) -> int | None:
    """
    Classify GEX into regime: +1 positive, 0 neutral, -1 negative.

    DEPRECATED: This binary flag is superseded by the continuous gex_normalized
    feature. Kept only for backward compatibility with existing DB rows.
    New code should use gex_normalized directly.
    """
    if gex_val is None or (isinstance(gex_val, float) and math.isnan(gex_val)):
        return None
    # Note: this threshold is approximate and not calibrated to Spot²-scaled GEX.
    # It exists only for legacy rows. The HAR model uses gex_normalized instead.
    if gex_val > 0:
        return 1
    elif gex_val < 0:
        return -1
    return 0


def _save_features(conn, df: pd.DataFrame, ticker: str = "SPX") -> None:
    """Upsert the feature matrix into model_features for a given ticker.

    `has_earnings` is read from the dataframe if present (for single-name
    stocks) and falls back to 0 otherwise — index/ETF tickers leave it 0.
    """
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    rows = 0

    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO model_features (
                week_start, ticker,
                log_range, range_pct,
                har_d1, har_w, har_m,
                vix_close, vix_implied_range,
                vix9d_close, vix3m_close, vix_ts_slope, vix_wk_ratio,
                hv5, hv10, hv20, hv_ratio,
                high_vol_regime,
                gex, gex_flag, gex_normalized,
                yield_spread, fed_funds,
                spx_return_lag1, abs_return_lag1,
                has_fomc, has_cpi, has_nfp, has_opex, event_count,
                has_earnings,
                updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(week_start, ticker) DO UPDATE SET
                log_range           = excluded.log_range,
                range_pct           = excluded.range_pct,
                har_d1              = excluded.har_d1,
                har_w               = excluded.har_w,
                har_m               = excluded.har_m,
                vix_close           = excluded.vix_close,
                vix_implied_range   = excluded.vix_implied_range,
                vix9d_close         = excluded.vix9d_close,
                vix3m_close         = excluded.vix3m_close,
                vix_ts_slope        = excluded.vix_ts_slope,
                vix_wk_ratio        = excluded.vix_wk_ratio,
                hv5                 = excluded.hv5,
                hv10                = excluded.hv10,
                hv20                = excluded.hv20,
                hv_ratio            = excluded.hv_ratio,
                high_vol_regime     = excluded.high_vol_regime,
                gex                 = excluded.gex,
                gex_flag            = excluded.gex_flag,
                gex_normalized      = excluded.gex_normalized,
                yield_spread        = excluded.yield_spread,
                fed_funds           = excluded.fed_funds,
                spx_return_lag1     = excluded.spx_return_lag1,
                abs_return_lag1     = excluded.abs_return_lag1,
                has_fomc            = excluded.has_fomc,
                has_cpi             = excluded.has_cpi,
                has_nfp             = excluded.has_nfp,
                has_opex            = excluded.has_opex,
                event_count         = excluded.event_count,
                has_earnings        = excluded.has_earnings,
                updated_at          = excluded.updated_at
        """, (
            week_start.strftime("%Y-%m-%d"), ticker,
            _f(row, "log_range"),        _f(row, "range_pct"),
            _f(row, "har_d1"),           _f(row, "har_w"),           _f(row, "har_m"),
            _f(row, "vix_close"),        _f(row, "vix_implied_range"),
            _f(row, "vix9d_close"),      _f(row, "vix3m_close"),
            _f(row, "vix_ts_slope"),     _f(row, "vix_wk_ratio"),
            _f(row, "hv5"),              _f(row, "hv10"),            _f(row, "hv20"),
            _f(row, "hv_ratio"),
            _i(row, "high_vol_regime"),
            _f(row, "gex"),              _i(row, "gex_flag"),        _f(row, "gex_normalized"),
            _f(row, "yield_spread"),     _f(row, "fed_funds"),
            _f(row, "spx_return_lag1"),  _f(row, "abs_return_lag1"),
            _i(row, "has_fomc"),         _i(row, "has_cpi"),
            _i(row, "has_nfp"),          _i(row, "has_opex"),
            _i(row, "event_count"),
            _i(row, "has_earnings"),
            now,
        ))
        rows += 1

    conn.commit()
    log.info(f"model_features ({ticker}): {rows} rows upserted")


# =============================================================================
# READERS
# =============================================================================

def get_features(conn, min_date: str = None, exclude_covid: bool = False,
                 ticker: str = "SPX") -> pd.DataFrame:
    """Load the model_features table for a given ticker as a DataFrame.

    Defaults to SPX so existing call sites that haven't been updated yet keep
    seeing the SPX feature matrix.
    """
    query = "SELECT * FROM model_features WHERE ticker = ? ORDER BY week_start ASC"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["week_start"])
    df.set_index("week_start", inplace=True)

    if min_date:
        df = df[df.index >= pd.to_datetime(min_date)]

    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2020-09-30")
        df = df[(df.index < covid_start) | (df.index > covid_end)]

    return df


def get_feature_for_week(conn, week_start: str, ticker: str = "SPX") -> pd.Series | None:
    """Fetch the feature row for a specific week + ticker."""
    df = pd.read_sql_query(
        "SELECT * FROM model_features WHERE week_start = ? AND ticker = ?",
        conn,
        params=(week_start, ticker),
        parse_dates=["week_start"],
    )
    if df.empty:
        log.warning(f"No feature row found for week_start={week_start} ticker={ticker}")
        return None
    df.set_index("week_start", inplace=True)
    return df.iloc[0]


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_feature_summary(df: pd.DataFrame) -> None:
    """Quick console summary of the feature matrix."""
    print("\n" + "=" * 65)
    print("  FEATURE BUILDER — FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"  Rows      : {len(df)}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Columns   : {len(df.columns)}")
    print()

    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if not null_pct.empty:
        print("  NULL % by column (non-zero only):")
        for col, pct in null_pct.items():
            print(f"    {col:<25} {pct:.1f}%")
    else:
        print("  No nulls in feature matrix.")

    print()
    print("  Target (log_range) stats:")
    print(df["log_range"].describe().to_string(float_format="{:.4f}".format))
    print("=" * 65 + "\n")


# =============================================================================
# UTILITY
# =============================================================================

def _f(row: pd.Series, col: str) -> float | None:
    """Safe float extractor."""
    val = row.get(col)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _i(row: pd.Series, col: str) -> int | None:
    """Safe int extractor."""
    val = _f(row, col)
    return None if val is None else int(val)
