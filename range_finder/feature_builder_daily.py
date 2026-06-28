"""
feature_builder_daily.py
Daily-cadence feature engineering for the 0DTE spread finder.

Mirror of ``feature_builder.py`` but indexed by ``session_date`` (daily)
instead of ``week_start`` (weekly). Reads daily OHLC + VIX + VIX1D from
``daily_spx`` and day-of event flags from ``event_flags_daily``.

SPX-only — the 0DTE finder supports SPX and XSP, but XSP reuses the SPX
forecast at inference time (XSP = SPX / 10 and they share ^VIX1D), so only
SPX features are built and persisted here.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from range_finder.data_collector import get_daily_spx

log = logging.getLogger(__name__)

# Same Brownian range factor as feature_builder.py's vix_implied_range — the
# expected high-low range of a Brownian motion with given sigma is
# 2 * sigma * sqrt(2 / pi). Used to convert VIX1D (1-SD vol) into an implied
# DAILY range that's directly comparable to realized daily range_pct.
_BM_RANGE_FACTOR = 2.0 * math.sqrt(2.0 / math.pi)


def compute_har_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute HAR components on daily range_pct.

    Same Corsi-2009 cascade structure as ``compute_har_features`` in the
    weekly builder, but the rolling windows are now in trading days:

        har_d1_daily — range_pct at lag 1                 (prior session)
        har_w_daily  — mean of lags 1..5  (rolling 5d)    (~1 trading week)
        har_m_daily  — mean of lags 1..20 (rolling 20d)   (~1 trading month)
    """
    df = daily_df[["range_pct"]].copy()
    lagged = df["range_pct"].shift(1)
    df["har_d1_daily"] = lagged
    df["har_w_daily"]  = lagged.rolling(5,  min_periods=3).mean()
    df["har_m_daily"]  = lagged.rolling(20, min_periods=10).mean()
    return df[["har_d1_daily", "har_w_daily", "har_m_daily"]]


def compute_hv5_daily(daily_df: pd.DataFrame) -> pd.Series:
    """5-day annualised historical vol from daily SPX log returns.

    Same formula as the weekly builder's ``compute_hv_windows`` but kept on
    the daily index (no resampling to weekly).
    """
    closes = daily_df["spx_close"]
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(5, min_periods=4).std() * math.sqrt(252)


def _load_event_flags_daily(conn) -> pd.DataFrame:
    """Return event_flags_daily indexed by session_date with renamed columns.

    Renames has_fomc/has_cpi/has_nfp to has_fomc_today/etc so the feature
    columns make their day-of meaning explicit downstream.
    """
    try:
        df = pd.read_sql_query(
            "SELECT * FROM event_flags_daily ORDER BY session_date ASC",
            conn,
            parse_dates=["session_date"],
        )
    except Exception:
        return pd.DataFrame(columns=["has_fomc_today", "has_cpi_today",
                                      "has_nfp_today", "event_count"])
    if df.empty:
        return pd.DataFrame(columns=["has_fomc_today", "has_cpi_today",
                                      "has_nfp_today", "event_count"])
    df.set_index("session_date", inplace=True)
    df.rename(columns={
        "has_fomc": "has_fomc_today",
        "has_cpi":  "has_cpi_today",
        "has_nfp":  "has_nfp_today",
    }, inplace=True)
    return df


def build_daily_features(conn, ticker: str = "SPX",
                          exclude_covid: bool = True) -> pd.DataFrame:
    """Assemble and persist the daily feature matrix.

    Only fits/persists SPX rows. XSP inference reuses the SPX-trained HAR
    fit (range_pct is scale-invariant; ``build_spread_plan`` scales to the
    passed-in spot).
    """
    if ticker != "SPX":
        log.warning(f"build_daily_features called with ticker={ticker} — "
                    f"only SPX is trained for 0DTE; XSP reuses the SPX fit.")

    log.info(f"Building daily feature matrix for {ticker}...")
    daily = get_daily_spx(conn, ticker=ticker)
    if daily.empty:
        log.warning(f"daily_spx empty for {ticker} — run bootstrap step 8 first")
        return pd.DataFrame()

    events = _load_event_flags_daily(conn)

    # --- HAR components ---
    har = compute_har_daily(daily)

    # --- 5-day HV (annualised) ---
    hv5 = compute_hv5_daily(daily)

    # --- VIX1D implied daily range ---
    # De-annualise by sqrt(252) for the daily 1-SD, then apply the Brownian
    # range factor so it's directly comparable to realised range_pct. Same
    # construction as feature_builder.py's vix_implied_range but on the
    # daily axis (sqrt(52) → sqrt(252)).
    daily["vix1d_implied_range"] = (
        (daily["vix1d_close"] / math.sqrt(252)) / 100 * _BM_RANGE_FACTOR
    )

    # --- Return lags ---
    daily["spx_return_lag1"] = daily["spx_return"].shift(1)
    daily["abs_return_lag1"] = daily["spx_return_lag1"].abs()

    # --- Assemble base columns ---
    df = daily[[
        "range_pct", "log_range",
        "vix_close", "vix1d_close", "vix1d_implied_range",
        "spx_return_lag1", "abs_return_lag1",
    ]].copy()

    # VIX and VIX1D need to be lagged — at session t (target) we observe
    # yesterday's close before today opens.
    df["vix_close"]           = df["vix_close"].shift(1)
    df["vix1d_close"]         = df["vix1d_close"].shift(1)
    df["vix1d_implied_range"] = df["vix1d_implied_range"].shift(1)

    # --- Join HAR (already shifted by shift(1) inside compute_har_daily) ---
    df = df.join(har, how="left")

    # --- Join HV5 (lag 1 — yesterday's HV is what we observe today) ---
    df["hv5"] = hv5.shift(1)

    # --- VRP daily: implied daily range minus prior-day realised ---
    # Both are unitless daily-range fractions. Positive ⇒ IV richer than RV ⇒
    # premium-selling has edge today. Persisted so the UI can read it without
    # recomputation.
    df["vrp_daily"] = df["vix1d_implied_range"] - df["har_d1_daily"]

    # --- Join event flags (no lag — today's events are today's events) ---
    if not events.empty:
        df = df.join(
            events[["has_fomc_today", "has_cpi_today", "has_nfp_today", "event_count"]],
            how="left",
        )
    else:
        df["has_fomc_today"] = 0
        df["has_cpi_today"]  = 0
        df["has_nfp_today"]  = 0
        df["event_count"]    = 0
    for col in ["has_fomc_today", "has_cpi_today", "has_nfp_today", "event_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # GEX is intraday — populated only at inference time from the live
    # GEXContext, never backfilled here.
    df["gex"] = np.nan
    df["gex_normalized"] = np.nan

    # Drop rows lacking sufficient lag history for HAR core. VIX1D is allowed
    # to be NaN (pre-2022 history) — M1_daily_baseline still trains; M2/M3
    # auto-skip those rows via feature_has_enough_data.
    df.dropna(subset=["har_d1_daily", "har_w_daily", "har_m_daily", "log_range"],
              inplace=True)

    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end   = pd.Timestamp("2020-09-30")
        pre_filter  = len(df)
        df = df[(df.index < covid_start) | (df.index > covid_end)]
        removed = pre_filter - len(df)
        if removed:
            log.info(f"exclude_covid: removed {removed} daily rows "
                     f"(2020-03-01 to 2020-09-30)")

    log.info(f"Daily feature matrix ({ticker}): "
             f"{len(df)} rows x {len(df.columns)} columns")

    _save_daily_features(conn, df, ticker=ticker)
    return df


def _save_daily_features(conn, df: pd.DataFrame, ticker: str = "SPX") -> None:
    """Upsert the daily feature matrix into daily_model_features."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    rows = 0
    for session_date, row in df.iterrows():
        cur.execute("""
            INSERT INTO daily_model_features (
                session_date, ticker,
                log_range, range_pct,
                har_d1_daily, har_w_daily, har_m_daily,
                vix_close, vix1d_close, vix1d_implied_range, vrp_daily,
                hv5,
                spx_return_lag1, abs_return_lag1,
                has_fomc_today, has_cpi_today, has_nfp_today, event_count,
                gex, gex_normalized,
                updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(session_date, ticker) DO UPDATE SET
                log_range            = excluded.log_range,
                range_pct            = excluded.range_pct,
                har_d1_daily         = excluded.har_d1_daily,
                har_w_daily          = excluded.har_w_daily,
                har_m_daily          = excluded.har_m_daily,
                vix_close            = excluded.vix_close,
                vix1d_close          = excluded.vix1d_close,
                vix1d_implied_range  = excluded.vix1d_implied_range,
                vrp_daily            = excluded.vrp_daily,
                hv5                  = excluded.hv5,
                spx_return_lag1      = excluded.spx_return_lag1,
                abs_return_lag1      = excluded.abs_return_lag1,
                has_fomc_today       = excluded.has_fomc_today,
                has_cpi_today        = excluded.has_cpi_today,
                has_nfp_today        = excluded.has_nfp_today,
                event_count          = excluded.event_count,
                gex                  = excluded.gex,
                gex_normalized       = excluded.gex_normalized,
                updated_at           = excluded.updated_at
        """, (
            session_date.strftime("%Y-%m-%d"), ticker,
            _f(row, "log_range"),         _f(row, "range_pct"),
            _f(row, "har_d1_daily"),      _f(row, "har_w_daily"),
            _f(row, "har_m_daily"),
            _f(row, "vix_close"),         _f(row, "vix1d_close"),
            _f(row, "vix1d_implied_range"), _f(row, "vrp_daily"),
            _f(row, "hv5"),
            _f(row, "spx_return_lag1"),   _f(row, "abs_return_lag1"),
            _i(row, "has_fomc_today"),    _i(row, "has_cpi_today"),
            _i(row, "has_nfp_today"),     _i(row, "event_count"),
            _f(row, "gex"),               _f(row, "gex_normalized"),
            now,
        ))
        rows += 1
    conn.commit()
    log.info(f"daily_model_features ({ticker}): {rows} rows upserted")


def get_daily_features(conn, ticker: str = "SPX",
                        min_date: str = None,
                        exclude_covid: bool = False) -> pd.DataFrame:
    """Load the daily feature matrix for a ticker as a DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM daily_model_features WHERE ticker = ? "
        "ORDER BY session_date ASC",
        conn,
        params=(ticker,),
        parse_dates=["session_date"],
    )
    if df.empty:
        return df
    df.set_index("session_date", inplace=True)

    if min_date:
        df = df[df.index >= pd.to_datetime(min_date)]

    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end   = pd.Timestamp("2020-09-30")
        df = df[(df.index < covid_start) | (df.index > covid_end)]

    return df


def get_daily_feature_for_date(conn, session_date: str,
                                ticker: str = "SPX") -> pd.Series | None:
    """Fetch the daily feature row for a specific (date, ticker)."""
    df = pd.read_sql_query(
        "SELECT * FROM daily_model_features "
        "WHERE session_date = ? AND ticker = ?",
        conn,
        params=(session_date, ticker),
        parse_dates=["session_date"],
    )
    if df.empty:
        log.warning(f"No daily feature row for session_date={session_date} "
                    f"ticker={ticker}")
        return None
    df.set_index("session_date", inplace=True)
    return df.iloc[0]


def _f(row, col):
    """Safe float extractor — same contract as feature_builder._f."""
    val = row.get(col) if hasattr(row, "get") else None
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _i(row, col):
    """Safe int extractor."""
    val = _f(row, col)
    return None if val is None else int(val)
