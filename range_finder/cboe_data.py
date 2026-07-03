# =============================================================================
# cboe_data.py
# Official Cboe index history — primary source for the VIX family.
#
# Cboe publishes free EOD OHLC CSVs for its indices at
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/{INDEX}_History.csv
# with verified coverage (2026-07): VIX9D 2011+, VIX3M 2009+, VXN 2009+,
# VIX decades. Feeds the weekly vol features and the Monday-anchor opens for
# vol indices.
#
# yfinance stays as the fallback everywhere; a Cboe outage degrades to current
# behavior with a warning, never an error.
# =============================================================================

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

CBOE_INDEX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
)

# Cboe has shipped both of these date formats in index-history CSVs.
_CBOE_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d")

# yfinance vol-index symbol -> Cboe CDN index name. Lets fetchers that are
# configured with yfinance symbols (ticker_config's vol_proxy_yf) find the
# official Cboe series without touching every config entry. Symbols not in
# this map simply have no Cboe primary and stay on yfinance.
YF_TO_CBOE_INDEX = {
    "^VIX":   "VIX",
    "^VIX1D": "VIX1D",
    "^VIX9D": "VIX9D",
    "^VIX3M": "VIX3M",
    "^VXN":   "VXN",
}


# =============================================================================
# FETCH
# =============================================================================

def fetch_cboe_index_history(index: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch full daily OHLC history for a Cboe index (e.g. "VIX1D", "VIX9D").

    Returns a DataFrame indexed by normalized tz-naive DatetimeIndex named
    ``session_date`` with float columns ``open, high, low, close``, sorted
    ascending. Raises on network errors / unexpected payloads — callers that
    must degrade gracefully (the merge helpers below) catch and fall back.
    """
    url = CBOE_INDEX_HISTORY_URL.format(index=index.upper())
    log.info(f"Fetching Cboe {index} history from {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    expected = {"date", "open", "high", "low", "close"}
    if not expected.issubset(raw.columns):
        raise ValueError(
            f"Cboe {index} CSV has unexpected columns {list(raw.columns)} "
            f"(expected {sorted(expected)})"
        )

    dates = _parse_cboe_dates(raw["date"], index)
    df = raw[["open", "high", "low", "close"]].astype(float).copy()
    df.index = dates
    df.index.name = "session_date"

    n_bad = int(df.index.isna().sum())
    if n_bad:
        log.warning(f"Cboe {index}: dropping {n_bad} rows with unparseable dates")
        df = df[~df.index.isna()]

    df = df.sort_index()
    log.info(
        f"Cboe {index}: {len(df)} rows "
        f"({df.index.min().date()} -> {df.index.max().date()})"
    )
    return df


def _parse_cboe_dates(col: pd.Series, index: str) -> pd.DatetimeIndex:
    """Parse Cboe date strings, trying each known format before coercing."""
    for fmt in _CBOE_DATE_FORMATS:
        parsed = pd.to_datetime(col, format=fmt, errors="coerce")
        # Accept the format that parses (nearly) everything — a wholesale
        # format mismatch coerces every row to NaT.
        if parsed.notna().sum() >= max(1, int(len(col) * 0.9)):
            return pd.DatetimeIndex(parsed).normalize()
    log.warning(f"Cboe {index}: no known date format fit — coercing best-effort")
    return pd.DatetimeIndex(pd.to_datetime(col, errors="coerce")).normalize()


# =============================================================================
# MERGE — Cboe primary, yfinance (caller's frame) fallback
# =============================================================================

def merge_cboe_closes(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Overlay Cboe closes onto ``df`` for each ``{column: cboe_index}`` pair.

    Returns a NEW frame where each mapped column is Cboe's close where Cboe
    has the session, else the original value (``combine_first``). Rows are
    never added or removed — only mapped columns change. Any Cboe failure
    degrades to the input values for that column with a warning.
    """
    out = df.copy()
    for col, index in col_map.items():
        try:
            hist = fetch_cboe_index_history(index)
        except Exception as e:
            log.warning(f"Cboe {index} unavailable ({e}) — keeping existing "
                        f"{col} values (yfinance fallback)")
            continue

        closes = hist["close"]
        # Align tz-ness with the caller's index so combine_first matches dates.
        if getattr(out.index, "tz", None) is not None:
            closes.index = closes.index.tz_localize(out.index.tz)
        closes = closes.reindex(out.index)

        before = out[col].notna().sum() if col in out.columns else 0
        base = out[col] if col in out.columns else pd.Series(index=out.index, dtype=float)
        out[col] = closes.combine_first(base)
        after = out[col].notna().sum()
        log.info(f"Cboe {index} -> {col}: non-null {before} -> {after} "
                 f"of {len(out)} rows (Cboe primary, existing values fill gaps)")
    return out


# =============================================================================
# WEEKLY RESAMPLING — Cboe daily bars -> Monday-anchored weekly bars
# =============================================================================

def resample_cboe_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a Cboe daily OHLC frame to Monday-anchored weekly bars.

    Matches yfinance's ``interval="1wk"`` convention: each bar covers
    Monday -> Friday (holiday-shortened weeks use whatever sessions exist)
    and is labeled with the week's Monday. Weeks with no sessions drop out.
    """
    weekly = daily.resample("W-MON", closed="left", label="left").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
    })
    return weekly.dropna(how="all")


def merge_cboe_weekly_ohlc(
    df: pd.DataFrame,
    index: str,
    col_map: dict[str, str],
) -> pd.DataFrame:
    """Overlay Monday-anchored weekly Cboe OHLC onto a weekly frame.

    ``col_map`` maps Cboe columns to the caller's column names, e.g.
    ``{"open": "vix_open", ..., "close": "vix_close"}``. Same contract as
    ``merge_cboe_closes``: returns a NEW frame, Cboe wins where it has the
    week, existing (yfinance) values fill gaps, rows never added/removed,
    and any Cboe failure degrades to the input with a warning.
    """
    out = df.copy()
    try:
        weekly = resample_cboe_weekly(fetch_cboe_index_history(index))
    except Exception as e:
        log.warning(f"Cboe {index} unavailable ({e}) — keeping existing "
                    f"{sorted(col_map.values())} values (yfinance fallback)")
        return out

    if getattr(out.index, "tz", None) is not None:
        weekly.index = weekly.index.tz_localize(out.index.tz)
    weekly = weekly.reindex(out.index)

    for cboe_col, df_col in col_map.items():
        base = out[df_col] if df_col in out.columns \
            else pd.Series(index=out.index, dtype=float)
        out[df_col] = weekly[cboe_col].combine_first(base)
    log.info(f"Cboe {index} weekly -> {sorted(col_map.values())}: overlaid "
             f"onto {len(out)} weekly rows (Cboe primary)")
    return out


def fetch_cboe_weekly_closes(index: str, name: str) -> pd.Series:
    """Monday-labeled weekly close series for a Cboe index (term structure).

    Raises on failure — callers that must degrade (feature_builder's
    ``fetch_vix_term_structure``) catch and fall back to yfinance.
    """
    weekly = resample_cboe_weekly(fetch_cboe_index_history(index))
    s = weekly["close"].copy()
    s.name = name
    s.index.name = "week_start"
    return s


# The daily_spx VIX1D backfill (vix1d_coverage / backfill_vix1d / CLI) was
# removed with the 0DTE finder (2026-07) — it existed solely to grow the
# daily HAR's VIX1D training window.
