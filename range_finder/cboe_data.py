# =============================================================================
# cboe_data.py
# Official Cboe index history — primary source for the VIX family.
#
# Cboe publishes free EOD OHLC CSVs for its indices at
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/{INDEX}_History.csv
# with verified coverage (2026-07): VIX1D 2022-05-13+, VIX9D 2011+, VIX3M 2009+,
# VXN 2009+, VIX decades. This matters most for VIX1D: yfinance's ^VIX1D series
# only starts at the 2023-04-24 index launch, but Cboe reconstructed values back
# to 2022-05-13 — backfilling those rows grows the 0DTE M2_daily_vix training
# set by ~30% (the daily window starts ~2022-07).
#
# yfinance stays as the fallback everywhere; a Cboe outage degrades to current
# behavior with a warning, never an error.
# =============================================================================

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger(__name__)

CBOE_INDEX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
)

# Cboe's reconstructed VIX1D series begins here — nothing earlier exists to
# backfill (true 0DTE SPX options didn't trade in size before 2022).
VIX1D_HISTORY_START = "2022-05-13"

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


def merge_cboe_vix1d(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay official Cboe VIX1D closes onto a daily frame's ``vix1d_close``."""
    return merge_cboe_closes(df, {"vix1d_close": "VIX1D"})


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


# =============================================================================
# DB BACKFILL — heal historical NULL vix1d_close rows in place
# =============================================================================

def vix1d_coverage(conn, ticker: str = "SPX") -> dict:
    """Read-only VIX1D coverage diagnostic for ``daily_spx``.

    Returns ``{min_date, max_date, non_null, total, null_in_cboe_window}``.
    ``null_in_cboe_window`` counts NULL vix1d_close rows on/after
    VIX1D_HISTORY_START — the rows ``backfill_vix1d`` can heal.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT MIN(session_date), MAX(session_date),
               SUM(CASE WHEN vix1d_close IS NOT NULL THEN 1 ELSE 0 END),
               COUNT(*)
        FROM daily_spx WHERE ticker = ?
        """,
        (ticker,),
    )
    mn, mx, non_null, total = cur.fetchone()
    cur.execute(
        """
        SELECT COUNT(*) FROM daily_spx
        WHERE ticker = ? AND vix1d_close IS NULL AND session_date >= ?
        """,
        (ticker, VIX1D_HISTORY_START),
    )
    null_in_window = cur.fetchone()[0]
    return {
        "min_date": mn,
        "max_date": mx,
        "non_null": int(non_null or 0),
        "total": int(total or 0),
        "null_in_cboe_window": int(null_in_window or 0),
    }


def backfill_vix1d(
    conn,
    start: str = VIX1D_HISTORY_START,
    overwrite: bool = False,
    ticker: str = "SPX",
) -> int:
    """Backfill ``daily_spx.vix1d_close`` from official Cboe history.

    UPDATE-only by design: rows are never inserted, so a Cboe session that the
    SPX OHLC series doesn't have (half-day quirks, source gaps) can't create a
    close-only skeleton row. Default fills only NULLs (idempotent — a rerun
    finds nothing to fill); ``overwrite=True`` replaces yfinance values with
    Cboe's official ones. Returns the number of rows actually updated.
    """
    hist = fetch_cboe_index_history("VIX1D")
    hist = hist[hist.index >= pd.Timestamp(start)]
    if hist.empty:
        log.warning(f"Cboe VIX1D history empty at/after {start} — nothing to backfill")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    null_guard = "" if overwrite else "AND vix1d_close IS NULL"
    updated = 0
    cur = conn.cursor()
    for session_date, row in hist.iterrows():
        close = float(row["close"])
        if pd.isna(close):
            continue
        cur.execute(
            f"""
            UPDATE daily_spx
            SET vix1d_close = ?, updated_at = ?
            WHERE session_date = ? AND ticker = ? {null_guard}
            """,
            (close, now, session_date.strftime("%Y-%m-%d"), ticker),
        )
        updated += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.commit()
    log.info(f"backfill_vix1d: {updated} daily_spx rows updated "
             f"(overwrite={overwrite}, start={start})")
    return updated


# =============================================================================
# CLI — coverage report + backfill
# =============================================================================
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.cboe_data [--overwrite]
#
# Prints VIX1D coverage before and after the backfill. Idempotent.

def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description="Backfill daily_spx.vix1d_close from Cboe")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing (yfinance) values with Cboe's, not just NULLs")
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set — cannot reach daily_spx.")
        sys.exit(1)

    from range_finder.db import get_connection

    conn = get_connection()
    before = vix1d_coverage(conn)
    print(f"\nBEFORE: {before}")

    updated = backfill_vix1d(conn, overwrite=args.overwrite)
    after = vix1d_coverage(conn)
    print(f"UPDATED: {updated} rows")
    print(f"AFTER:  {after}\n")


if __name__ == "__main__":
    main()
