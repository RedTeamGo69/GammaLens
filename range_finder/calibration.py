# =============================================================================
# calibration.py
# Empirical prediction-interval coverage audit.
#
# The HAR forecast quotes an 80% prediction interval (PI_ALPHA = 0.20) and
# strike placement rides its upper bound — but until this module existed,
# NOTHING ever checked whether the realized weekly range actually lands
# inside that interval ~80% of the time. spread_log stores the forecast
# bounds (point/lower/upper _pct) AND, once scored, the realized range
# (actual_range_pct); forecast_log_daily does the same at daily cadence for
# the 0DTE model. This module turns those rows into coverage numbers.
#
# Interpretation:
#   * one-sided coverage (P[actual <= upper]) is the number that matters for
#     strike placement — shorts are placed off the PI UPPER bound. Nominal
#     for an 80% central interval is 90% one-sided (10% in each tail).
#   * two-sided coverage (P[lower <= actual <= upper]) audits the interval
#     as a whole. Nominal is 80%.
#   * buffer breach rate audits the heuristic buffer ON TOP of the PI:
#     P[actual > effective_range] should be ~the tail mass the buffer is
#     believed to absorb.
#
# All functions are pure DataFrame -> dict/DataFrame transforms; the conn
# wrappers only do SELECTs (sqlite-compatible SQL, tested against sqlite).
# Nothing here is called by the cron — reports run on demand.
# =============================================================================

import logging
import math

import pandas as pd

log = logging.getLogger(__name__)

# Below this many scored observations, print the numbers but refuse to call
# them calibration evidence (mirrors analyze_wall_calibration's stance).
MIN_SAMPLE = 15

# Nominal levels implied by PI_ALPHA = 0.20 (an 80% central interval).
NOMINAL_TWO_SIDED = 0.80
NOMINAL_ONE_SIDED = 0.90


# =============================================================================
# LOADERS — thin, read-only
# =============================================================================

def load_completed_forecasts(conn, ticker: str = "SPX") -> pd.DataFrame:
    """spread_log rows that have BOTH a forecast and a scored outcome."""
    df = pd.read_sql_query(
        """
        SELECT week_start, ticker, model_name,
               point_pct, lower_pct, upper_pct, effective_range_pct,
               buffer_pct, event_count,
               actual_high, actual_low, actual_range_pct, outcome
        FROM spread_log
        WHERE ticker = ?
          AND upper_pct IS NOT NULL
          AND actual_range_pct IS NOT NULL
        ORDER BY week_start ASC
        """,
        conn,
        params=(ticker,),
        parse_dates=["week_start"],
    )
    return df


def load_completed_daily_forecasts(conn, ticker: str = "SPX",
                                   model_name: str = None) -> pd.DataFrame:
    """forecast_log_daily rows with both a forecast and a scored outcome."""
    query = """
        SELECT session_date, ticker, model_name,
               point_pct, lower_pct, upper_pct,
               actual_range_pct
        FROM forecast_log_daily
        WHERE ticker = ?
          AND upper_pct IS NOT NULL
          AND actual_range_pct IS NOT NULL
    """
    params: tuple = (ticker,)
    if model_name:
        query += " AND model_name = ?"
        params = (ticker, model_name)
    query += " ORDER BY session_date ASC"
    return pd.read_sql_query(query, conn, params=params,
                             parse_dates=["session_date"])


# =============================================================================
# COVERAGE MATH — pure functions
# =============================================================================

def _binomial_ci(successes: int, n: int, conf: float = 0.90) -> tuple[float, float]:
    """Two-sided binomial CI on a proportion (Wilson score interval).

    Wilson keeps sane bounds at the small n this audit starts life with,
    without needing scipy at import time.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = {0.90: 1.6449, 0.95: 1.9600}.get(conf, 1.6449)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def pi_coverage(df: pd.DataFrame, nominal_two_sided: float = NOMINAL_TWO_SIDED) -> dict:
    """Empirical PI coverage from completed forecast rows.

    Expects columns: actual_range_pct, upper_pct, and (optionally, may be
    NULL on legacy rows) lower_pct. Returns one-sided coverage (primary),
    two-sided coverage over the rows that carry lower_pct, Wilson 90% CIs,
    and sample sizes.
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "one_sided": float("nan"), "two_sided": float("nan"),
                "one_sided_ci": (float("nan"), float("nan")),
                "two_sided_ci": (float("nan"), float("nan")),
                "n_two_sided": 0, "nominal_one_sided": NOMINAL_ONE_SIDED,
                "nominal_two_sided": nominal_two_sided,
                "sufficient": False}

    inside_upper = (df["actual_range_pct"] <= df["upper_pct"])
    one_sided = float(inside_upper.mean())

    has_lower = df["lower_pct"].notna() if "lower_pct" in df.columns \
        else pd.Series(False, index=df.index)
    two_df = df[has_lower]
    if len(two_df):
        inside_both = ((two_df["actual_range_pct"] >= two_df["lower_pct"])
                       & (two_df["actual_range_pct"] <= two_df["upper_pct"]))
        two_sided = float(inside_both.mean())
        two_ci = _binomial_ci(int(inside_both.sum()), len(two_df))
    else:
        two_sided = float("nan")
        two_ci = (float("nan"), float("nan"))

    return {
        "n": n,
        "one_sided": one_sided,
        "one_sided_ci": _binomial_ci(int(inside_upper.sum()), n),
        "two_sided": two_sided,
        "two_sided_ci": two_ci,
        "n_two_sided": int(len(two_df)),
        "nominal_one_sided": NOMINAL_ONE_SIDED,
        "nominal_two_sided": nominal_two_sided,
        "sufficient": n >= MIN_SAMPLE,
    }


def buffer_breach_rate(df: pd.DataFrame) -> dict:
    """How often the realized range blew through PI-upper PLUS the buffer.

    This is the first empirical audit of the heuristic buffer
    (spread_levels.compute_buffer: 0.3% base, FOMC/CPI/NFP/OpEx
    multipliers, GEX adjustment). effective_range_pct = upper_pct + buffer.
    """
    scored = df[df["effective_range_pct"].notna()] \
        if "effective_range_pct" in df.columns else df.iloc[0:0]
    n = len(scored)
    if n == 0:
        return {"n": 0, "breach_rate": float("nan"),
                "breach_ci": (float("nan"), float("nan")), "sufficient": False}
    breached = (scored["actual_range_pct"] > scored["effective_range_pct"])
    return {
        "n": n,
        "breach_rate": float(breached.mean()),
        "breach_ci": _binomial_ci(int(breached.sum()), n),
        "sufficient": n >= MIN_SAMPLE,
    }


def rolling_coverage(df: pd.DataFrame, window: int = 26,
                     date_col: str = "week_start") -> pd.DataFrame:
    """Rolling one-sided coverage — drift detection.

    Returns a frame indexed by date with a `rolling_one_sided` column; NaN
    until `window` observations accumulate.
    """
    if df.empty:
        return pd.DataFrame(columns=["rolling_one_sided"])
    s = (df.set_index(date_col)
           .sort_index()
           .apply(lambda r: float(r["actual_range_pct"] <= r["upper_pct"]),
                  axis=1))
    return s.rolling(window, min_periods=window).mean() \
            .to_frame("rolling_one_sided")


def coverage_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-spec coverage breakdown. Legacy rows without a stored spec pool
    under "(unknown)"."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["model_name"] = work["model_name"].fillna("(unknown)")
    rows = []
    for name, grp in work.groupby("model_name"):
        cov = pi_coverage(grp)
        rows.append({"model_name": name, "n": cov["n"],
                     "one_sided": cov["one_sided"],
                     "two_sided": cov["two_sided"],
                     "sufficient": cov["sufficient"]})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# =============================================================================
# CONN WRAPPERS
# =============================================================================

def weekly_pi_coverage(conn, ticker: str = "SPX") -> dict:
    """One-call weekly coverage summary for the UI caption."""
    df = load_completed_forecasts(conn, ticker=ticker)
    cov = pi_coverage(df)
    cov["buffer"] = buffer_breach_rate(df)
    return cov


def daily_pi_coverage(conn, ticker: str = "SPX", model_name: str = None) -> dict:
    """One-call 0DTE coverage summary for the UI caption."""
    df = load_completed_daily_forecasts(conn, ticker=ticker, model_name=model_name)
    return pi_coverage(df)


# =============================================================================
# CLI REPORT
# =============================================================================
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.calibration
#
# Read-only. Prints the weekly + daily coverage report, refusing conclusions
# under MIN_SAMPLE observations.

def _fmt_pct(x: float) -> str:
    return "—" if x != x else f"{x:.0%}"


def _print_coverage_block(title: str, cov: dict) -> None:
    print(f"\n  {title}")
    print(f"    scored observations : {cov['n']}")
    if cov["n"] == 0:
        print("    (nothing scored yet — outcomes accumulate one per period)")
        return
    lo, hi = cov["one_sided_ci"]
    print(f"    one-sided coverage  : {_fmt_pct(cov['one_sided'])} "
          f"(CI {_fmt_pct(lo)}-{_fmt_pct(hi)}, nominal "
          f"{_fmt_pct(cov['nominal_one_sided'])})")
    if cov["n_two_sided"]:
        lo2, hi2 = cov["two_sided_ci"]
        print(f"    two-sided coverage  : {_fmt_pct(cov['two_sided'])} "
              f"(CI {_fmt_pct(lo2)}-{_fmt_pct(hi2)}, nominal "
              f"{_fmt_pct(cov['nominal_two_sided'])}, "
              f"n={cov['n_two_sided']})")
    if not cov["sufficient"]:
        # ASCII only — Windows cp1252 consoles choke on warning glyphs
        print(f"    [!] n < {MIN_SAMPLE} - numbers shown, conclusions refused. "
              "Keep accumulating.")


def main() -> None:
    import os
    import sys

    logging.basicConfig(level=logging.WARNING)

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set — cannot read spread_log.")
        sys.exit(1)

    from range_finder.db import get_connection

    conn = get_connection()

    print("=" * 70)
    print("  PREDICTION-INTERVAL CALIBRATION REPORT")
    print("=" * 70)

    df_w = load_completed_forecasts(conn, ticker="SPX")
    cov_w = pi_coverage(df_w)
    _print_coverage_block("WEEKLY (spread_log, SPX)", cov_w)

    buf = buffer_breach_rate(df_w)
    if buf["n"]:
        lo, hi = buf["breach_ci"]
        print(f"    buffer breach rate  : {_fmt_pct(buf['breach_rate'])} "
              f"(CI {_fmt_pct(lo)}-{_fmt_pct(hi)}) - realized range beyond "
              "PI-upper + buffer")

    by_model = coverage_by_model(df_w)
    if not by_model.empty:
        print("\n    per-spec breakdown:")
        for _, r in by_model.iterrows():
            print(f"      {r['model_name']:22s} n={r['n']:<4d} "
                  f"one-sided={_fmt_pct(r['one_sided'])}")

    try:
        df_d = load_completed_daily_forecasts(conn, ticker="SPX")
        _print_coverage_block("DAILY / 0DTE (forecast_log_daily, SPX)",
                              pi_coverage(df_d))
    except Exception as e:
        print(f"\n  DAILY / 0DTE: table not available yet ({e})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
