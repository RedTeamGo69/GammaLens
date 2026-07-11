# =============================================================================
# covid_experiment.py
# The "should COVID stay excluded?" experiment — offline, evidence-gated.
#
# Production strips 2020-03-01 → 2020-09-30 from every feature build and
# read (feature_builder.COVID_START/COVID_END). This experiment asks whether
# that exclusion actually helps: per ticker × spec × training policy it
# walk-forward-compares two arms that differ ONLY in COVID-row inclusion:
#
#   control    — COVID rows filtered out (the production policy, baseline)
#   treatment  — COVID rows kept in the training window
#
# ISOLATION BY CONSTRUCTION: features are built ONCE at full depth with
# exclude_covid=False; the two arms are read-time splits of the same
# model_features rows (build_features applies the COVID filter after all
# HAR/HV computation, so shared weeks carry byte-identical feature values).
# Both arms are then scored on the IDENTICAL set of post-COVID OOS weeks
# (index-intersected, >= 2020-10-01), so deltas are pure prediction effects.
#
# Known approximations (accepted, same class as history_experiment):
#   - max_train counts ROWS, not calendar: for the same OOS week the control
#     arm's rolling window reaches ~31 weeks further back in calendar time.
#   - Refit boundaries land on slightly different weeks between arms (the
#     treatment has ~31 more rows) — inherent to the treated condition.
#   - Earnings flags are NOT backfilled to depth (has_earnings is not a
#     model feature in any spec; do not read single-name results as
#     earnings-aware).
#
# SAFE BY CONSTRUCTION: production reads pass min_date=train_window_min_date()
# AND exclude_covid=True (cron + UI hardened alongside this experiment), and
# the COVID rows this experiment upserts into model_features are purged in a
# finally block unless --keep-covid-rows is passed. TRAIN_WINDOW_YEARS and
# the exclude_covid defaults are never touched here — adoption is a separate,
# deliberate flip gated on these results.
# =============================================================================

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from range_finder.feature_builder import COVID_END, COVID_START
from range_finder.har_model import (
    _MIN_MAPE_IMPROVEMENT,
    _MIN_R2_IMPROVEMENT,
)
from range_finder.history_experiment import ROLLING_6Y_ROWS
from range_finder.walkforward import (
    _DEFAULT_MIN_TRAIN,
    _DEFAULT_STEP,
    _MAX_COVERAGE_WORSENING,
    summarize,
    walk_forward_evaluate,
)

log = logging.getLogger(__name__)

# OOS scoring starts after the excluded window so neither arm is graded on
# predicting the COVID weeks themselves.
POST_COVID_OOS_START = pd.Timestamp("2020-10-01")

# A ticker must have feature rows reaching back at least this far to
# participate — otherwise there's almost no pre-COVID training mass and the
# "include COVID" question is moot for it.
MIN_FIRST_WEEK = pd.Timestamp("2019-03-01")

# Mondays inside [COVID_START, COVID_END] — the expected treatment-vs-control
# row delta for a full-history ticker.
EXPECTED_COVID_ROWS = 31

# Tolerated shortfall when a backfill comes back shallower than requested
# before the yfinance fallback kicks in.
_DEPTH_GRACE_DAYS = 180

INDEX_ETF_TICKERS = ("SPX", "SPY", "QQQ", "NDX")
SINGLE_NAME_TICKERS = ("AMZN", "AMD", "NVDA", "JPM", "CAT", "MU")

NOT_APPLICABLE = {
    "PLTR": "IPO 2020-09-30 — no COVID-period history",
    "CRWV": "IPO 2025-03 — no COVID history; below walk-forward min_train",
    "XSP": "shares SPX features (shares_har_with) — covered by the SPX row",
}

DEFAULT_TICKERS = INDEX_ETF_TICKERS + SINGLE_NAME_TICKERS + tuple(NOT_APPLICABLE)

ALL_SPECS = ("M1_baseline", "M2_vix", "M3_extended", "M4_full")

POLICIES = {"rolling6y": ROLLING_6Y_ROWS, "expanding": None}


def _group_of(ticker: str) -> str:
    return "index_etf" if ticker in INDEX_ETF_TICKERS else "single_name"


# =============================================================================
# BACKFILL — deep weekly bars per ticker, with a depth check Tradier lacks
# =============================================================================

def _reaches_depth(df: pd.DataFrame, years: int) -> bool:
    """True if the series actually extends back ~years (with grace)."""
    if df is None or df.empty:
        return False
    wanted = pd.Timestamp(datetime.today()) - pd.Timedelta(
        days=int(years * 365 - _DEPTH_GRACE_DAYS))
    return df.index.min() <= wanted


def _fetch_deepest(ticker: str, fetch_fn, years: int) -> tuple[pd.DataFrame, str]:
    """Fetch bars; if the primary source silently truncated the history,
    retry pinned to yfinance and keep whichever series reaches deeper.

    bar_sources._validate_bars checks gaps and Monday-labeling but never
    DEPTH, so a Tradier response covering only part of the window passes
    validation and would silently shrink the experiment.
    """
    from range_finder import bar_sources

    df = fetch_fn()
    if _reaches_depth(df, years):
        return df, "ok"

    have = "nothing" if df is None or df.empty else str(df.index.min().date())
    log.warning(f"[{ticker}] primary source shallow (reaches {have}, "
                f"wanted ~{years}y) — retrying pinned to yfinance")

    prior = bar_sources.SOURCE_MAP.get(ticker)
    bar_sources.SOURCE_MAP[ticker] = "yfinance"
    try:
        df_yf = fetch_fn()
    except Exception as e:
        log.warning(f"[{ticker}] yfinance depth retry failed: {e}")
        df_yf = None
    finally:
        if prior is None:
            bar_sources.SOURCE_MAP.pop(ticker, None)
        else:
            bar_sources.SOURCE_MAP[ticker] = prior

    if df_yf is not None and not df_yf.empty and (
            df is None or df.empty or df_yf.index.min() < df.index.min()):
        df = df_yf
    return df, ("ok" if _reaches_depth(df, years) else "shallow_history")


def backfill_ticker(conn, ticker: str, years: int) -> str:
    """Upsert `years` of weekly bars for one ticker. Returns depth status."""
    from phase1.ticker_config import get_config
    from range_finder.data_collector import (
        fetch_spx_vix, fetch_underlying_weekly,
        save_spx_vix, save_underlying_weekly,
    )

    if ticker == "SPX":
        df, status = _fetch_deepest(
            "SPX", lambda: fetch_spx_vix(years=years), years)
        save_spx_vix(conn, df)
    else:
        cfg = get_config(ticker)
        df, status = _fetch_deepest(
            ticker,
            lambda: fetch_underlying_weekly(
                ticker, cfg["yf_symbol"], cfg["vol_proxy_yf"], years=years),
            years,
        )
        save_underlying_weekly(conn, ticker, df)

    log.info(f"[{ticker}] backfill {status}: {len(df)} weekly rows "
             f"({df.index.min().date()} → {df.index.max().date()})")
    return status


def backfill_shared(conn, years: int) -> None:
    """One-time deep backfill of the ticker-independent inputs."""
    from range_finder.data_collector import fetch_fred_macro, save_fred_macro
    from range_finder.event_calendars import build_event_flags

    log.info(f"Backfilling {years}y of FRED macro...")
    try:
        save_fred_macro(conn, fetch_fred_macro(years=years))
    except Exception as e:
        log.warning(f"FRED backfill failed ({e}) — yield_spread stays NaN on "
                    "the deep window; M4_full's clean frame shrinks accordingly")

    build_event_flags(conn)


# =============================================================================
# FRAMES — one build, two read-time arms
# =============================================================================

def build_frames(conn, ticker: str, years: int,
                 skip_build: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (control, treatment): identical rows except the COVID weeks.

    Builds features ONCE with exclude_covid=False (COVID rows land in
    model_features until the end-of-run purge), then splits at read time.
    """
    from range_finder.feature_builder import build_features, get_features

    if not skip_build:
        build_features(conn, ticker=ticker, history_years=years,
                       exclude_covid=False)

    treatment = get_features(conn, ticker=ticker, exclude_covid=False)
    control = get_features(conn, ticker=ticker, exclude_covid=True)

    n_covid = len(treatment) - len(control)
    if n_covid > EXPECTED_COVID_ROWS:
        raise RuntimeError(
            f"{ticker}: {n_covid} COVID rows > {EXPECTED_COVID_ROWS} Mondays "
            "in the window — read-time split is broken")
    # Shared weeks must carry byte-identical feature values (the COVID filter
    # runs after all feature computation, so this is a hard invariant).
    pd.testing.assert_frame_equal(treatment.loc[control.index], control)

    log.info(f"[{ticker}] frames: control={len(control)} rows, "
             f"treatment={len(treatment)} rows ({n_covid} COVID rows)")
    return control, treatment


# =============================================================================
# ONE PAIR — walk-forward both arms, align OOS weeks, gate
# =============================================================================

def run_pair(control: pd.DataFrame, treatment: pd.DataFrame,
             feat_cols: tuple[str, ...], *, max_train: int | None,
             min_train: int, step: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate control vs treatment on identical post-COVID OOS weeks.

    Returns (record, control_preds, treatment_preds); the record carries
    ctrl_/trt_-prefixed summaries, deltas, and the adoption-gate verdict.
    """
    kwargs = dict(min_train=min_train, step=step, max_train=max_train)
    frame_c = walk_forward_evaluate(control, list(feat_cols), **kwargs)
    frame_t = walk_forward_evaluate(treatment, list(feat_cols), **kwargs)

    fc = frame_c[frame_c.index >= POST_COVID_OOS_START]
    ft = frame_t[frame_t.index >= POST_COVID_OOS_START]
    common = fc.index.intersection(ft.index)
    fc, ft = fc.loc[common], ft.loc[common]

    if not fc.index.equals(ft.index):
        raise RuntimeError("OOS week alignment failed — index mismatch")
    if not np.allclose(fc["y_true"].to_numpy(), ft["y_true"].to_numpy()):
        raise RuntimeError("OOS y_true mismatch between arms — the two frames "
                           "disagree on realized targets for shared weeks")

    s_c, s_t = summarize(fc), summarize(ft)

    delta_r2 = s_t["oos_r2"] - s_c["oos_r2"]
    delta_mae_pct = s_c["mae_pct"] - s_t["mae_pct"]
    keep = bool(
        ((delta_r2 > _MIN_R2_IMPROVEMENT)
         or (delta_mae_pct > _MIN_MAPE_IMPROVEMENT))
        and (abs(s_t["coverage_gap"])
             <= abs(s_c["coverage_gap"]) + _MAX_COVERAGE_WORSENING)
    )

    record = {
        "oos_start": common.min().date().isoformat(),
        "oos_end": common.max().date().isoformat(),
        "n_oos": int(len(common)),
        **{f"ctrl_{k}": v for k, v in s_c.items() if k != "n_oos"},
        **{f"trt_{k}": v for k, v in s_t.items() if k != "n_oos"},
        "delta_r2": delta_r2,
        "delta_mae_pct": delta_mae_pct,
        "keep": keep,
    }
    return record, fc, ft


# =============================================================================
# THE EXPERIMENT
# =============================================================================

def run_covid_experiment(conn, tickers: tuple[str, ...] = DEFAULT_TICKERS,
                         specs: tuple[str, ...] = ALL_SPECS,
                         years: int = 10, step: int = _DEFAULT_STEP,
                         min_train: int = _DEFAULT_MIN_TRAIN,
                         policies: tuple[str, ...] = ("rolling6y", "expanding"),
                         skip_backfill: bool = False,
                         preds_sink=None) -> pd.DataFrame:
    """Run the COVID-inclusion comparison. Returns one row per
    ticker × spec × policy plus status-only rows for skipped tickers.

    ``preds_sink(ticker, spec, policy, arm, frame)`` is called with each
    aligned per-obs prediction frame when provided (--save-predictions).
    """
    from range_finder.har_model import MODEL_SPECS, feature_has_enough_data

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {"run_ts": run_ts, "years": years,
            "min_train": min_train, "step": step}
    rows: list[dict] = []

    def _status_row(ticker, status, note="", **extra):
        rows.append({**base, "ticker": ticker, "group": _group_of(ticker),
                     "spec": None, "policy": None, "status": status,
                     "note": note, **extra})

    if not skip_backfill:
        backfill_shared(conn, years)

    for ticker in tickers:
        if ticker in NOT_APPLICABLE:
            _status_row(ticker, "not_applicable", NOT_APPLICABLE[ticker])
            continue

        try:
            depth = "skipped" if skip_backfill else backfill_ticker(
                conn, ticker, years)
            control, treatment = build_frames(conn, ticker, years,
                                              skip_build=skip_backfill)
        except Exception as e:
            log.error(f"[{ticker}] backfill/build failed: {e}")
            _status_row(ticker, "build_failed", str(e))
            continue

        n_covid = len(treatment) - len(control)
        if control.empty or n_covid == 0:
            _status_row(ticker, "no_covid_rows",
                        "no feature rows inside the COVID window — nothing "
                        "to include (short history or purged rows with "
                        "--skip-backfill)", backfill=depth)
            continue
        if control.index.min() > MIN_FIRST_WEEK:
            _status_row(ticker, "insufficient_pre_covid_history",
                        f"features start {control.index.min().date()} — "
                        f"need <= {MIN_FIRST_WEEK.date()}", backfill=depth)
            continue

        for spec in specs:
            feat_cols = tuple(c for c in MODEL_SPECS[spec]
                              if feature_has_enough_data(control, c))
            dropped = sorted(set(MODEL_SPECS[spec]) - set(feat_cols))
            if dropped:
                log.info(f"[{ticker}] {spec}: dropped on deep window: {dropped}")
            if len(feat_cols) < 2:
                rows.append({**base, "ticker": ticker,
                             "group": _group_of(ticker), "spec": spec,
                             "policy": None, "status": "spec_skipped_features",
                             "note": f"only {len(feat_cols)} usable features",
                             "backfill": depth})
                continue

            for policy in policies:
                try:
                    record, fc, ft = run_pair(
                        control, treatment, feat_cols,
                        max_train=POLICIES[policy],
                        min_train=min_train, step=step)
                except Exception as e:
                    log.error(f"[{ticker}] {spec}/{policy} failed: {e}")
                    rows.append({**base, "ticker": ticker,
                                 "group": _group_of(ticker), "spec": spec,
                                 "policy": policy, "status": "error",
                                 "note": str(e), "backfill": depth})
                    continue

                if preds_sink is not None:
                    preds_sink(ticker, spec, policy, "ctrl", fc)
                    preds_sink(ticker, spec, policy, "trt", ft)

                rows.append({**base, "ticker": ticker,
                             "group": _group_of(ticker), "spec": spec,
                             "policy": policy, "status": "ok",
                             "note": "", "backfill": depth,
                             "feat_cols": "|".join(feat_cols),
                             "n_rows_control": len(control),
                             "n_rows_treatment": len(treatment),
                             "n_covid_rows": n_covid,
                             **record})

    return pd.DataFrame(rows)


# =============================================================================
# PURGE — restore the "no COVID rows in model_features" invariant
# =============================================================================

def purge_covid_rows(conn) -> int:
    """Delete every model_features row inside the COVID window (all tickers).
    Returns the number of rows deleted."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM model_features WHERE week_start >= ? AND week_start <= ?",
        (COVID_START.strftime("%Y-%m-%d"), COVID_END.strftime("%Y-%m-%d")),
    )
    deleted = cur.rowcount
    conn.commit()
    log.info(f"purged {deleted} COVID-window rows from model_features")
    return deleted


# =============================================================================
# REPORTING
# =============================================================================

_PAIR_COLS = ["ticker", "spec", "policy", "n_oos",
              "ctrl_oos_r2", "trt_oos_r2", "delta_r2",
              "ctrl_mae_pct", "trt_mae_pct", "delta_mae_pct",
              "ctrl_coverage_two_sided", "trt_coverage_two_sided", "keep"]


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))


def print_report(results: pd.DataFrame, alpha_nominal: float = 0.80) -> None:
    if results.empty:
        print("\nNo results produced.")
        return
    ok = results[results["status"] == "ok"]

    print("\n" + "=" * 100)
    print("  COVID-INCLUSION EXPERIMENT — control (COVID excluded, production "
          "policy) vs treatment (COVID kept)")
    print("=" * 100)

    if not ok.empty:
        for group, label in (("index_etf", "INDEX / ETF"),
                             ("single_name", "SINGLE NAME")):
            sub = ok[ok["group"] == group]
            if sub.empty:
                continue
            print(f"\n  {label}")
            print("  " + "-" * 96)
            print(_fmt(sub[[c for c in _PAIR_COLS if c in sub.columns]]))

        print("\n  GROUP SUMMARY (mean deltas; keep = adoption gate passed)")
        print("  " + "-" * 96)
        summary = (ok.groupby(["group", "spec", "policy"])
                     .agg(mean_delta_r2=("delta_r2", "mean"),
                          mean_delta_mae_pct=("delta_mae_pct", "mean"),
                          n_keep=("keep", "sum"),
                          n_run=("keep", "size"))
                     .reset_index())
        print(_fmt(summary))
        print(f"\n  Gate: keep if delta_r2 > {_MIN_R2_IMPROVEMENT} OR "
              f"delta_mae_pct > {_MIN_MAPE_IMPROVEMENT * 100:.1f}pp, AND "
              f"|coverage - nominal| doesn't worsen by > "
              f"{_MAX_COVERAGE_WORSENING * 100:.0f}pp vs control "
              f"(nominal two-sided {alpha_nominal:.0%})")

    skipped = results[results["status"] != "ok"]
    if not skipped.empty:
        print("\n  SKIPPED / NOT APPLICABLE")
        print("  " + "-" * 96)
        print(_fmt(skipped[["ticker", "spec", "policy", "status", "note"]]
                   .fillna("-")))
    print("=" * 100)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Walk-forward comparison: does including the COVID "
                    "window in training improve OOS accuracy?")
    parser.add_argument("--years", type=int, default=10,
                        help="Backfill depth in years (default 10)")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--specs", default=",".join(ALL_SPECS))
    parser.add_argument("--step", type=int, default=_DEFAULT_STEP)
    parser.add_argument("--min-train", type=int, default=_DEFAULT_MIN_TRAIN,
                        help=f"Walk-forward warm-up rows (default "
                             f"{_DEFAULT_MIN_TRAIN}; use {ROLLING_6Y_ROWS} "
                             "for a full-production-window sensitivity run)")
    parser.add_argument("--policies", default="rolling6y,expanding")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Reuse existing model_features rows (rerun after "
                             "a --keep-covid-rows run)")
    parser.add_argument("--keep-covid-rows", action="store_true",
                        help="Skip the end-of-run purge (iterating on "
                             "analysis; rerun purge_covid_rows when done)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also dump the aligned per-obs prediction "
                             "frames per (ticker, spec, policy, arm)")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set.")
        sys.exit(1)

    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        log.error(f"unknown policies {unknown} — choose from {list(POLICIES)}")
        sys.exit(1)

    from range_finder.db import get_connection, init_all_tables

    conn = get_connection()
    init_all_tables(conn)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.abspath(
        os.path.join(args.outdir, f"covid_experiment_{stamp}.csv"))

    preds_sink = None
    if args.save_predictions:
        def preds_sink(ticker, spec, policy, arm, frame):
            path = os.path.join(
                args.outdir,
                f"covid_experiment_{stamp}_preds_{ticker}_{spec}_{policy}_{arm}.csv")
            frame.to_csv(path)

    results = pd.DataFrame()
    try:
        results = run_covid_experiment(
            conn,
            tickers=tuple(t.strip().upper()
                          for t in args.tickers.split(",") if t.strip()),
            specs=tuple(s.strip() for s in args.specs.split(",") if s.strip()),
            years=args.years, step=args.step, min_train=args.min_train,
            policies=policies, skip_backfill=args.skip_backfill,
            preds_sink=preds_sink,
        )
    finally:
        if not results.empty:
            results.to_csv(csv_path, index=False)
        if args.keep_covid_rows:
            log.warning("--keep-covid-rows: purge skipped — model_features "
                        "still holds COVID rows; production reads are "
                        "exclude_covid-hardened, but rerun the purge "
                        "(python -c \"...purge_covid_rows...\" or a normal "
                        "run) when done iterating")
        else:
            purge_covid_rows(conn)

    print_report(results)

    print("\nDECISION PROCEDURE")
    print("  1. COVID inclusion is adoptable ONLY where keep=True above")
    print("     (R²/MAE gate + coverage guard, on identical OOS weeks).")
    print("  2. To adopt: flip the exclude_covid defaults in")
    print("     feature_builder.build_features/get_features AND remove the")
    print("     exclude_covid=True args at the production read sites")
    print("     (scheduled_snapshot, ui_spread_finder, har_model,")
    print("     walkforward CLI) — TRAIN_WINDOW_YEARS stays untouched (the")
    print("     6y window only reaches the COVID tail until ~Sep 2026).")
    print("  3. keep=False everywhere -> production stays exactly as it is;")
    print("     COVID rows were purged; deeper non-COVID rows are harmless")
    print("     (all production reads pin min_date=train_window_min_date()).")
    print(f"\nResults CSV: {csv_path}")


if __name__ == "__main__":
    main()
