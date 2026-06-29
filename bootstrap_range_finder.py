#!/usr/bin/env python3
"""
Range Finder bootstrap — one-shot fresh-start setup.

Run this after a Postgres wipe, after a HAR feature-definition change, or
when first deploying the app. It:

  1. Creates every range_finder table (idempotent — CREATE IF NOT EXISTS).
  2. Pulls 6 years of weekly SPX/VIX OHLC from yfinance and upserts to
     weekly_spx.
  3. Pulls FRED macro (10Y, 2Y, FedFunds) if FRED_API_KEY is set.
  4. Builds event flags (FOMC / CPI / NFP / OpEx) from the static calendars.
  5. Rebuilds the full feature matrix (model_features table) using the
     canonical HAR lag structure.
  6. Fits the HAR model across all specs (M1_baseline..M4_full) and reports
     out-of-sample R² + MAE for each.
  7. Saves the best-spec model to saved_models so the Spread Finder tab
     loads it on first open.

Requires:
  DATABASE_URL  — Postgres connection string (Neon, etc.)
  FRED_API_KEY  — optional, for macro features

Usage:
  DATABASE_URL=postgres://... FRED_API_KEY=... python bootstrap_range_finder.py
  DATABASE_URL=postgres://... python bootstrap_range_finder.py --skip-fred

No Tradier token needed — this only touches yfinance + FRED + Postgres.
GEX values will be NaN for all historical weeks (expected; fresh Monday
cron runs will start populating gex_inputs going forward).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fred", action="store_true",
                        help="Skip FRED macro fetch (yield_spread/fed_funds features will be NULL)")
    parser.add_argument("--model", default="M3_extended",
                        choices=["M1_baseline", "M2_vix", "M3_extended", "M4_full"],
                        help="Which weekly model spec to fit and save (default: M3_extended)")
    parser.add_argument("--years", type=int, default=6,
                        help="Years of weekly history to fetch (default: 6)")
    parser.add_argument("--skip-daily", action="store_true",
                        help="Skip the 0DTE / daily-cadence bootstrap steps (8-10)")
    parser.add_argument("--daily-only", action="store_true",
                        help="Run ONLY the 0DTE / daily-cadence bootstrap steps "
                             "(skip weekly steps 1-7; assumes tables and macro "
                             "data already exist)")
    parser.add_argument("--daily-model", default="M2_daily_vix",
                        choices=["M1_daily_baseline", "M2_daily_vix", "M3_daily_extended"],
                        help="Which daily model spec to fit and save (default: M2_daily_vix)")
    parser.add_argument("--daily-years", type=int, default=4,
                        help="Years of daily history to fetch (default: 4; VIX1D only "
                             "exists from ~2022 so longer windows have NULL VIX1D)")
    args = parser.parse_args()

    if args.skip_daily and args.daily_only:
        _log.error("--skip-daily and --daily-only are mutually exclusive.")
        sys.exit(1)

    # ── Validate env ──
    if not os.environ.get("DATABASE_URL", "").strip():
        _log.error("DATABASE_URL not set — cannot connect to Postgres.")
        _log.error("Set it: export DATABASE_URL='postgres://user:pass@host/db'")
        sys.exit(1)

    fred_key = os.environ.get("FRED_API_KEY", "").strip()
    if not fred_key and not args.skip_fred:
        _log.warning("FRED_API_KEY not set — macro features will be NULL. "
                     "Pass --skip-fred to suppress this warning.")

    # ── Imports ──
    from range_finder.db import get_connection, init_all_tables
    from range_finder.data_collector import (
        fetch_spx_vix, save_spx_vix,
        fetch_fred_macro, save_fred_macro,
        build_event_flags, print_summary,
        fetch_daily_spx_vix, save_daily_spx,
    )
    from range_finder.event_calendars import build_event_flags_daily
    from range_finder.feature_builder import build_features, get_features
    from range_finder.feature_builder_daily import build_daily_features
    from range_finder.har_model import (
        MODEL_SPECS, time_series_split, fit_model, evaluate_oos,
        feature_has_enough_data,
    )
    from range_finder.har_model_daily import run_daily_pipeline
    from range_finder.model_persistence import save_model

    # ── Connect + init tables ──
    _log.info("Step 1/10  Connecting to Postgres and initializing tables...")
    conn = get_connection()
    init_all_tables(conn)

    # --daily-only short-circuits to the daily steps (8-10) and exits.
    if args.daily_only:
        _log.info("--daily-only requested — skipping weekly steps 2-7")
        _run_daily_bootstrap(
            conn, args.daily_years, args.daily_model,
            fetch_daily_spx_vix, save_daily_spx,
            build_event_flags_daily, build_daily_features,
            run_daily_pipeline,
        )
        print_summary(conn)
        _log.info("Daily-only bootstrap complete.")
        return

    # ── SPX/VIX history ──
    _log.info(f"Step 2/10  Fetching {args.years} years of weekly SPX/VIX from yfinance...")
    try:
        df_spx = fetch_spx_vix(years=args.years)
        n_written = save_spx_vix(conn, df_spx)
        _log.info(f"  ✓ {n_written} weekly rows upserted into weekly_spx")
    except Exception as e:
        _log.error(f"  ✗ yfinance fetch failed: {e}")
        _log.error("  Cannot proceed without weekly_spx data. Check network / yfinance.")
        sys.exit(1)

    # ── FRED macro ──
    if fred_key and not args.skip_fred:
        _log.info("Step 3/10  Fetching FRED macro (DGS10, DGS2, DFF)...")
        try:
            df_macro = fetch_fred_macro(years=args.years)
            n_macro = save_fred_macro(conn, df_macro)
            _log.info(f"  ✓ {n_macro} daily rows upserted into macro_daily")
        except Exception as e:
            _log.warning(f"  ⚠ FRED fetch failed: {e} (continuing without macro features)")
    else:
        _log.info("Step 3/10  Skipping FRED macro fetch")

    # ── Event flags ──
    _log.info("Step 4/10  Building event flags (FOMC / CPI / NFP / OpEx)...")
    try:
        build_event_flags(conn)
        _log.info("  ✓ event_flags populated")
    except Exception as e:
        _log.warning(f"  ⚠ event flag build failed: {e}")

    # ── Feature matrix rebuild ──
    _log.info("Step 5/10  Rebuilding feature matrix (canonical HAR lag structure)...")
    try:
        df_feat = build_features(conn)
        _log.info(f"  ✓ {len(df_feat)} feature rows written to model_features")
    except Exception as e:
        _log.error(f"  ✗ Feature rebuild failed: {e}")
        sys.exit(1)

    if df_feat.empty:
        _log.error("  ✗ Feature matrix is empty — cannot fit model.")
        sys.exit(1)

    # ── Fit all specs and print a comparison table ──
    _log.info("Step 6/10  Fitting all weekly model specs and comparing OOS metrics...")

    results = {}
    for spec_name in ["M1_baseline", "M2_vix", "M3_extended", "M4_full"]:
        feat_cols = MODEL_SPECS.get(spec_name, [])
        # Drop features that have too few non-null rows (eg. gex_normalized on
        # a fresh DB — there won't be any historical GEX values)
        avail_cols = [c for c in feat_cols if feature_has_enough_data(df_feat, c)]
        dropped = set(feat_cols) - set(avail_cols)
        if dropped:
            _log.info(f"  {spec_name}: dropping insufficient-data features: {sorted(dropped)}")

        if not avail_cols:
            _log.warning(f"  {spec_name}: no usable features, skipping")
            continue

        try:
            X_train, X_test, y_train, y_test = time_series_split(
                df_feat, feature_cols=avail_cols
            )
            result = fit_model(X_train, y_train, model_name=spec_name)
            metrics = evaluate_oos(result, X_test, y_test, model_name=spec_name)
            results[spec_name] = {
                "result": result,
                "metrics": metrics,
                "features": avail_cols,
            }
        except Exception as e:
            _log.warning(f"  {spec_name} fit failed: {e}")

    if not results:
        _log.error("No specs fit successfully — aborting.")
        sys.exit(1)

    # Print comparison table
    print()
    print("=" * 80)
    print("  MODEL COMPARISON — fresh HAR rebuild")
    print("=" * 80)
    print(f"  {'Spec':<16} {'OOS R²':>10} {'MAE %':>10} {'Direction':>12} {'N features':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for spec, info in results.items():
        m = info["metrics"]
        print(
            f"  {spec:<16} "
            f"{m['oos_r2']:>10.4f} "
            f"{m['mae_pct']*100:>9.2f}% "
            f"{m['direction_acc']:>11.2%} "
            f"{len(info['features']):>12}"
        )
    print()

    # ── Save the user-selected model ──
    if args.model not in results:
        _log.error(f"Requested --model {args.model} did not fit successfully. "
                   f"Available: {list(results.keys())}")
        sys.exit(1)

    _log.info(f"Step 7/10  Saving {args.model} to saved_models...")
    chosen = results[args.model]
    save_model(
        chosen["result"],
        chosen["features"],
        args.model,
        chosen["metrics"],
        conn=conn,
    )
    _log.info(f"  ✓ Weekly model {args.model} saved.")

    # ── Daily (0DTE) bootstrap ──
    if args.skip_daily:
        _log.info("--skip-daily — skipping 0DTE bootstrap steps 8-10")
    else:
        _run_daily_bootstrap(
            conn, args.daily_years, args.daily_model,
            fetch_daily_spx_vix, save_daily_spx,
            build_event_flags_daily, build_daily_features,
            run_daily_pipeline,
        )

    # ── Final summary ──
    print()
    print_summary(conn)
    _log.info("Bootstrap complete. The Spread Finder and 0DTE Finder tabs will now")
    _log.info("load their respective saved models. Subsequent cron runs will keep")
    _log.info("gex_inputs, weekly_setup, and daily_spx fresh.")


def _run_daily_bootstrap(
    conn, daily_years, daily_model,
    fetch_daily_spx_vix, save_daily_spx,
    build_event_flags_daily, build_daily_features,
    run_daily_pipeline,
) -> None:
    """Run the 0DTE / daily-cadence steps 8-10. Idempotent."""
    _log.info(f"Step 8/10  Fetching {daily_years} years of daily SPX/VIX/VIX1D from yfinance...")
    try:
        df_d = fetch_daily_spx_vix(years=daily_years)
        n = save_daily_spx(conn, df_d, ticker="SPX")
        _log.info(f"  ✓ {n} daily rows upserted into daily_spx")
    except Exception as e:
        _log.error(f"  ✗ Daily SPX/VIX/VIX1D fetch failed: {e}")
        _log.error("  Cannot proceed with daily HAR. Check network / yfinance.")
        return

    _log.info("Step 9/10  Building daily event flags and feature matrix...")
    try:
        build_event_flags_daily(conn)
    except Exception as e:
        _log.warning(f"  ⚠ Daily event flag build failed: {e}")

    try:
        df_dfeat = build_daily_features(conn, ticker="SPX")
        _log.info(f"  ✓ {len(df_dfeat)} daily feature rows written to daily_model_features")
    except Exception as e:
        _log.error(f"  ✗ Daily feature rebuild failed: {e}")
        return
    if df_dfeat.empty:
        _log.error("  ✗ Daily feature matrix empty — cannot fit daily HAR.")
        return

    _log.info(f"Step 10/10  Fitting daily HAR specs and saving {daily_model}...")
    try:
        out = run_daily_pipeline(conn, preferred_model=daily_model)
        _log.info(f"  ✓ Daily model {out['preferred']} saved (SPX).")
    except Exception as e:
        _log.error(f"  ✗ Daily HAR pipeline failed: {e}")


if __name__ == "__main__":
    main()
