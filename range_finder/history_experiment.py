# =============================================================================
# history_experiment.py
# The "is 6 years of history enough?" experiment — offline, evidence-gated.
#
# Backfills weekly SPX/VIX/macro/events to a deeper window (default 10y —
# adds the 2018 Volmageddon, the 2018 Q4 selloff, and the 2015-16
# corrections to the tail sample), rebuilds the feature matrix at full
# depth, then walk-forward-compares three TRAINING POLICIES on IDENTICAL
# out-of-sample weeks:
#
#   rolling-6y OLS   — what production does today (the baseline)
#   expanding OLS    — always train on all available history
#   expanding WLS    — all history, exponentially downweighting old regimes
#                      (half-life 52w): long-tail info without stale drag
#
# min_train = 312 weekly rows (~6y) so every prediction has at least the
# production window behind it and the policies genuinely differ from the
# first OOS observation. With ~10y of data that puts the OOS window at
# ~2022 -> now — the 0DTE-era regime the finder actually trades in.
#
# SAFE BY CONSTRUCTION: the backfill only upserts additional PAST rows into
# weekly_spx / model_features. Every production fit path (Monday cron, UI,
# bootstrap, run_full_pipeline) pins its read window to
# har_model.TRAIN_WINDOW_YEARS, so nothing retrains on the deeper history
# until that constant is deliberately flipped.
#
# Decision procedure (printed as the epilogue): adopt a winner ONLY if it
# clears the compare_configs gate; then flip TRAIN_WINDOW_YEARS (and swap
# fit_model_wls into the cron loop if WLS won), force a weekly refit, and
# keep SCHEMA_VERSION unchanged (same features, different window/weights —
# the same class of change as an ordinary weekly refit).
# =============================================================================

import logging

import pandas as pd

log = logging.getLogger(__name__)

# ~6 years of weekly rows — the production window as a row count.
ROLLING_6Y_ROWS = 312


def run_history_experiment(conn, years: int = 10, ticker: str = "SPX",
                           specs: tuple[str, ...] = ("M2_vix", "M3_extended"),
                           step: int = 4) -> pd.DataFrame:
    """Backfill deep history, rebuild features, compare training policies.

    Returns the combined comparison DataFrame (one compare_configs block
    per spec, concatenated).
    """
    from range_finder.data_collector import (
        fetch_fred_macro, fetch_spx_vix, save_fred_macro, save_spx_vix,
    )
    from range_finder.event_calendars import build_event_flags
    from range_finder.feature_builder import build_features, get_features
    from range_finder.har_model import (
        MODEL_SPECS, feature_has_enough_data,
    )
    from range_finder.walkforward import WalkForwardConfig, compare_configs

    # ── 1: Backfill (additive upserts — production reads stay pinned) ──
    log.info(f"Backfilling {years}y of weekly SPX/VIX (Tradier/Cboe primary)...")
    save_spx_vix(conn, fetch_spx_vix(years=years))

    log.info(f"Backfilling {years}y of FRED macro...")
    try:
        save_fred_macro(conn, fetch_fred_macro(years=years))
    except Exception as e:
        log.warning(f"FRED backfill failed ({e}) — yield_spread will be NaN "
                    "on the deep window; M4_full shrinks accordingly")

    build_event_flags(conn)   # calendars now reach back to 2016

    # ── 2: Rebuild features at full depth ──
    log.info(f"Rebuilding {ticker} features with history_years={years}...")
    build_features(conn, ticker=ticker, history_years=years)

    # ── 3: Load the full-depth matrix (NO min_date — this is the experiment) ──
    df = get_features(conn, exclude_covid=True, ticker=ticker)
    log.info(f"Full-depth feature matrix: {len(df)} rows "
             f"({df.index.min().date()} -> {df.index.max().date()})")

    if len(df) <= ROLLING_6Y_ROWS + 40:
        raise RuntimeError(
            f"Only {len(df)} feature rows — not enough beyond the 6y baseline "
            f"({ROLLING_6Y_ROWS}) for a meaningful comparison. Did the "
            "backfill actually extend weekly_spx?"
        )

    # ── 4: Compare training policies per spec on identical OOS weeks ──
    frames = []
    for spec in specs:
        feat_cols = tuple(c for c in MODEL_SPECS[spec]
                          if feature_has_enough_data(df, c))
        log.info(f"{spec}: usable features {list(feat_cols)}")
        if len(feat_cols) < 2:
            log.warning(f"{spec}: <2 usable features on the deep window — skipping")
            continue

        # Flag GEX shrinkage explicitly rather than letting it silently
        # narrow the fit (gex_normalized only exists since the app started
        # collecting it — deep-history rows carry NaN).
        dropped = set(MODEL_SPECS[spec]) - set(feat_cols)
        if dropped:
            log.info(f"{spec}: dropped on deep window: {sorted(dropped)}")

        baseline = f"{spec} rolling-6y OLS"
        configs = [
            WalkForwardConfig(label=baseline, feature_cols=feat_cols,
                              max_train=ROLLING_6Y_ROWS),
            WalkForwardConfig(label=f"{spec} expanding-{years}y OLS",
                              feature_cols=feat_cols),
            WalkForwardConfig(label=f"{spec} expanding-{years}y WLS(hl=52)",
                              feature_cols=feat_cols,
                              fit_fn_name="wls", fit_kwargs={"half_life": 52}),
        ]
        comp = compare_configs(df, configs, baseline_label=baseline,
                               min_train=ROLLING_6Y_ROWS, step=step)
        comp["spec"] = spec
        frames.append(comp)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Walk-forward comparison of weekly-history training policies")
    parser.add_argument("--years", type=int, default=10,
                        help="Backfill depth in years (default 10)")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set.")
        sys.exit(1)

    from range_finder.db import get_connection, init_all_tables

    conn = get_connection()
    init_all_tables(conn)

    comp = run_history_experiment(conn, years=args.years, ticker=args.ticker,
                                  step=args.step)

    print("\nDECISION PROCEDURE")
    print("  1. A policy is adoptable ONLY where keep=True above (R²/MAE gate")
    print("     + coverage guard, evaluated on identical OOS weeks).")
    print("  2. To adopt a longer window: set har_model.TRAIN_WINDOW_YEARS to")
    print(f"     {args.years} — every production fit path reads through it.")
    print("  3. To adopt WLS: swap fit_model -> fit_model_wls(half_life=52) in")
    print("     scheduled_snapshot's weekly fit loop (and bootstrap).")
    print("  4. No SCHEMA_VERSION bump either way (same feature definitions).")
    print("     Force a refit afterwards: FORCE_WEEKLY_SETUP=1 cron run or the")
    print("     UI Forecast button.")
    print("  5. keep=False everywhere -> production stays exactly as it is;")
    print("     the deeper weekly_spx/model_features rows are harmless (all")
    print("     production reads are pinned to TRAIN_WINDOW_YEARS).")


if __name__ == "__main__":
    main()
