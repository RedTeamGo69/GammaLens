# =============================================================================
# feature_experiments.py
# Candidate-feature experiments for the daily (0DTE) HAR — offline,
# evidence-gated, in-memory only.
#
# Candidates (computed here from daily_spx OHLC — NO schema or
# feature-builder changes until one clears the adoption gate):
#
#   hv5_park       — 5-day Parkinson vol (range-based; ~5x more efficient
#                    than the close-to-close hv5), lagged 1 session.
#   gap_abs        — |open_t / close_{t-1} - 1|. INFORMATION SET NOTE: the
#                    overnight gap is legitimately observable AT THE OPEN of
#                    the session whose range is being predicted (unlike the
#                    lag-1 features) — the 0DTE forecast is made at/after
#                    the open, so this is fair game, matching
#                    session_backtest.py's move_ratio construction.
#   move_ratio     — gap_abs / VIX-implied 1-day expected-move fraction
#                    (vix_close in daily_model_features is already lag-1).
#   is_monday /
#   is_friday      — day-of-week dummies (weekend gap in HAR lags; Friday
#                    positioning effects).
#
# Adoption path on keep=True: add the column to feature_builder_daily +
# daily_model_features DDL (ADD COLUMN IF NOT EXISTS), add it to
# MODEL_SPECS_DAILY, and bump model_persistence.SCHEMA_VERSION -> 4 (the
# documented v3 precedent — a feature-definition change invalidates saved
# fits; cron/UI self-heal refit automatically).
# =============================================================================

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# 1 trading year of daily rows before the first OOS prediction; refit
# cadence coarser than weekly's because there are 5x the observations.
_MIN_TRAIN_DAILY = 252
_STEP_DAILY = 10


def build_candidate_matrix(conn, ticker: str = "SPX") -> pd.DataFrame:
    """Persisted daily features + in-memory candidate columns, aligned on
    session_date."""
    from range_finder.data_collector import get_daily_spx
    from range_finder.feature_builder_daily import get_daily_features
    from range_finder.vol_estimators import compute_parkinson_vol

    feats = get_daily_features(conn, ticker=ticker)
    if feats.empty:
        raise RuntimeError(f"daily_model_features empty for {ticker}")

    bars = get_daily_spx(conn, ticker=ticker)

    # -- Parkinson 5d, lag-1 (yesterday's estimate is what's observable) --
    park = compute_parkinson_vol(
        bars, window=5,
        high_col="spx_high", low_col="spx_low",
        trading_periods=252, min_periods=4,
    )
    feats["hv5_park"] = park.shift(1).reindex(feats.index)

    # -- Overnight gap (observable at the target session's open) --
    gap = (bars["spx_open"] / bars["spx_close"].shift(1) - 1.0).abs()
    feats["gap_abs"] = gap.reindex(feats.index)

    # -- move_ratio: gap vs VIX-implied 1-day EM fraction (lag-1 VIX) --
    em_frac = feats["vix_close"] / 100.0 / math.sqrt(252)
    feats["move_ratio"] = (feats["gap_abs"] / em_frac).replace(
        [np.inf, -np.inf], np.nan)

    # -- Day-of-week dummies --
    feats["is_monday"] = (feats.index.weekday == 0).astype(int)
    feats["is_friday"] = (feats.index.weekday == 4).astype(int)

    return feats


def run_feature_experiments(conn, ticker: str = "SPX",
                            step: int = _STEP_DAILY) -> pd.DataFrame:
    """Walk-forward every candidate spec against the production daily spec."""
    from range_finder.har_model import (
        DEFAULT_MIN_DAYS_FOR_FIT, feature_has_enough_data,
    )
    from range_finder.har_model_daily import MODEL_SPECS_DAILY
    from range_finder.walkforward import WalkForwardConfig, compare_configs

    df = build_candidate_matrix(conn, ticker=ticker)
    log.info(f"Candidate matrix: {len(df)} rows x {len(df.columns)} cols")

    base_cols = tuple(
        c for c in MODEL_SPECS_DAILY["M3_daily_extended"]
        if feature_has_enough_data(df, c, min_obs=DEFAULT_MIN_DAYS_FOR_FIT)
    )
    log.info(f"Baseline (M3_daily_extended) features: {list(base_cols)}")

    baseline = "M3_daily baseline"
    configs = [
        WalkForwardConfig(label=baseline, feature_cols=base_cols),
        WalkForwardConfig(label="+ gap/move_ratio",
                          feature_cols=base_cols + ("gap_abs", "move_ratio")),
        WalkForwardConfig(label="+ day-of-week",
                          feature_cols=base_cols + ("is_monday", "is_friday")),
        WalkForwardConfig(label="hv5 -> hv5_park",
                          feature_cols=tuple(
                              "hv5_park" if c == "hv5" else c
                              for c in base_cols)),
        WalkForwardConfig(label="+ all candidates",
                          feature_cols=tuple(
                              "hv5_park" if c == "hv5" else c
                              for c in base_cols
                          ) + ("gap_abs", "move_ratio",
                               "is_monday", "is_friday")),
    ]

    return compare_configs(df, configs, baseline_label=baseline,
                           min_train=_MIN_TRAIN_DAILY, step=step)


def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Walk-forward candidate-feature experiments (daily HAR)")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--step", type=int, default=_STEP_DAILY)
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set.")
        sys.exit(1)

    from range_finder.db import get_connection

    conn = get_connection()
    run_feature_experiments(conn, ticker=args.ticker, step=args.step)

    print("ADOPTION PATH (only where keep=True):")
    print("  1. Add the column to feature_builder_daily + the")
    print("     daily_model_features DDL (ADD COLUMN IF NOT EXISTS pattern).")
    print("  2. Add it to the spec in har_model_daily.MODEL_SPECS_DAILY.")
    print("  3. Bump model_persistence.SCHEMA_VERSION -> 4 (feature-definition")
    print("     change; cron/UI self-heal refit automatically).")
    print("  keep=False everywhere -> nothing changes anywhere.")


if __name__ == "__main__":
    main()
