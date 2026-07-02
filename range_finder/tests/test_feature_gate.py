"""Cadence-aware feature gate (har_model.feature_has_enough_data).

The gate counts non-null observations at the CALLER's cadence: weekly
callers use the default (20 weeks), daily callers pass
min_obs=DEFAULT_MIN_DAYS_FOR_FIT (126 trading days ≈ half a year) so a
daily spec can no longer squeak through on ~1 month of data. Per-feature
overrides (gex_normalized=12) always win.
"""
import pandas as pd

from range_finder.har_model import (
    DEFAULT_MIN_DAYS_FOR_FIT,
    DEFAULT_MIN_WEEKS_FOR_FIT,
    GEX_MIN_WEEKS_FOR_FIT,
    feature_has_enough_data,
)


def _frame(col: str, non_null: int, total: int | None = None) -> pd.DataFrame:
    total = total if total is not None else non_null
    vals = [1.0] * non_null + [None] * (total - non_null)
    return pd.DataFrame({col: vals})


# ── weekly default boundary ────────────────────────────────────────────────────

def test_weekly_default_passes_above_threshold():
    df = _frame("vix_close", DEFAULT_MIN_WEEKS_FOR_FIT + 1)   # 21 non-null
    assert feature_has_enough_data(df, "vix_close") is True


def test_weekly_default_fails_at_threshold():
    df = _frame("vix_close", DEFAULT_MIN_WEEKS_FOR_FIT)       # 20 non-null (> is strict)
    assert feature_has_enough_data(df, "vix_close") is False


# ── gex override ───────────────────────────────────────────────────────────────

def test_gex_override_passes_at_13():
    df = _frame("gex_normalized", GEX_MIN_WEEKS_FOR_FIT + 1)  # 13
    assert feature_has_enough_data(df, "gex_normalized") is True


def test_gex_override_fails_at_12():
    df = _frame("gex_normalized", GEX_MIN_WEEKS_FOR_FIT)      # 12
    assert feature_has_enough_data(df, "gex_normalized") is False


def test_override_beats_min_obs():
    # Even when a caller passes a huge min_obs, the per-feature override wins
    df = _frame("gex_normalized", GEX_MIN_WEEKS_FOR_FIT + 1)
    assert feature_has_enough_data(df, "gex_normalized",
                                   min_obs=DEFAULT_MIN_DAYS_FOR_FIT) is True


# ── daily min_obs boundary ─────────────────────────────────────────────────────

def test_daily_min_obs_fails_where_weekly_default_passed():
    # 100 daily rows: sails past the old 20-row bar, correctly fails at 126
    df = _frame("vix1d_close", 100)
    assert feature_has_enough_data(df, "vix1d_close") is True   # old behavior
    assert feature_has_enough_data(
        df, "vix1d_close", min_obs=DEFAULT_MIN_DAYS_FOR_FIT) is False


def test_daily_min_obs_boundary():
    df = _frame("vix1d_close", DEFAULT_MIN_DAYS_FOR_FIT + 1)   # 127
    assert feature_has_enough_data(
        df, "vix1d_close", min_obs=DEFAULT_MIN_DAYS_FOR_FIT) is True

    df = _frame("vix1d_close", DEFAULT_MIN_DAYS_FOR_FIT)       # 126 (strict >)
    assert feature_has_enough_data(
        df, "vix1d_close", min_obs=DEFAULT_MIN_DAYS_FOR_FIT) is False


# ── misc contracts ─────────────────────────────────────────────────────────────

def test_missing_column_is_false():
    df = _frame("something_else", 500)
    assert feature_has_enough_data(df, "vix_close") is False


def test_nulls_do_not_count():
    df = _frame("vix_close", DEFAULT_MIN_WEEKS_FOR_FIT, total=500)  # 20 real, 480 NaN
    assert feature_has_enough_data(df, "vix_close") is False
