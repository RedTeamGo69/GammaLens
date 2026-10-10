"""Candidate regressors for the weekly range model (candidate_features.py).

These columns are not in production — they exist so gap_experiment can score
them through the adoption gate. But the one mistake that a green test suite
would otherwise miss is the LAG BOUNDARY: a prior-week path statistic joined
without a lag leaks the target week's own path into its forecast, and still
produces a perfectly plausible in-sample R2. So the tests that matter most
here are the ones pinning which columns are lagged and which are not:

  * weekend_gap — SAME-WEEK, computed from the PRIOR row's close, nulled
    across holes in the index rather than differenced over untraded weeks,
  * overnight/intraday, semivariance, quarticity — PRIOR-WEEK arithmetic,
  * build_candidate_matrix — applies each class its own lag, so callers
    never choose.
"""
import numpy as np
import pandas as pd
import pytest

from range_finder.candidate_features import (
    PRIOR_WEEK_COLUMNS,
    SAME_WEEK_COLUMNS,
    build_candidate_matrix,
    overnight_intraday_split,
    realized_quarticity,
    semivariance_split,
    weekend_gap,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _weekly(opens, closes, index=None):
    """Monday-indexed weekly OHLC. Highs/lows are wide enough to stay valid."""
    idx = index if index is not None else pd.date_range(
        "2026-01-05", periods=len(opens), freq="W-MON")
    o = np.asarray(opens, dtype=float)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"spx_open": o, "spx_close": c,
         "spx_high": np.maximum(o, c) * 1.01,
         "spx_low": np.minimum(o, c) * 0.99},
        index=idx,
    )


def _daily(closes, opens=None, index=None):
    """Daily OHLC over consecutive business days from a Monday."""
    idx = index if index is not None else pd.bdate_range(
        "2026-01-05", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else c.copy()
    return pd.DataFrame(
        {"open": o, "close": c,
         "high": np.maximum(o, c) * 1.001,
         "low": np.minimum(o, c) * 0.999,
         "volume": 1.0},
        index=idx,
    )


# ── weekend_gap: the same-week feature ───────────────────────────────────────

def test_gap_uses_prior_week_close_not_same_week_close():
    # The off-by-one that would make this leaky: using spx_close[t] (a Friday
    # value, unknown at Monday's open) instead of spx_close[t-1].
    df = _weekly(opens=[100.0, 102.0, 99.0], closes=[100.0, 101.0, 103.0])
    out = weekend_gap(df)

    assert pd.isna(out["weekend_gap"].iloc[0])          # no predecessor
    assert out["weekend_gap"].iloc[1] == pytest.approx(102.0 / 100.0 - 1)
    assert out["weekend_gap"].iloc[2] == pytest.approx(99.0 / 101.0 - 1)

    # Using the same week's close would have given these instead — pin that
    # they are NOT what we produced.
    assert out["weekend_gap"].iloc[1] != pytest.approx(102.0 / 101.0 - 1)


def test_gap_absolute_column_is_magnitude():
    df = _weekly(opens=[100.0, 95.0], closes=[100.0, 96.0])
    out = weekend_gap(df)
    assert out["weekend_gap"].iloc[1] < 0
    assert out["abs_weekend_gap"].iloc[1] == pytest.approx(
        abs(out["weekend_gap"].iloc[1]))


def test_gap_nulled_across_a_missing_week():
    # A hole in the index (e.g. an exclude_covid-filtered frame) must NOT be
    # differenced: that would manufacture a two-week "weekend gap" that never
    # traded as one.
    idx = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-26"])
    df = _weekly(opens=[100.0, 101.0, 120.0], closes=[100.0, 102.0, 121.0],
                 index=idx)
    out = weekend_gap(df)

    assert out["weekend_gap"].iloc[1] == pytest.approx(101.0 / 100.0 - 1)
    assert pd.isna(out["weekend_gap"].iloc[2])          # 14-day spacing


def test_gap_kept_for_holiday_shortened_week():
    # A holiday Monday does not change the weekly index — the bar is still
    # Monday-labeled, it just opens Tuesday. The move from Friday's close to
    # that open is exactly the quantity the model needs, so it must survive.
    df = _weekly(opens=[100.0, 103.0], closes=[100.0, 104.0])
    out = weekend_gap(df)
    assert out["weekend_gap"].iloc[1] == pytest.approx(0.03)


def test_gap_rejects_non_positive_prices():
    df = _weekly(opens=[100.0, 0.0, 102.0], closes=[100.0, 101.0, 103.0])
    out = weekend_gap(df)
    assert pd.isna(out["weekend_gap"].iloc[1])


def test_gap_on_empty_frame_returns_named_columns():
    out = weekend_gap(pd.DataFrame())
    assert list(out.columns) == ["weekend_gap", "abs_weekend_gap"]


# ── prior-week decompositions ────────────────────────────────────────────────

def test_overnight_intraday_split_separates_the_two_sessions():
    # Every day opens exactly where the prior closed → all variance is
    # intraday, no overnight component.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    opens = [100.0, 100.0, 101.0, 102.0, 103.0]
    out = overnight_intraday_split(_daily(closes, opens))

    assert len(out) == 1
    assert out["rv_overnight"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["rv_intraday"].iloc[0] > 0
    assert out["overnight_share"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_overnight_share_is_one_when_all_moves_are_gaps():
    # Every day closes where it opened → all variance is overnight.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    opens = closes
    out = overnight_intraday_split(_daily(closes, opens))

    assert out["rv_intraday"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["overnight_share"].iloc[0] == pytest.approx(1.0)


def test_weekly_bins_land_on_the_monday_of_their_own_week():
    # The Friday->Monday relabel must not slide a week: daily bars for the
    # week of Mon 2026-01-05 belong to index entry 2026-01-05.
    out = overnight_intraday_split(_daily([100.0, 101.0, 102.0, 103.0, 104.0]))
    assert out.index[0] == pd.Timestamp("2026-01-05")


def test_semivariance_isolates_the_downside():
    out = semivariance_split(_daily([100.0, 101.0, 102.0, 103.0, 104.0]))
    assert out["rs_neg"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["rs_pos"].iloc[0] > 0
    assert out["rs_skew"].iloc[0] == pytest.approx(1.0)

    out_dn = semivariance_split(_daily([104.0, 103.0, 102.0, 101.0, 100.0]))
    assert out_dn["rs_pos"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out_dn["rs_skew"].iloc[0] == pytest.approx(-1.0)


def test_realized_quarticity_is_positive_and_sqrt_matches():
    out = realized_quarticity(_daily([100.0, 101.0, 99.0, 103.0, 100.0]))
    assert out["rq"].iloc[0] > 0
    assert out["rq_sqrt"].iloc[0] == pytest.approx(np.sqrt(out["rq"].iloc[0]))


# ── the lag boundary ─────────────────────────────────────────────────────────

def test_build_candidate_matrix_lags_prior_week_but_not_same_week():
    """The leak guard.

    Week 1 (Mon 2026-01-05) is volatile; week 2 (Mon 2026-01-12) is flat. A
    correctly lagged prior-week column carries week 1's variance on week 2's
    ROW. If it appeared on week 1's own row, the forecast for week 1 would be
    reading week 1's realized path.
    """
    daily = _daily(
        closes=[100.0, 106.0, 98.0, 105.0, 99.0,      # week 1 — volatile
                100.0, 100.0, 100.0, 100.0, 100.0],   # week 2 — flat
    )
    weekly = _weekly(opens=[100.0, 101.0], closes=[99.0, 100.0])

    out = build_candidate_matrix(weekly, daily)
    w1, w2 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")

    # PRIOR-WEEK: week 1's variance shows up on week 2, and week 1 has none.
    assert pd.isna(out.loc[w1, "rs_pos"]) or out.loc[w1, "rs_pos"] == 0
    assert out.loc[w2, "rs_pos"] > 0 or out.loc[w2, "rs_neg"] > 0

    # SAME-WEEK: the gap is week 2's own open against week 1's close.
    assert out.loc[w2, "weekend_gap"] == pytest.approx(101.0 / 99.0 - 1)


def test_build_candidate_matrix_emits_every_declared_column():
    daily = _daily([100.0, 101.0, 102.0, 103.0, 104.0])
    weekly = _weekly(opens=[100.0, 101.0], closes=[99.0, 100.0])
    out = build_candidate_matrix(weekly, daily)

    for col in SAME_WEEK_COLUMNS + PRIOR_WEEK_COLUMNS:
        assert col in out.columns, f"{col} missing from candidate matrix"


def test_build_candidate_matrix_survives_missing_daily_data():
    weekly = _weekly(opens=[100.0, 101.0], closes=[99.0, 100.0])
    out = build_candidate_matrix(weekly, pd.DataFrame())

    assert out["weekend_gap"].notna().any()          # same-week still works
    for col in PRIOR_WEEK_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()
