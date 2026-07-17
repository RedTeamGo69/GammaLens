"""Per-side strike placement (option 2 of the calibration-audit finding).

The HAR model forecasts the week's RANGE; the legacy placement split it
half-up/half-down around the anchor. But a range says nothing about where
the open sits inside it — empirically the open concentrates near one
extreme (arcsine law), so half-split strikes get touched far more often
than the range-coverage audit implies. These tests pin:

  * estimate_side_share_quantile — empirical (1-beta) quantile of the
    pooled per-side range share, with a min-obs fallback and the q >= 0.5
    floor,
  * build_spread_plan(side_share_q=...) — wider bands than legacy, exact
    per-side arithmetic, clamping, and None → byte-identical legacy plan,
  * forecast_next_week's price levels honoring the same share,
  * tier bands following the plan's share.
"""
import numpy as np
import pandas as pd
import pytest

from range_finder.har_model import (
    SIDE_SHARE_MIN_OBS,
    estimate_side_share_quantile,
)
from range_finder.spread_levels import build_spread_plan, build_spread_tiers


# ── estimator ────────────────────────────────────────────────────────────────

def _weekly_frame(s_up_values, open_=6000.0, range_pts=120.0):
    """Weekly OHLC where week i puts s_up_values[i] of its range above the
    open (H = O + s*R, L = O - (1-s)*R)."""
    idx = pd.date_range("2020-01-06", periods=len(s_up_values), freq="W-MON")
    s = np.asarray(s_up_values, dtype=float)
    high = open_ + s * range_pts
    low = open_ - (1.0 - s) * range_pts
    return pd.DataFrame(
        {"spx_open": open_, "spx_high": high, "spx_low": low},
        index=idx,
    )


def test_estimator_balanced_weeks_give_q_near_half():
    # Every week's open sits exactly mid-range → s_up = 0.5 always → any
    # quantile of the pooled sample is 0.5.
    df = _weekly_frame([0.5] * 100)
    est = estimate_side_share_quantile(df, beta=0.10)
    assert est["n"] == 100
    assert est["q"] == pytest.approx(0.5, abs=1e-9)


def test_estimator_trending_weeks_push_q_toward_one():
    # 80% of weeks are one-directional (open near an extreme, s in {.9,.1}),
    # 20% balanced. The pooled 90th percentile must land near 0.9.
    s_vals = ([0.9] * 40) + ([0.1] * 40) + ([0.5] * 20)
    est = estimate_side_share_quantile(_weekly_frame(s_vals), beta=0.10)
    assert 0.85 <= est["q"] <= 0.95


def test_estimator_thin_history_falls_back():
    df = _weekly_frame([0.7] * (SIDE_SHARE_MIN_OBS - 1))
    est = estimate_side_share_quantile(df)
    assert est["q"] is None
    assert est["n"] == SIDE_SHARE_MIN_OBS - 1


def test_estimator_never_returns_below_half():
    # Pathological sample (all balanced, low quantile requested) still
    # floors at 0.5 — a q < 0.5 would place strikes TIGHTER than legacy.
    df = _weekly_frame([0.5] * 100)
    est = estimate_side_share_quantile(df, beta=0.60)
    assert est["q"] >= 0.5


def test_estimator_missing_columns_or_empty():
    assert estimate_side_share_quantile(pd.DataFrame())["q"] is None
    assert estimate_side_share_quantile(None)["q"] is None
    bad = pd.DataFrame({"spx_open": [1.0]})
    assert estimate_side_share_quantile(bad)["q"] is None


def test_estimator_ignores_invalid_bars():
    # Zero-range and O-outside-[L,H] bars must not poison the sample.
    df = _weekly_frame([0.6] * 60)
    df.iloc[0, df.columns.get_loc("spx_high")] = df.iloc[0]["spx_low"]  # zero range
    df.iloc[1, df.columns.get_loc("spx_open")] = df.iloc[1]["spx_high"] + 50  # O > H
    est = estimate_side_share_quantile(df, min_obs=52)
    assert est["n"] == 58
    assert est["q"] == pytest.approx(0.6, abs=1e-9)


# ── plan wiring ──────────────────────────────────────────────────────────────

def _forecast(ref=6000.0, upper=0.0300):
    return {
        "point_pct": 0.0200,
        "lower_pct": 0.0120,
        "upper_pct": upper,
        "vix_implied_pct": 0.0250,
        "model_vs_vix": -0.0050,
        "confidence_level": 80,
        "spx_ref_close": ref,
    }


def test_plan_side_share_widens_bands_exactly():
    ref = 6000.0
    legacy = build_spread_plan(forecast=_forecast(ref), feature_row=None,
                               week_start="2026-06-15", vix_level=18.0,
                               ticker="SPX")
    shared = build_spread_plan(forecast=_forecast(ref), feature_row=None,
                               week_start="2026-06-15", vix_level=18.0,
                               ticker="SPX", side_share_q=0.8)

    eff = shared.effective_range_pct
    assert shared.side_share_q == pytest.approx(0.8)
    assert legacy.side_share_q == pytest.approx(0.5)
    # Exact per-side arithmetic: band = ref * (1 ± eff*q).
    assert shared.effective_upper_px == pytest.approx(ref * (1 + eff * 0.8), abs=0.011)
    assert shared.effective_lower_px == pytest.approx(ref * (1 - eff * 0.8), abs=0.011)
    # And strictly wider than the legacy half-split on both sides.
    assert shared.effective_upper_px > legacy.effective_upper_px
    assert shared.effective_lower_px < legacy.effective_lower_px
    # Short strikes move outward with the bands (rounding is outward-only).
    assert shared.call_spreads[0].short_strike >= legacy.call_spreads[0].short_strike
    assert shared.put_spreads[0].short_strike <= legacy.put_spreads[0].short_strike


def test_plan_none_q_is_legacy_behavior():
    a = build_spread_plan(forecast=_forecast(), feature_row=None,
                          week_start="2026-06-15", vix_level=18.0, ticker="SPX")
    b = build_spread_plan(forecast=_forecast(), feature_row=None,
                          week_start="2026-06-15", vix_level=18.0, ticker="SPX",
                          side_share_q=None)
    assert a.effective_upper_px == b.effective_upper_px
    assert a.effective_lower_px == b.effective_lower_px
    assert b.side_share_q == pytest.approx(0.5)


def test_plan_clamps_out_of_range_q():
    low = build_spread_plan(forecast=_forecast(), feature_row=None,
                            week_start="2026-06-15", vix_level=18.0,
                            ticker="SPX", side_share_q=0.3)   # below floor
    hi = build_spread_plan(forecast=_forecast(), feature_row=None,
                           week_start="2026-06-15", vix_level=18.0,
                           ticker="SPX", side_share_q=1.7)    # above cap
    assert low.side_share_q == pytest.approx(0.5)
    assert hi.side_share_q == pytest.approx(1.0)


def test_tiers_follow_plan_share():
    ref = 6000.0
    fc = _forecast(ref)
    plan = build_spread_plan(forecast=fc, feature_row=None,
                             week_start="2026-06-15", vix_level=18.0,
                             ticker="SPX", side_share_q=0.8)
    tiers = build_spread_tiers(forecast=fc, plan=plan, spx_ref=ref,
                               vix_level=18.0, ticker="SPX")
    # The PI-upper tier's raw call level uses upper*0.8, rounded UP to the
    # SPX $5 grid: 6000*(1+0.03*0.8) = 6144 → 6145.
    pi_tier = [t for t in tiers if "PI Upper" in t.label][0]
    assert pi_tier.call_short == 6145.0
    # Legacy /2 would have been 6000*1.015 = 6090.
    assert pi_tier.call_short > 6090.0


# ── forecast price levels ────────────────────────────────────────────────────

def test_forecast_levels_honor_side_share(monkeypatch):
    """forecast_next_week's pi/point px must use the same share; verified
    through a minimal fitted OLS so the statsmodels path stays real."""
    import statsmodels.api as sm
    from range_finder.har_model import forecast_next_week

    rng = np.random.RandomState(3)
    X = pd.DataFrame({"har_d1": rng.uniform(-4.5, -3.2, 120)})
    y = 0.4 * X["har_d1"] + rng.normal(0, 0.15, 120) - 2.2
    result = sm.OLS(y, sm.add_constant(X)).fit()
    row = pd.Series({"har_d1": -3.9, "vix_implied_range": 0.02})

    ref = 6000.0
    legacy = forecast_next_week(result, row, ["har_d1"], ref)
    shared = forecast_next_week(result, row, ["har_d1"], ref, side_share_q=0.85)

    assert legacy["side_share_q"] == pytest.approx(0.5)
    assert shared["side_share_q"] == pytest.approx(0.85)
    assert shared["upper_pct"] == legacy["upper_pct"]          # range unchanged
    # upper_pct in the dict is rounded to 4dp while the px levels derive
    # from the unrounded value — allow ref*5e-5*q of rounding slack.
    u = shared["upper_pct"]
    assert shared["pi_upper_px"] == pytest.approx(ref * (1 + u * 0.85), abs=0.5)
    assert shared["pi_lower_px"] == pytest.approx(ref * (1 - u * 0.85), abs=0.5)
    assert shared["pi_upper_px"] > legacy["pi_upper_px"]
    # Out-of-band q values are ignored, not clamped up — the forecast layer
    # treats them as "no valid estimate".
    junk = forecast_next_week(result, row, ["har_d1"], ref, side_share_q=1.4)
    assert junk["side_share_q"] == pytest.approx(0.5)
