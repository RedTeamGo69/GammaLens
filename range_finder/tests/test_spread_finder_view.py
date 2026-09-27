"""The Spread Finder view-model, end to end through a real HAR fit."""
import pytest

from range_finder.recommendations import fit_specs
from range_finder.tests.spread_finder_fixtures import WEEK, spread_finder_features
from range_finder.spread_finder_view import (
    blocked_message, build_spread_finder_view, regime_shift_message,
)

@pytest.fixture(scope="module")
def m1_fit():
    fit = fit_specs(spread_finder_features(), specs=["M1_baseline"]).fits["M1_baseline"]
    return fit.result, fit.feature_cols


def _view(features, fit, **kw):
    result, cols = fit
    args = dict(features=features, week_start=WEEK, result=result,
                feature_cols=cols, reference=6000.0, vix=16.0, live_vix=16.5,
                ticker="SPX")
    args.update(kw)
    return build_spread_finder_view(**args)


def test_ready_view_has_forecast_plan_and_four_tiers(m1_fit):
    view = _view(spread_finder_features(), m1_fit)
    assert view.ready and view.regime_shift is None
    assert 0 < view.forecast["point_pct"] < view.forecast["upper_pct"] < 1
    assert len(view.tiers) == 4
    assert view.plan.effective_lower_px < 6000.0 < view.plan.effective_upper_px


def test_missing_week_is_blocked_with_the_newest_row_named(m1_fit):
    view = _view(spread_finder_features(week="2026-06-08"), m1_fit)
    assert not view.ready and view.forecast is None
    assert "no feature row exists for 2026-06-15" in blocked_message(view)
    assert "2026-06-08" in blocked_message(view)


def test_stale_path_is_blocked(m1_fit):
    view = _view(spread_finder_features(stale=True), m1_fit)
    assert view.blocked.reason == "stale_path"
    assert "requires **2026-06-08**" in blocked_message(view)


def test_vix_spike_attaches_a_regime_shift_but_still_forecasts(m1_fit):
    view = _view(spread_finder_features(vix=15.0), m1_fit, live_vix=31.0)
    assert view.ready
    assert view.regime_shift.severity == "extreme"
    assert "2.07×" in regime_shift_message(view.regime_shift)


def test_no_live_vix_never_judges_a_shift(m1_fit):
    view = _view(spread_finder_features(vix=15.0), m1_fit, live_vix=None)
    assert view.regime_shift is None
