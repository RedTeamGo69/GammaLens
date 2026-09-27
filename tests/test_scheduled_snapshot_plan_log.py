"""Monday cron: the calibration plan is logged only off a servable row.

Regression: the cron used to fall back to ``df_feat.iloc[-1]`` when this
week's row was missing, writing a stale-feature plan into ``spread_log`` —
the input of the PI-coverage audit.
"""
import pandas as pd
import pytest

import scheduled_snapshot as snapshot

WEEK = "2026-06-15"
PRIOR = pd.Timestamp("2026-06-08")


def _features(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.DatetimeIndex(df.index, name="week_start")
    return df


@pytest.fixture
def plan_calls(monkeypatch):
    calls = {"forecast": [], "logged": []}

    def fake_recommendations(*, feature_row, **kwargs):
        calls["forecast"].append(feature_row)
        forecast = {"point_pct": 0.02, "lower_pct": 0.01, "upper_pct": 0.03}
        return forecast, {"plan": True}, []

    import range_finder.recommendations as rec
    import range_finder.spread_levels as spread_levels
    # The cron imports these lazily at call time, so patch the source modules.
    monkeypatch.setattr(rec, "build_recommendations", fake_recommendations)
    monkeypatch.setattr(spread_levels, "log_spread_plan",
                        lambda conn, plan, **k: calls["logged"].append(plan))
    monkeypatch.setattr(snapshot, "_side_share_q", lambda conn: None)
    return calls


def test_plan_logged_off_this_weeks_fresh_row(plan_calls):
    df = _features({pd.Timestamp(WEEK): {"har_d1": 0.02, "path_source_week": PRIOR}})
    logged = snapshot._log_calibration_plan(
        conn=None, df_feat=df, cal_fit=(object(), ["har_d1"]),
        week_start=WEEK, monday_open=6000.0, monday_vix=15.0, spot=6010.0,
    )
    assert logged is True
    assert plan_calls["logged"] == [{"plan": True}]
    assert plan_calls["forecast"][0]["har_d1"] == pytest.approx(0.02)


def test_missing_row_skips_plan_instead_of_falling_back(plan_calls):
    df = _features({PRIOR: {"har_d1": 0.05, "path_source_week": PRIOR - pd.Timedelta(days=7)}})
    logged = snapshot._log_calibration_plan(
        conn=None, df_feat=df, cal_fit=(object(), ["har_d1"]),
        week_start=WEEK, monday_open=6000.0, monday_vix=15.0, spot=6010.0,
    )
    assert logged is False
    assert plan_calls["forecast"] == []
    assert plan_calls["logged"] == []


def test_stale_path_row_skips_plan(plan_calls):
    two_back = PRIOR - pd.Timedelta(days=7)
    df = _features({pd.Timestamp(WEEK): {"har_d1": 0.02, "path_source_week": two_back}})
    logged = snapshot._log_calibration_plan(
        conn=None, df_feat=df, cal_fit=(object(), ["har_d1"]),
        week_start=WEEK, monday_open=6000.0, monday_vix=15.0, spot=6010.0,
    )
    assert logged is False
    assert plan_calls["logged"] == []


def test_off_schedule_run_does_not_overwrite_mondays_plan(plan_calls):
    """A forced Wed/weekend refresh must not rewrite the week's logged plan —
    on a weekend the refit has already trained on that week's outcome."""
    import datetime as dt
    df = _features({pd.Timestamp(WEEK): {"har_d1": 0.02, "path_source_week": PRIOR}})
    logged = snapshot._log_calibration_plan(
        conn=None, df_feat=df, cal_fit=(object(), ["har_d1"]),
        week_start=WEEK, monday_open=6000.0, monday_vix=15.0, spot=6010.0,
        run_date=dt.date(2026, 6, 20), anchor_date=dt.date(2026, 6, 15),
    )
    assert logged is False
    assert plan_calls["logged"] == []
