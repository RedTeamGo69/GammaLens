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

    def fake_forecast(result, row, cols, ref, side_share_q=None):
        calls["forecast"].append(row)
        return {"point_pct": 0.02, "lower_pct": 0.01, "upper_pct": 0.03}

    import range_finder.har_model as har_model
    import range_finder.spread_levels as spread_levels
    # The cron imports these lazily at call time, so patch the source modules.
    monkeypatch.setattr(har_model, "forecast_next_week", fake_forecast)
    monkeypatch.setattr(spread_levels, "build_spread_plan", lambda *a, **k: {"plan": True})
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


def test_setup_week_start_is_the_week_the_setup_writes():
    """The post-check must look where the setup wrote: on a forced weekend
    run that is the week just ended, not the upcoming EM week."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ny = ZoneInfo("America/New_York")
    assert snapshot._setup_week_start(datetime(2026, 6, 15, 9, 45, tzinfo=ny)) == "2026-06-15"
    assert snapshot._setup_week_start(datetime(2026, 6, 17, 9, 45, tzinfo=ny)) == "2026-06-15"
    assert snapshot._setup_week_start(datetime(2026, 6, 20, 11, 0, tzinfo=ny)) == "2026-06-15"
