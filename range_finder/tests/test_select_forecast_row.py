"""The one "is this week's feature row servable?" rule.

UI tab, xlsx export, forward capture and the Monday cron all forecast off a
``model_features`` row. Before this rule was shared, the cron silently fell
back to ``df_feat.iloc[-1]`` — a stale row — and logged that plan to
``spread_log``, the calibration audit's input.
"""
import pandas as pd
import pytest

from range_finder.feature_builder import ForecastRowBlocked, select_forecast_row

WEEK = pd.Timestamp("2026-06-15")
PRIOR = WEEK - pd.Timedelta(days=7)


def _features(rows: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.DatetimeIndex(df.index, name="week_start")
    return df


def test_fresh_row_is_served():
    df = _features({WEEK: {"har_d1": 0.02, "path_source_week": PRIOR}})
    row, blocked = select_forecast_row(df, "2026-06-15")
    assert blocked is None
    assert row["har_d1"] == pytest.approx(0.02)


def test_missing_row_blocks_and_names_newest_row():
    df = _features({PRIOR: {"har_d1": 0.02, "path_source_week": PRIOR - pd.Timedelta(days=7)}})
    row, blocked = select_forecast_row(df, WEEK)
    assert row is None
    assert isinstance(blocked, ForecastRowBlocked)
    assert blocked.reason == "missing"
    assert blocked.newest_week == PRIOR
    assert "no feature row for 2026-06-15" in blocked.describe()


def test_stale_path_blocks_with_source_and_required_weeks():
    two_back = PRIOR - pd.Timedelta(days=7)
    df = _features({WEEK: {"har_d1": 0.02, "path_source_week": two_back}})
    row, blocked = select_forecast_row(df, WEEK)
    assert row is None
    assert blocked.reason == "stale_path"
    assert blocked.source_week == two_back
    assert blocked.required_week == PRIOR
    assert "requires 2026-06-08" in blocked.describe()


def test_missing_provenance_counts_as_stale():
    df = _features({WEEK: {"har_d1": 0.02, "path_source_week": None}})
    row, blocked = select_forecast_row(df, WEEK)
    assert row is None
    assert blocked.reason == "stale_path"
    assert blocked.source_week is None


def test_empty_frame_blocks_as_missing():
    row, blocked = select_forecast_row(pd.DataFrame(), WEEK)
    assert row is None
    assert blocked.reason == "missing"
    assert blocked.newest_week is None
