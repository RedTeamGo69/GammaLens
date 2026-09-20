from copy import deepcopy
from pathlib import Path
import runpy

import pytest
from streamlit.testing.v1 import AppTest

from ui_forward_summary import weekly_summary, ticker_recap


def fixture_rows():
    path = Path(__file__).parent / "fixtures" / "forward_summary_rows.py"
    return runpy.run_path(str(path))["summary_rows"]()


def test_completed_recap_counts_recoveries_and_tight_margin():
    result = weekly_summary(fixture_rows())
    assert result["complete"] and result["captured"] == 64
    m = result["metrics"]
    assert (m["close_hits"], m["close_n"], m["breaches"], m["recoveries"]) == (53, 64, 17, 6)
    assert [r["Closed inside"] for r in result["models"]] == [
        "13/16 · 81.2%", "12/16 · 75.0%", "14/16 · 87.5%", "14/16 · 87.5%"]
    assert "11 of the 11" in result["spotlight"]
    assert "$0.18 margin" in result["spotlight"]
    assert "+15.1%" in result["spotlight"]


def test_missing_path_data_does_not_become_a_clean_week():
    rows = fixture_rows()[:16]
    for r in rows:
        r.update(path_eligible=False, status="incomplete", either_breach=None)
    result = weekly_summary(rows)
    assert not result["complete"]
    assert result["metrics"]["close_n"] == 16 and result["metrics"]["path_n"] == 0
    assert result["tickers"][0]["Breached"] == "Awaiting data"
    assert "No breaches" not in ticker_recap(rows)
    assert "lack complete data" in ticker_recap(rows)


def test_pending_and_missed_rows_are_not_reported_as_losses():
    rows = fixture_rows()
    for i, r in enumerate(rows):
        r.update(close_eligible=False, path_eligible=False, close_inside=None,
                 either_breach=None, returned_inside=None, final_close=None,
                 status="pending" if i < 32 else "missed")
        if i >= 32:
            r["forecast_id"] = None
    result = weekly_summary(rows)
    assert not result["complete"] and result["captured"] == 32
    assert result["metrics"]["close_n"] == result["metrics"]["call_failures"] == 0
    assert result["spotlight"] == ""
    assert result["metrics"]["pending"] == result["metrics"]["missed"] == 32


@pytest.mark.parametrize("field,value", [("study_id", "second"), ("cohort", "late"), ("week_start", "2026-09-21")])
def test_recap_rejects_mixed_scopes(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="one study"):
        weekly_summary(rows)


def test_summary_detail_toggle_preserves_filters_and_week(monkeypatch):
    import ui_forward_test
    rows = fixture_rows()
    older = deepcopy(rows)
    for row in older:
        row["week_start"] = "2026-09-07"
    monkeypatch.setattr(ui_forward_test, "load_snapshot", lambda: (rows + older, [], []))
    at = AppTest.from_string("from ui_forward_test import render_forward_test\nrender_forward_test()").run(timeout=30)
    assert not at.exception
    assert at.button_group(key="ft_view").value == "Summary"
    assert at.selectbox(key="ft_summary_week").value == "2026-09-14"
    assert any("53 of 64" in m.value for m in at.markdown)
    assert not at.multiselect
    at.selectbox(key="ft_summary_week").set_value("2026-09-07").run()
    at.button_group(key="ft_view").set_value("Detailed data").run()
    at.multiselect(key="ft_filter_ticker").select("AMD").run()
    assert len(at.dataframe[1].value) == 32
    at.button_group(key="ft_view").set_value("Summary").run()
    assert not at.exception
    assert at.selectbox(key="ft_summary_week").value == "2026-09-07"
    assert any("53 of 64" in m.value for m in at.markdown)
    at.button_group(key="ft_view").set_value(None).run()
    assert at.button_group(key="ft_view").value == "Summary"
    at.button_group(key="ft_view").set_value("Detailed data").run()
    assert not at.exception
    assert at.multiselect(key="ft_filter_ticker").value == ["AMD"]
    assert len(at.dataframe[1].value) == 32
