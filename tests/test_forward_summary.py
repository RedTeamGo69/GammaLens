from copy import deepcopy
from pathlib import Path
import runpy

import pytest
from streamlit.testing.v1 import AppTest

from range_finder.recommendations import TIER_LABELS
from ui_forward_summary import close_landings, hidden_lines, ticker_recap, weekly_summary, width_tradeoff


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


@pytest.mark.parametrize("field,value", [("study_id", "second"), ("cohort", "late")])
def test_recap_rejects_mixed_scopes(field, value):
    rows = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match="one study"):
        weekly_summary(rows)


def test_recap_pools_several_weeks_of_one_study():
    rows = fixture_rows()
    older = deepcopy(rows)
    for row in older:
        row["week_start"] = "2026-09-07"
    result = weekly_summary(rows + older)
    assert result["week_count"] == 2 and result["captured"] == 128
    m = result["metrics"]
    assert (m["close_hits"], m["close_n"]) == (106, 128)
    # One ticker's per-week prices differ across weeks, so no single
    # reference/close pair is quoted; the nearest-boundary line names its week.
    assert "opening reference" not in result["spotlight"]
    assert "week of" in result["spotlight"]


def _summary_app(monkeypatch, rows):
    import ui_forward_test
    monkeypatch.setattr(ui_forward_test, "load_snapshot", lambda: (rows, [], []))
    return AppTest.from_string("from ui_forward_test import render_forward_test\nrender_forward_test()").run(timeout=30)


def _two_weeks():
    rows = fixture_rows()
    older = deepcopy(rows)
    for row in older:
        row["week_start"] = "2026-09-07"
    return rows + older


def test_summary_week_ticker_and_model_filters(monkeypatch):
    at = _summary_app(monkeypatch, _two_weeks())
    assert not at.exception
    assert at.multiselect(key="ft_summary_week_start").value == ["2026-09-14"]
    assert at.multiselect(key="ft_summary_ticker").value == []
    assert at.multiselect(key="ft_summary_model").value == []
    at.multiselect(key="ft_summary_week_start").select("2026-09-07").run()
    assert any("106 of 128" in m.value for m in at.markdown)
    assert any("2 weeks at a glance" in s.value for s in at.subheader)
    at.multiselect(key="ft_summary_ticker").select("AMD").run()
    assert any("of 32 scored" in m.value for m in at.markdown)
    at.multiselect(key="ft_summary_model").select("M1_baseline").run()
    assert not at.exception
    assert any("of 8 scored" in m.value for m in at.markdown)
    # Clearing the week picker means every week, matching Detailed data.
    at.multiselect(key="ft_summary_week_start").set_value([]).run()
    assert any("of 8 scored" in m.value for m in at.markdown)


def test_summary_filters_with_no_matching_rows(monkeypatch):
    rows = fixture_rows()
    older = [r for r in deepcopy(rows) if r["ticker"] == "SPX"]
    for row in older:
        row["week_start"] = "2026-09-07"
    at = _summary_app(monkeypatch, rows + older)
    at.multiselect(key="ft_summary_ticker").select("AMD").run()
    at.multiselect(key="ft_summary_week_start").set_value(["2026-09-07"]).run()
    assert not at.exception
    assert any("No predictions match" in i.value for i in at.info)


def test_summary_detail_toggle_preserves_filters_and_week(monkeypatch):
    at = _summary_app(monkeypatch, _two_weeks())
    assert not at.exception
    assert at.button_group(key="ft_view").value == "Summary"
    assert any("53 of 64" in m.value for m in at.markdown)
    at.multiselect(key="ft_summary_week_start").set_value(["2026-09-07"]).run()
    at.multiselect(key="ft_summary_model").select("M2_vix").run()
    at.button_group(key="ft_view").set_value("Detailed data").run()
    # Summary picks do not leak into the Detailed filters, and vice versa.
    assert at.multiselect(key="ft_filter_model").value == []
    at.multiselect(key="ft_filter_ticker").select("AMD").run()
    assert len(at.dataframe[1].value) == 32
    at.button_group(key="ft_view").set_value("Summary").run()
    assert not at.exception
    assert at.multiselect(key="ft_summary_week_start").value == ["2026-09-07"]
    assert at.multiselect(key="ft_summary_model").value == ["M2_vix"]
    assert at.multiselect(key="ft_summary_ticker").value == []
    assert any("of 16 scored" in m.value for m in at.markdown)
    at.button_group(key="ft_view").set_value(None).run()
    assert at.button_group(key="ft_view").value == "Summary"
    at.button_group(key="ft_view").set_value("Detailed data").run()
    assert not at.exception
    assert at.multiselect(key="ft_filter_ticker").value == ["AMD"]
    assert len(at.dataframe[1].value) == 32


def test_select_all_collapses_to_the_all_placeholder(monkeypatch):
    at = _summary_app(monkeypatch, _two_weeks())
    tickers = at.multiselect(key="ft_summary_ticker")
    tickers.set_value(list(tickers.options)).run()
    assert at.multiselect(key="ft_summary_ticker").value == []
    at.multiselect(key="ft_summary_week_start").set_value(["2026-09-14", "2026-09-07"]).run()
    assert at.multiselect(key="ft_summary_week_start").value == []
    assert any("2 weeks at a glance" in s.value for s in at.subheader)
    at.button_group(key="ft_view").set_value("Detailed data").run()
    models = at.multiselect(key="ft_filter_model")
    models.set_value(list(models.options)).run()
    assert not at.exception and at.multiselect(key="ft_filter_model").value == []
    assert len(at.dataframe[1].value) == 128


def test_close_landings_show_quasimodos_model_and_where_each_close_fell():
    week, shown, ranges, closes = close_landings(fixture_rows())
    assert week == "2026-09-14" and shown == ["M2_vix", "M3_extended"]
    assert {r["Model"] for r in ranges if r["Ticker"] == "AMD"} == {"M2 VIX"}
    assert {r["Model"] for r in ranges if r["Ticker"] == "SPX"} == {"M3 Extended"}
    # Widest first, so the tighter ranges draw on top.
    assert [r["Range"] for r in ranges if r["Ticker"] == "SPX"] == [
        "Effective (+buffer)", "80% PI Upper", "Point Estimate", "Lower PI"]
    by = {c["Ticker"]: c for c in closes}
    assert by["AAPL"]["Result"] == "Inside every range"
    assert by["AMD"]["Result"] == "Above every call short"
    assert by["AMD"]["Outcome"] == "Past even the widest strike"
    assert by["AMD"]["Close"] == pytest.approx(559.82 / 486.28 - 1)
    m3 = [r for r in fixture_rows() if r["model"] == "M3_extended"]
    _, shown, _, closes = close_landings(m3)
    assert shown == ["M3_extended"]
    assert {c["Ticker"]: c["Result"] for c in closes}["AMD"] == "Above the tighter calls"


def test_close_landings_wait_for_a_valid_close():
    rows = fixture_rows()
    for r in rows:
        r.update(close_eligible=False, final_close=None, close_inside=None)
    _, _, ranges, closes = close_landings(rows)
    assert len(ranges) == 16 and closes == []


def test_width_tradeoff_has_one_point_per_model_and_width():
    points = width_tradeoff(fixture_rows())
    assert len(points) == 16
    point = next(p for p in points if p["Model"] == "M2 VIX" and p["Range"] == "Effective (+buffer)")
    assert point["Closed inside"] == 0.75 and point["Scored"] == 4
    assert point["Width"] == pytest.approx((0.6 + 145 / 486.28) / 4)


def test_summary_draws_both_charts(monkeypatch):
    at = _summary_app(monkeypatch, _two_weeks())
    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 2
    assert any("Where each close landed" in s.value for s in at.subheader)


def test_hidden_lines_names_models_that_land_on_each_other():
    points = width_tradeoff(fixture_rows())
    # The fixture gives M3 and M4 the same strikes everywhere; M1/M2 differ on AMD.
    assert hidden_lines(points) == [("M3 Extended", "M4 Full")]
    for p in points:
        if p["Model"] == "M4 Full" and p["Range"] == "Lower PI":
            p["Width"] += .01
    assert hidden_lines(points) == []


def test_a_put_side_miss_is_named_as_below():
    rows = [r for r in fixture_rows() if r["ticker"] == "AAPL" and r["model"] == "M2_vix"]
    for r in rows:
        r.update(final_close=95.0, put_short={"lower_pi": 96.0}.get(r["tier"], 90.0))
        r["close_inside"] = r["put_short"] <= 95.0
    _, _, _, closes = close_landings(rows)
    assert closes[0]["Result"] == "Below the tighter puts"
    for r in rows:
        r.update(put_short=99.0, close_inside=False)
    assert close_landings(rows)[3][0]["Result"] == "Below every put short"


def test_ranges_with_identical_strikes_are_named_together():
    _, _, ranges, _ = close_landings(fixture_rows())
    # AAPL's four ranges share 90-110; AMD's M2 VIX ranges all differ.
    assert {r["Ranges"] for r in ranges if r["Ticker"] == "AAPL"} == {
        "Lower PI = Point Estimate = 80% PI Upper = Effective (+buffer)"}
    assert {r["Ranges"] for r in ranges if r["Ticker"] == "AMD"} == set(TIER_LABELS)


def test_stacked_dots_each_carry_every_model_in_the_stack():
    points = {(p["Model"], p["Range"]): p for p in width_tradeoff(fixture_rows())}
    # All four share Lower PI strikes; at Effective only M3 and M4 coincide.
    assert points["M2 VIX", "Lower PI"]["Models"] == "M1 Baseline / M2 VIX / M3 Extended / M4 Full"
    for model in ("M3 Extended", "M4 Full"):
        p = points[model, "Effective (+buffer)"]
        assert p["Models"] == "M3 Extended / M4 Full" and p["Ranges"] == "Effective (+buffer)"
    assert points["M1 Baseline", "Effective (+buffer)"]["Models"] == "M1 Baseline"
