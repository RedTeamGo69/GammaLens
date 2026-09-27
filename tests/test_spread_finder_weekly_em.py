"""The Spread Finder's EM must describe the week it is planning.

Regression: on Fri–Sun the tab plans NEXT week's strikes but was handed the
EM snapshot keyed to the expiring week, so the EM band and "inside EM"
warnings described the wrong contract.
"""
from datetime import date

import ui_spread_finder as usf

AVAIL = ["2026-05-08", "2026-05-11", "2026-05-15", "2026-05-22"]


def _compute(calls):
    def compute(exp):
        calls.append(exp)
        return {"expected_move_pts": 99.0, "expiration": exp}
    return compute


def test_mon_thu_uses_this_weeks_snapshot():
    calls = []
    em = usf.spread_finder_weekly_em(
        AVAIL, date(2026, 5, 6), weekly_exp="2026-05-08",
        weekly_em_snap={"expected_move_pts": 40.0}, compute_em=_compute(calls),
    )
    assert em == {"expected_move_pts": 40.0}
    assert calls == []


def test_friday_uses_live_em_for_next_weeks_expiration():
    calls = []
    em = usf.spread_finder_weekly_em(
        AVAIL, date(2026, 5, 8), weekly_exp="2026-05-08",
        weekly_em_snap={"expected_move_pts": 40.0}, compute_em=_compute(calls),
    )
    assert calls == ["2026-05-15"]
    assert em["expiration"] == "2026-05-15"


def test_friday_with_no_next_week_em_is_empty_not_this_weeks():
    em = usf.spread_finder_weekly_em(
        AVAIL, date(2026, 5, 8), weekly_exp="2026-05-08",
        weekly_em_snap={"expected_move_pts": 40.0}, compute_em=lambda exp: None,
    )
    assert em == {}


def test_failed_live_em_is_empty():
    def boom(exp):
        raise RuntimeError("chain fetch failed")
    em = usf.spread_finder_weekly_em(
        AVAIL, date(2026, 5, 8), weekly_exp="2026-05-08",
        weekly_em_snap={"expected_move_pts": 40.0}, compute_em=boom,
    )
    assert em == {}
