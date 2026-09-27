"""A backup cron that fires after the morning window must still be useful.

GitHub delays the backup `schedule:` runs by ~4 hours (13:45 UTC fires at
~17:45 UTC), so they always landed outside the 9:20-10:15 ET window and
exited green doing nothing: a dead primary produced no capture AND no failure
notification, and a failed Monday weekly setup (2026-09-14, 2026-09-21) was
never retried.

Late policy: the weekly setup is recoverable late (the anchor comes from the
first session's daily bar, fits don't depend on the hour) so it runs; the
opening EM snapshots are not, so their absence fails the run loudly.
"""
import pytest

import scheduled_snapshot as snapshot


def _state(daily=True, weekly=True, setup=True):
    return {"daily_em": daily, "weekly_em": weekly, "weekly_setup": setup,
            "complete": daily and weekly and setup}


def test_complete_day_needs_nothing():
    assert snapshot._late_run_plan(_state(), True) == (False, None)


def test_missing_weekly_setup_on_first_session_is_rerun_late():
    run_setup, fail = snapshot._late_run_plan(_state(setup=False), True)
    assert run_setup is True and fail is None


def test_weekly_setup_is_not_run_on_other_days():
    assert snapshot._late_run_plan(_state(setup=False), False) == (False, None)


def test_missing_opening_em_fails_loudly():
    run_setup, fail = snapshot._late_run_plan(_state(daily=False), False)
    assert run_setup is False
    assert "daily_em" in fail


def test_unknown_state_fails_loudly():
    run_setup, fail = snapshot._late_run_plan(None, True)
    assert run_setup is False and fail


@pytest.fixture
def setup_calls(monkeypatch):
    calls = []

    def fake_setup(ticker, spot, run_now, fred_key, client, avail, levels,
                   regime_info, *, late_recovery=False):
        calls.append({"ticker": ticker, "spot": spot, "late": late_recovery})
        return True

    monkeypatch.setattr(snapshot, "_run_weekly_spread_setup", fake_setup)
    monkeypatch.setattr(snapshot, "_weekly_setup_artifacts_complete", lambda t, w: True)
    return calls


def test_late_recovery_runs_setup_without_live_inputs(setup_calls):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime(2026, 9, 21, 14, 45, tzinfo=ZoneInfo("America/New_York"))
    code = snapshot._late_recovery("SPX", now, _state(setup=False), True, fred_key="")
    assert code == 0
    assert setup_calls == [{"ticker": "SPX", "spot": None, "late": True}]


def test_late_recovery_exit_code_reports_missing_em(setup_calls):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime(2026, 9, 22, 14, 45, tzinfo=ZoneInfo("America/New_York"))
    code = snapshot._late_recovery("SPX", now, _state(daily=False), False, fred_key="")
    assert code == 1
    assert setup_calls == []
