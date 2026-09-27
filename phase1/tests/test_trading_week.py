"""The one calendar that answers "which week?" for every caller.

Before this module owned the question there were seven local rules (UI tab ×3,
EM date key, GEX bridge, cron, freeze-day check) that disagreed on Fridays and
weekends. Each question below is a distinct domain concept:

  planning_week     the week the Spread Finder plans     Mon–Thu this, Fri–Sun next
  em_week           the week the weekly EM describes     Mon–Fri this, Sat–Sun next
  positioning_week  the week a GEX observation feeds     this until Fri 16:00 ET, then next
  setup_week        the week the Monday cron writes      always this calendar week
  is_first_session  the week's first exchange session    Monday, or Tuesday after a holiday
"""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from phase1.trading_week import (
    em_week, is_first_session, planning_week, positioning_week, setup_week,
)

NY = ZoneInfo("America/New_York")


def _ny(y, m, d, hh=10, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


# Week of 2026-06-08 (Mon) .. 2026-06-12 (Fri); next Monday 2026-06-15.
@pytest.mark.parametrize("day,expected", [
    (8, "2026-06-08"), (11, "2026-06-08"),                  # Mon, Thu → this
    (12, "2026-06-15"), (13, "2026-06-15"), (14, "2026-06-15"),  # Fri–Sun → next
])
def test_planning_week(day, expected):
    assert planning_week(_ny(2026, 6, day)).key == expected


@pytest.mark.parametrize("day,expected", [
    (8, "2026-06-08"), (12, "2026-06-08"),                  # Mon–Fri → this
    (13, "2026-06-15"), (14, "2026-06-15"),                 # Sat–Sun → next
])
def test_em_week(day, expected):
    assert em_week(_ny(2026, 6, day)).key == expected


def test_positioning_week_rolls_at_friday_close():
    assert positioning_week(_ny(2026, 6, 12, 15, 59)).key == "2026-06-08"
    assert positioning_week(_ny(2026, 6, 12, 16, 0)).key == "2026-06-15"
    assert positioning_week(_ny(2026, 6, 13)).key == "2026-06-15"


@pytest.mark.parametrize("day", [8, 10, 12, 13, 14])
def test_setup_week_is_always_this_calendar_week(day):
    assert setup_week(_ny(2026, 6, day)).key == "2026-06-08"


def test_questions_use_the_exchange_clock_not_utc():
    # Thu 21:00 ET is already Friday in UTC — still Thursday for planning.
    thu_evening_utc = datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc)
    assert planning_week(thu_evening_utc).key == "2026-06-08"


def test_accepts_plain_dates():
    assert planning_week(date(2026, 6, 12)).key == "2026-06-15"
    assert em_week(date(2026, 6, 12)).key == "2026-06-08"


def test_first_session_is_monday_or_tuesday_after_holiday():
    assert is_first_session(date(2026, 6, 8)) is True       # plain Monday
    assert is_first_session(date(2026, 6, 9)) is False      # plain Tuesday
    assert is_first_session(date(2026, 5, 25)) is False     # Memorial Day
    assert is_first_session(date(2026, 5, 26)) is True      # Tuesday after it
    assert is_first_session(date(2026, 6, 13)) is False     # Saturday


def test_week_exposes_calendar_friday_and_final_session():
    week = planning_week(_ny(2026, 6, 16))                  # Juneteenth week
    assert week.friday == date(2026, 6, 19)
    assert week.sessions[-1].day == date(2026, 6, 18)       # Thu is final
    assert week.first_session_day == date(2026, 6, 15)
