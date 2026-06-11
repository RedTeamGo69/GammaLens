"""Tests for the weekly / monthly EM snapshot date keys."""
from datetime import datetime
from zoneinfo import ZoneInfo

from phase1.gex_history import get_weekly_em_date_key, get_monthly_em_date_key

NY_TZ = ZoneInfo("America/New_York")


def _dt(y, m, d, hour=10):
    return datetime(y, m, d, hour, 0, tzinfo=NY_TZ)


def test_weekly_key_weekdays_map_to_their_own_monday():
    # 2026-06-08 is a Monday
    assert get_weekly_em_date_key(_dt(2026, 6, 8)) == "2026-06-08"
    assert get_weekly_em_date_key(_dt(2026, 6, 10)) == "2026-06-08"  # Wed
    assert get_weekly_em_date_key(_dt(2026, 6, 12)) == "2026-06-08"  # Fri


def test_weekly_key_weekend_rolls_to_upcoming_monday():
    """On Sat/Sun the completed week's straddle has expired and the UI's
    find_weekly_expiration already points at next Friday — the snapshot
    key must point at next Monday too, or weekends restore (and chart)
    the dead week's EM band."""
    assert get_weekly_em_date_key(_dt(2026, 6, 13)) == "2026-06-15"  # Sat
    assert get_weekly_em_date_key(_dt(2026, 6, 14)) == "2026-06-15"  # Sun


def test_monthly_key_is_monday_after_most_recent_third_friday():
    # June 2026: 3rd Friday is 2026-06-19. Mid-cycle (June 11) the most
    # recent third Friday strictly before today is May 15 → key = May 18.
    assert get_monthly_em_date_key(_dt(2026, 6, 11)) == "2026-05-18"
    # The Monday after June OpEx starts the new cycle.
    assert get_monthly_em_date_key(_dt(2026, 6, 22)) == "2026-06-22"
