"""Exchange-session boundaries and the one answer to "which week?".

Every caller that needs a week key (Spread Finder, weekly EM, GEX bridge,
Monday cron, forward study) asks here instead of doing weekday arithmetic.
The distinct questions are distinct functions on purpose — they genuinely
disagree on Fridays and weekends, and that disagreement used to be spread
across seven local copies:

  planning_week     the week the Spread Finder plans     Mon–Thu this, Fri–Sun next
  em_week           the week the weekly EM describes     Mon–Fri this, Sat–Sun next
  positioning_week  the week a GEX observation feeds     this until Fri 16:00 ET, then next
  setup_week        the week the Monday cron writes      always this calendar week

All of them read the EXCHANGE clock: a UTC-hosted server rolls to "tomorrow"
at 20:00 ET, so a datetime is converted to New York time before its weekday
is taken. A plain ``date`` is taken as an exchange date.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime
    close: datetime


@dataclass(frozen=True)
class TradingWeek:
    monday: date
    sessions: tuple[Session, ...]

    @property
    def capture_start(self):
        return self.sessions[0].open

    @property
    def capture_end(self):
        return self.sessions[0].open + timedelta(minutes=45)

    @property
    def evaluation_close(self):
        return self.sessions[-1].close

    def admits(self, now: datetime) -> bool:
        return self.capture_start <= now < self.capture_end

    @property
    def key(self) -> str:
        """The week's storage key: its calendar Monday, even when a holiday."""
        return self.monday.isoformat()

    @property
    def friday(self) -> date:
        """Calendar Friday (the final session may be earlier on holidays)."""
        return self.monday + timedelta(days=4)

    @property
    def first_session_day(self) -> date:
        return self.sessions[0].day


@lru_cache(maxsize=512)
def trading_week(day: date) -> TradingWeek:
    monday = day - timedelta(days=day.weekday())
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=monday, end_date=monday + timedelta(days=4))
    sessions = tuple(Session(idx.date(), row.market_open.to_pydatetime(),
                             row.market_close.to_pydatetime())
                     for idx, row in schedule.iterrows())
    if not sessions:
        raise ValueError(f"No exchange sessions in week {monday}")
    return TradingWeek(monday, sessions)


def listed_week_expiration(available: list[str], week: TradingWeek) -> str | None:
    """Require the actual final session, never a nearby next-week contract."""
    target = week.sessions[-1].day.isoformat()
    return target if target in available else None


def _exchange_now(now) -> datetime:
    """``now`` as an ET datetime; a plain date is midnight ET that day."""
    if isinstance(now, datetime):
        return now.astimezone(NY) if now.tzinfo else now.replace(tzinfo=NY)
    return datetime(now.year, now.month, now.day, tzinfo=NY)


def _this_or_next(now: datetime, roll_to_next: bool) -> TradingWeek:
    day = now.date()
    monday = day - timedelta(days=day.weekday())
    return trading_week(monday + timedelta(days=7) if roll_to_next else monday)


def planning_week(now) -> TradingWeek:
    """Week the Spread Finder plans: this week Mon–Thu, next week Fri–Sun.

    On Friday this week's contract is 0DTE, so new weekly spreads belong to
    next week. Strikes lock to the planning week's Monday open.
    """
    et = _exchange_now(now)
    return _this_or_next(et, et.weekday() >= 4)


def em_week(now) -> TradingWeek:
    """Week the weekly expected move describes: this week Mon–Fri, next on
    the weekend (the completed week's straddle has expired)."""
    et = _exchange_now(now)
    return _this_or_next(et, et.weekday() >= 5)


# Friday's cash close. After it, dealer positioning describes the week ahead.
_POSITIONING_ROLL_HOUR = 16


def positioning_week(now) -> TradingWeek:
    """Week a GEX observation feeds as a model input: this week until
    Friday's 16:00 ET close, then the upcoming week."""
    et = _exchange_now(now)
    wd = et.weekday()
    return _this_or_next(et, wd >= 5 or (wd == 4 and et.hour >= _POSITIONING_ROLL_HOUR))


def setup_week(now) -> TradingWeek:
    """Week the Monday cron's weekly setup writes (anchor, plan, fit check).

    Always the current calendar week, even on a forced weekend run — next
    week's Monday open does not exist yet.
    """
    return _this_or_next(_exchange_now(now), False)


def is_first_session(day) -> bool:
    """Whether ``day`` is its week's first exchange session (Monday, or
    Tuesday after a Monday holiday) — the weekly freeze/setup day."""
    day = _exchange_now(day).date()
    return trading_week(day).first_session_day == day
