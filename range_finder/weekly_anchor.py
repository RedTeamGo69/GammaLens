"""The weekly strike anchor: one resolver for "what price are this week's
strikes locked to?".

Spread Finder strikes lock to the planning week's first-session daily Open
(Mon–Thu), persisted in ``weekly_setup``. Precedence is DB-first:

  1. Fri–Sun the tab plans NEXT week, whose open does not exist yet → no
     anchor (callers show live spot / Friday's close as a proxy).
  2. A persisted ``weekly_setup`` row always wins.
  3. Missing row after the first session has opened → capture the TRUE
     daily-candle Open through the Tradier → Cboe → yfinance source chain and
     persist it. The UI never persists a live spot tick; only the Monday
     cron, which must leave a row behind, passes a spot fallback.

Before this module the UI freeze-day path read yfinance only, fell back to a
live tick, kept it in session state and outranked the DB row — so a Monday
session and a fresh Tuesday session could lock to different opens.
"""
from dataclasses import dataclass

from phase1.trading_week import planning_week, setup_week
from range_finder.data_collector import (
    capture_and_save_monday_anchor, live_vol_close,
)


@dataclass(frozen=True)
class Anchor:
    week_start: str
    open: "float | None" = None
    vix: "float | None" = None
    # db | captured | next_week | pre_open | unavailable
    status: str = "unavailable"
    source: str = ""              # where a captured open came from
    captured: bool = False
    self_healed: bool = False     # captured after the first session day

    @property
    def locked(self) -> bool:
        return self.open is not None


def read_weekly_setup(conn, week_start, ticker):
    """``(monday_open, monday_vix)`` from weekly_setup, or None."""
    cur = conn.cursor()
    cur.execute(
        "SELECT monday_open, monday_vix FROM weekly_setup "
        "WHERE week_start = ? AND ticker = ?",
        (week_start, ticker),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row and row[0] else None


def resolve_anchor(conn, ticker, now, *, read_setup=read_weekly_setup,
                   live_vix=None, cfg=None,
                   capture=capture_and_save_monday_anchor) -> Anchor:
    """The planning week's anchor for ``ticker`` at ``now``.

    ``read_setup(conn, week_start, ticker)`` is the weekly_setup reader (the
    UI passes its cross-session cached one and clears it when
    ``Anchor.captured``). Never raises: an unreadable DB or a failed capture
    leaves the week unanchored and the caller falls back to live spot.
    """
    week = planning_week(now)
    if week.monday != setup_week(now).monday:
        return Anchor(week.key, status="next_week")

    try:
        row = read_setup(conn, week.key, ticker)
    except Exception:
        # Can't tell whether a row exists — never capture over one blind.
        return Anchor(week.key, status="unavailable")
    if row:
        return Anchor(week.key, float(row[0]),
                      float(row[1]) if row[1] else None, status="db")

    if now < week.capture_start:
        return Anchor(week.key, status="pre_open")
    try:
        opened, vix, _source = capture(
            conn, ticker, week.key, week.first_session_day,
            spot_fallback=None, live_vix_fallback=live_vix, cfg=cfg,
        )
    except Exception:
        return Anchor(week.key, status="unavailable")
    healed = now.date() > week.first_session_day
    return Anchor(week.key, opened, vix, status="captured",
                  captured=True, self_healed=healed)


def capture_setup_anchor(conn, ticker, now, *, spot_fallback, cfg=None,
                         capture=capture_and_save_monday_anchor) -> Anchor:
    """The Monday cron's capture: always the current calendar week, pinned
    to its first session (a forced mid-week or weekend run must not
    persist a later day's open), with spot as the last-resort fallback so
    the required weekly_setup row always exists. Raises on failure."""
    from phase1.ticker_config import get_config
    week = setup_week(now)
    cfg = cfg or get_config(ticker)
    opened, vix, source = capture(
        conn, ticker, week.key, week.first_session_day,
        spot_fallback=spot_fallback,
        live_vix_fallback=live_vol_close(cfg.get("vol_proxy_yf", "^VIX")),
        cfg=cfg,
    )
    return Anchor(week.key, opened, vix, status="captured", source=source,
                  captured=True)
