"""The Monday anchor module: one resolver for the weekly strike anchor.

Before it existed the anchor was resolved five ways. The UI's freeze-day path
read yfinance only, fell back to a live spot tick, kept the result in
session state and OUTRANKED the persisted weekly_setup row — so a Monday
session could lock to a different open than the cron saved, and a fresh
Tuesday session would restore the DB value and move every strike.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import range_finder.weekly_anchor as wa

NY = ZoneInfo("America/New_York")


def _ny(y, m, d, hh=11, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


class _Capture:
    def __init__(self, result=(6800.0, 16.5, "src"), exc=None):
        self.calls, self.result, self.exc = [], result, exc

    def __call__(self, conn, ticker, week_start, target_date, **kw):
        self.calls.append((ticker, week_start, target_date, kw))
        if self.exc:
            raise self.exc
        return self.result


def _no_row(conn, week, ticker):
    return None


def test_friday_plans_next_week_so_there_is_no_anchor_yet():
    capture = _Capture()
    reads = []
    anchor = wa.resolve_anchor(
        "conn", "SPX", _ny(2026, 6, 12),
        read_setup=lambda c, w, t: reads.append(w), capture=capture)
    assert anchor.locked is False
    assert anchor.status == "next_week"
    assert anchor.week_start == "2026-06-15"
    assert reads == [] and capture.calls == []


def test_persisted_row_wins_and_nothing_is_captured():
    capture = _Capture()
    anchor = wa.resolve_anchor(
        "conn", "SPX", _ny(2026, 6, 10),
        read_setup=lambda c, w, t: (6750.25, 15.1), capture=capture)
    assert (anchor.open, anchor.vix, anchor.status) == (6750.25, 15.1, "db")
    assert anchor.locked and not anchor.captured
    assert capture.calls == []


def test_missing_row_after_the_open_captures_the_first_session_open():
    capture = _Capture()
    anchor = wa.resolve_anchor(
        "conn", "XSP", _ny(2026, 6, 10), read_setup=_no_row,
        live_vix=17.0, cfg={"yf_symbol": "^GSPC"}, capture=capture)
    ticker, week, target, kw = capture.calls[0]
    assert (ticker, week, str(target)) == ("XSP", "2026-06-08", "2026-06-08")
    assert kw["spot_fallback"] is None          # never persist a live tick
    assert kw["live_vix_fallback"] == 17.0
    assert kw["cfg"] == {"yf_symbol": "^GSPC"}
    assert (anchor.open, anchor.vix, anchor.status) == (6800.0, 16.5, "captured")
    assert anchor.captured


def test_monday_before_the_open_does_not_try_to_capture():
    capture = _Capture()
    anchor = wa.resolve_anchor("conn", "SPX", _ny(2026, 6, 8, 8, 0),
                               read_setup=_no_row, capture=capture)
    assert anchor.locked is False
    assert anchor.status == "pre_open"
    assert capture.calls == []


def test_capture_failure_leaves_the_week_unanchored():
    capture = _Capture(exc=RuntimeError("no daily bar yet"))
    anchor = wa.resolve_anchor("conn", "SPX", _ny(2026, 6, 8, 9, 31),
                               read_setup=_no_row, capture=capture)
    assert anchor.locked is False
    assert anchor.status == "unavailable"


def test_holiday_week_anchors_to_tuesday():
    capture = _Capture()
    wa.resolve_anchor("conn", "SPX", _ny(2026, 5, 27), read_setup=_no_row,
                      capture=capture)
    assert str(capture.calls[0][2]) == "2026-05-26"        # Memorial Day week


def test_self_healed_only_when_captured_after_the_first_session_day():
    heal = wa.resolve_anchor("conn", "SPX", _ny(2026, 6, 10),
                             read_setup=_no_row, capture=_Capture())
    live = wa.resolve_anchor("conn", "SPX", _ny(2026, 6, 8, 9, 40),
                             read_setup=_no_row, capture=_Capture())
    assert heal.self_healed is True
    assert live.self_healed is False


# ── cron capture ──────────────────────────────────────────────────────────────

def test_capture_setup_anchor_uses_setup_week_and_spot_fallback(monkeypatch):
    capture = _Capture()
    monkeypatch.setattr(wa, "live_vol_close", lambda sym: 19.0)
    anchor = wa.capture_setup_anchor(
        "conn", "SPX", _ny(2026, 6, 13), spot_fallback=6790.0,
        cfg={"vol_proxy_yf": "^VIX"}, capture=capture)
    ticker, week, target, kw = capture.calls[0]
    # Forced weekend run: this calendar week, pinned to its first session.
    assert (week, str(target)) == ("2026-06-08", "2026-06-08")
    assert kw["spot_fallback"] == 6790.0
    assert kw["live_vix_fallback"] == 19.0
    assert anchor.status == "captured" and anchor.open == 6800.0


def test_unreadable_weekly_setup_is_unavailable_not_a_crash():
    capture = _Capture()

    def broken(conn, week, ticker):
        raise ConnectionError("Neon asleep")
    anchor = wa.resolve_anchor("conn", "SPX", _ny(2026, 6, 10),
                               read_setup=broken, capture=capture)
    assert anchor.status == "unavailable" and not anchor.locked
    assert capture.calls == []           # never overwrite a row we couldn't read
