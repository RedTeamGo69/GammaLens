"""
Expected-Move snapshot freeze/restore helpers.

These helpers freeze the daily / weekly / OpEx-cycle expected move at the
appropriate session start and restore the cron's persisted value on later
renders so the chart's EM markers stay stable through each session. Only the
scheduled 9:30 capture writes em_snapshots: a page view freezes into session
state alone, so a mid-day visit can never stand in for the opening capture
(the backup cron's missed-capture alarm keys off that row). Imported by
`streamlit_app.py`. The weekly freeze day is phase1.trading_week.is_first_session.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from phase1.market_clock import now_ny
from phase1.gex_history import get_em_snapshot


def _is_monthly_freeze_day(now_et):
    """
    True if today is the first trading day of the current OpEx cycle: the
    first session of the week that opens on the Monday after the most recent
    standard 3rd-Friday OpEx (Tuesday when that Monday is a holiday).
    """
    from phase1.gex_history import get_monthly_em_date_key
    from phase1.trading_week import trading_week

    cycle_monday = date.fromisoformat(get_monthly_em_date_key(now_et))
    return trading_week(cycle_monday).first_session_day == now_et.date()


def _apply_typed_em_snapshot(em_live_data, is_market_open, spot, ticker, em_type, date_key, should_freeze):
    """
    Generic EM snapshot freeze/restore for any em_type (daily/weekly/monthly).

    em_live_data: dict with expected_move_pts, upper_level, etc. (or None)
    Returns the frozen snapshot dict, or None if unavailable.
    """
    sk_snap = f"em_snapshot_{em_type}_{ticker}"
    sk_date = f"em_snapshot_date_{em_type}_{ticker}"
    sk_time = f"em_snapshot_time_{em_type}_{ticker}"
    # "We already looked in Postgres for today's snapshot and the row was
    # missing" marker. Without this, a missing snapshot re-fires the
    # get_em_snapshot query on every Streamlit rerun (every widget
    # interaction + every auto-refresh tick), and get_em_snapshot opens a
    # fresh psycopg2 connection each time — Neon's free tier runs out of
    # CU-hours fast under that load.
    sk_miss = f"em_snapshot_missing_{em_type}_{ticker}"

    # Clear stale snapshot from a previous period
    if st.session_state.get(sk_date) != date_key:
        st.session_state.pop(sk_snap, None)
        st.session_state.pop(sk_date, None)
        st.session_state.pop(sk_time, None)
        st.session_state.pop(sk_miss, None)

    # Try to restore from Postgres whenever session state is empty — this must
    # run regardless of market-open status, otherwise fresh sessions outside
    # 9:30–16:00 ET never pick up the frozen weekly/monthly snap that the
    # scheduled cron already persisted, and downstream code falls back to the
    # live (drifting) EM.
    if sk_snap not in st.session_state and st.session_state.get(sk_miss) != date_key:
        try:
            db_snap = get_em_snapshot(date_key, ticker=ticker, em_type=em_type)
        except Exception:
            db_snap = None
        if db_snap and db_snap.get("expected_move_pts"):
            st.session_state[sk_snap] = db_snap
            st.session_state[sk_date] = date_key
            captured = db_snap.get("captured_at", "")
            if captured:
                try:
                    from datetime import datetime as dt_cls
                    cap_dt = dt_cls.fromisoformat(captured)
                    st.session_state[sk_time] = cap_dt.strftime("%I:%M:%S %p ET")
                except Exception:
                    st.session_state[sk_time] = "restored from DB"
            else:
                st.session_state[sk_time] = "restored from DB"
        else:
            # Remember that the DB was empty for this date so we don't
            # re-query every rerun. A successful capture later in this
            # function will set sk_snap and effectively override the miss.
            st.session_state[sk_miss] = date_key

    if not is_market_open:
        # Outside market hours we never capture anything new — just return
        # whatever we restored (or None if nothing was ever saved).
        return st.session_state.get(sk_snap)

    # First capture of the period — only on the correct freeze day. Session
    # state only: the cron owns the persisted opening snapshot.
    if sk_snap not in st.session_state and should_freeze and em_live_data and em_live_data.get("expected_move_pts"):
        snap = {
            "expected_move_pts": em_live_data.get("expected_move_pts"),
            "expected_move_pct": em_live_data.get("expected_move_pct"),
            "upper_level": em_live_data.get("upper_level"),
            "lower_level": em_live_data.get("lower_level"),
            "straddle": em_live_data.get("straddle"),
            "expiration": em_live_data.get("expiration"),
        }
        st.session_state[sk_snap] = snap
        st.session_state[sk_date] = date_key
        st.session_state[sk_time] = now_ny().strftime("%I:%M:%S %p ET")

    return st.session_state.get(sk_snap)


def _apply_em_snapshot(em_analysis, is_market_open, regime, levels, spot, ticker="SPX"):
    """Freeze expected move at first market-hours refresh; recompute classification with live data."""
    today_str = now_ny().strftime("%Y-%m-%d")
    em_live = em_analysis.get("expected_move", {})

    snap = _apply_typed_em_snapshot(em_live, is_market_open, spot, ticker, "daily", today_str, True)

    if snap and snap.get("expected_move_pts"):
        em_pct = snap.get("expected_move_pct") or em_analysis.get("expected_move", {}).get("expected_move_pct")
        em_analysis["expected_move"] = {
            **em_analysis.get("expected_move", {}),
            "expected_move_pts": snap["expected_move_pts"],
            "expected_move_pct": em_pct,
            "upper_level": snap["upper_level"],
            "lower_level": snap["lower_level"],
            "straddle": snap.get("straddle"),
        }
        on_pts = em_analysis.get("overnight_move", {}).get("overnight_move_pts")
        if on_pts is not None and snap["expected_move_pts"] > 0:
            from phase1.expected_move import classify_session
            em_analysis["classification"] = classify_session(
                expected_move_pts=snap["expected_move_pts"],
                overnight_move_pts=on_pts,
                gamma_regime=regime["regime"],
            )
            em_analysis["classification"]["move_source"] = ticker.lower()
        if snap.get("upper_level") is not None:
            em_analysis["level_context"] = {
                "em_upper": snap["upper_level"],
                "em_lower": snap["lower_level"],
                "zero_gamma": round(levels["zero_gamma"], 2),
                "zero_gamma_within_em": (
                    snap["lower_level"] <= levels["zero_gamma"] <= snap["upper_level"]
                ),
                "zero_gamma_distance_to_spot": round(spot - levels["zero_gamma"], 2),
            }

    return em_analysis
