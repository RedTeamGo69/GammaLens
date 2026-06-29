"""
Native Streamlit control widgets for the terminal redesign.

These replace the anchor / query-param controls so every interaction travels
over Streamlit's websocket (a partial rerun that patches the page in place)
instead of a hard ``<a href="?…">`` navigation that reloads the whole SPA and
flashes the screen white.

Pure UI — no data/model logic. Each renderer reads/writes ``st.session_state``
and returns the resolved value(s) that ``streamlit_app.main()`` feeds into the
(unchanged) data pipeline. State lives in session, not the URL.
"""
from __future__ import annotations

import streamlit as st

from ui_theme import QUICK_TICKERS, EXP_MODES, REFRESH_MODES, TABS

# Canonical "currently active ticker" — written by the search/pill callbacks,
# read at the top of main() each rerun (callbacks fire before the script body).
ACTIVE_TICKER = "active_ticker"

_EXP_TOKENS = [t for t, _ in EXP_MODES]
_EXP_LABELS = dict(EXP_MODES)
_REFRESH_TOKENS = [t for t, _ in REFRESH_MODES]
_REFRESH_LABELS = dict(REFRESH_MODES)
_TAB_TOKENS = [t for t, _ in TABS]
_TAB_LABELS = dict(TABS)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks (run at the start of the rerun, before the script body)
# ─────────────────────────────────────────────────────────────────────────────
def _cb_search() -> None:
    val = (st.session_state.get("ticker_search") or "").strip().upper()
    if val:
        st.session_state[ACTIVE_TICKER] = val
        st.session_state["ticker_search"] = ""  # clear the box on submit


def _cb_quick() -> None:
    val = st.session_state.get("quick_pills")
    if val:
        st.session_state[ACTIVE_TICKER] = val


def _cb_recent() -> None:
    val = st.session_state.get("recent_pills")
    if val:
        st.session_state[ACTIVE_TICKER] = val


def _cb_refresh_no_deselect() -> None:
    """Keep the auto-refresh control single-select (radio-like).

    ``st.segmented_control`` lets the user click the active pill to clear the
    selection, which left the control showing nothing while the app silently
    kept the old cadence. Snap an empty selection back to the previous choice so
    only the explicit "Off" pill disables refresh.
    """
    if st.session_state.get("refresh_seg") is None:
        st.session_state["refresh_seg"] = st.session_state.get("_refresh_last", "off")


# ─────────────────────────────────────────────────────────────────────────────
# Small HTML label helper (eyebrows match the .card-eyebrow look)
# ─────────────────────────────────────────────────────────────────────────────
def _eyebrow(text: str) -> None:
    st.markdown(f'<div class="ctl-eyebrow">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Settings controls (rendered inside the sidebar settings card)
# ─────────────────────────────────────────────────────────────────────────────
def render_settings_controls(ticker: str, ticker_type: str,
                             recents: list[str]) -> tuple[str, str, str, str]:
    """Render instrument + expiration + auto-refresh controls as native widgets.

    ``ticker`` is the already-validated active symbol; the pills are pre-set to
    reflect it so the highlight always tracks the resolved ticker (and a search
    that lands on a non-pick symbol leaves both pill groups unselected).

    Returns ``(exp_token, refresh_token, cal_start, cal_end)``.
    """
    # ── Instrument ──
    _eyebrow("Instrument")
    st.text_input(
        "Search symbol", key="ticker_search", on_change=_cb_search,
        placeholder="Search any symbol — AAPL, NVDA, IWM…",
        label_visibility="collapsed",
    )

    # Quick picks — pre-set so the active ticker shows selected (or none, if the
    # active symbol isn't a quick pick).
    st.session_state["quick_pills"] = ticker if ticker in QUICK_TICKERS else None
    st.pills(
        "Quick picks", QUICK_TICKERS, selection_mode="single",
        key="quick_pills", on_change=_cb_quick, label_visibility="collapsed",
    )

    # Recent — its own labelled group (quick picks already shown above).
    if recents:
        _eyebrow("Recent")
        st.session_state["recent_pills"] = ticker if ticker in recents else None
        st.pills(
            "Recent", recents, selection_mode="single",
            key="recent_pills", on_change=_cb_recent, label_visibility="collapsed",
        )

    st.markdown(
        f'<div class="ctl-active">Active: '
        f'<span class="t">{ticker}</span> '
        f'<span class="ty">· {ticker_type}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Expiration ──
    _eyebrow("Expiration")
    exp_sel = st.pills(
        "Expiration", _EXP_TOKENS, selection_mode="single",
        default=st.session_state.get("_exp_last", "0dte"),
        format_func=lambda t: _EXP_LABELS[t],
        key="exp_pills", label_visibility="collapsed",
    )
    exp_token = exp_sel or st.session_state.get("_exp_last", "0dte")
    st.session_state["_exp_last"] = exp_token

    # ── Custom date range (only when expiration == Custom) ──
    cal_start = cal_end = ""
    if exp_token == "custom":
        rng = st.date_input(
            "Custom range", value=(), key="cal_range",
            format="YYYY-MM-DD", label_visibility="collapsed",
        )
        if isinstance(rng, (list, tuple)):
            if len(rng) >= 1 and rng[0]:
                cal_start = rng[0].isoformat()
            if len(rng) >= 2 and rng[1]:
                cal_end = rng[1].isoformat()

    # ── Auto-refresh ──
    _eyebrow("Auto-refresh")
    # Seed once via session_state (not ``default=``) so the no-deselect callback
    # can own the widget's value without fighting a per-run default.
    if "refresh_seg" not in st.session_state:
        st.session_state["refresh_seg"] = st.session_state.get("_refresh_last", "off")
    st.segmented_control(
        "Auto-refresh", _REFRESH_TOKENS, selection_mode="single",
        format_func=lambda t: _REFRESH_LABELS[t],
        key="refresh_seg", label_visibility="collapsed",
        on_change=_cb_refresh_no_deselect,
    )
    refresh_token = st.session_state["refresh_seg"] or "off"
    st.session_state["_refresh_last"] = refresh_token

    return exp_token, refresh_token, cal_start, cal_end


def render_refresh_button() -> None:
    """A native 'refresh data' button — clears the cache and reruns in place."""
    if st.button("⟳ Refresh data", key="refresh_now_btn", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab control (rendered in the main column, above the tab content)
# ─────────────────────────────────────────────────────────────────────────────
def render_tab_control() -> str:
    """Render the Strike GEX / Spread Finder / 0DTE Finder selector. Returns the
    active tab token (``gex`` | ``spread`` | ``0dte``)."""
    sel = st.segmented_control(
        "View", _TAB_TOKENS, selection_mode="single",
        default=st.session_state.get("_tab_last", "gex"),
        format_func=lambda t: _TAB_LABELS[t],
        key="tab_seg", label_visibility="collapsed",
    )
    tab = sel or st.session_state.get("_tab_last", "gex")
    st.session_state["_tab_last"] = tab
    return tab
