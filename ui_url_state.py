"""URL-backed persistence for Gamma Lens navigation state.

Streamlit Community Cloud recycles the app process (idle timeout, memory
pressure) and a websocket reconnect starts a brand-new session — clearing
``st.session_state``.  The native-widget UI keeps ALL navigation state (active
ticker, active tab, expiration mode, recents, export extras) in session_state
ONLY, so a reset drops the user back to SPX / GEX / no-recents mid-workflow.
That single event reads to the user as three separate bugs (recents vanished,
export tickers vanished, tab jumped back to GEX) — it's one lost session.

This module mirrors that state into ``st.query_params`` on every run and
rehydrates it from the URL once per session, restoring the reload-resilience
the pre-refactor query-param UI had — without giving up the smooth
native-widget websocket reruns.  (The pre-refactor design already proved the
URL is a safe home for ticker/exp/tab; this just re-homes it there.)

Two entry points, both called from ``streamlit_app.main()``:
  • ``rehydrate_from_url()`` — near the top, before the nav widgets: seeds
    session_state from ?params for any managed key not already set this
    session (a live widget interaction always wins over the URL).
  • ``sync_to_url()`` — at the end: writes the current nav state back to the
    URL, but only when it actually changed (stable URL, no write/rerun loop).

Only presentation/navigation state is persisted — no secrets, no model data.
Each scalar is round-tripped verbatim; the consuming widget sanitizes its own
token (``render_tab_control`` for the tab, ``render_settings_controls`` for
the expiration), so a stale or tampered value degrades gracefully instead of
crashing a widget.  Tickers flow through the existing Tradier validation.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping

# session_state key  ->  short URL query-param name.
_SCALAR_KEYS = {
    "active_ticker": "t",
    "_tab_last": "tab",
    "_exp_last": "exp",
}
# List-valued keys, serialized comma-joined.
_LIST_KEYS = {
    "recent_tickers": "recents",
    "_sf_xlsx_extra": "xtra",
}
_LIST_SEP = ","
_MAX_LIST_ITEMS = 40          # hard bound so a tampered URL can't bloat state
_REHYDRATED_FLAG = "_url_state_rehydrated"

_ALL_PARAMS = list(_SCALAR_KEYS.values()) + list(_LIST_KEYS.values())


def _parse_list(raw: "str | None") -> list[str]:
    """Comma string -> deduped, upper-cased ticker list (bounded)."""
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(_LIST_SEP):
        t = part.strip().upper()
        if t and t not in out:
            out.append(t)
        if len(out) >= _MAX_LIST_ITEMS:
            break
    return out


def _serialize_list(vals) -> str:
    """Ticker list -> comma string (order preserved)."""
    return _LIST_SEP.join(str(v) for v in (vals or []))


def _rehydrate(session: MutableMapping, params: Mapping) -> None:
    """Seed ``session`` from ``params`` once per session — never clobber a value
    the session already holds (a live widget interaction wins over the URL)."""
    if session.get(_REHYDRATED_FLAG):
        return
    session[_REHYDRATED_FLAG] = True
    for sk, pk in _SCALAR_KEYS.items():
        if sk not in session:
            val = params.get(pk)
            if val:
                session[sk] = val
    for sk, pk in _LIST_KEYS.items():
        if sk not in session:
            items = _parse_list(params.get(pk))
            if items:
                session[sk] = items


def _compute_url(session: Mapping) -> dict[str, str]:
    """The managed query params reflecting the current nav state.  Empty /
    missing values are omitted so the URL only carries what's actually set."""
    out: dict[str, str] = {}
    for sk, pk in _SCALAR_KEYS.items():
        val = session.get(sk)
        if val:
            out[pk] = str(val)
    for sk, pk in _LIST_KEYS.items():
        s = _serialize_list(session.get(sk))
        if s:
            out[pk] = s
    return out


def rehydrate_from_url() -> None:
    """Streamlit entry point: seed session_state from ``st.query_params``.

    Call once near the top of ``main()``, before the nav setdefaults and the
    nav widgets, so a URL value takes precedence over the hard-coded defaults.
    """
    import streamlit as st
    _rehydrate(st.session_state, st.query_params)


def sync_to_url() -> None:
    """Streamlit entry point: mirror nav state into ``st.query_params``.

    Call at the end of ``main()``.  Writes only when the managed params
    actually changed — assigning to ``st.query_params`` every run is pointless
    churn, and change-gating also rules out any write/rerun feedback loop.
    """
    import streamlit as st
    desired = _compute_url(st.session_state)
    qp = st.query_params
    if all(qp.get(pk) == desired.get(pk) for pk in _ALL_PARAMS):
        return
    for pk in _ALL_PARAMS:
        if pk in desired:
            qp[pk] = desired[pk]
        elif pk in qp:
            del qp[pk]
