"""
Pre-Flight tab — last-look gate for a proposed options trade.

Three entry paths feed one normalized trade payload: the Cockpit handoff
(session state), the last persisted cockpit proposal (Neon — survives a
Streamlit Cloud session reset), and a manual structure-preset form. On "Run
Pre-Flight" the pure gate engine (range_finder.preflight_engine) evaluates
behavioral / model / structure / broker gates and the evaluation is appended
to ``preflight_log``.

Degraded-mode honesty: until the Public.com history pipeline is wired, the
behavioral gate reports "na — not configured" (via a guarded import) and the
broker preflight card reports "unavailable". "na" gates are itemized and
never block a GREEN — unavailable is a state, not a pass or a fail.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from theme import COLORS, SF_BEAR, SF_BULL, SF_CARD, SF_WARN
from models import GEXData
from phase1.data_client import TradierDataClient
from phase1.gex_history import get_em_snapshot, get_weekly_em_date_key
from phase1.market_clock import now_ny
from phase1.ticker_config import feature_source_ticker
from range_finder.cockpit_config import COCKPIT_MODEL_SPEC, COCKPIT_TICKER, CockpitConfig
from range_finder.cockpit_engine import build_leg_quote_map
from range_finder.har_model import forecast_next_week
from range_finder.preflight_engine import (
    PreflightReport,
    ProposedLeg,
    ProposedTrade,
    category_key,
    report_to_row,
    run_preflight,
)
from range_finder.verdict_persistence import load_last_proposal, save_preflight_log
from ui_theme import esc
from ui_cockpit import _planning_week
from ui_spread_finder import (
    _cached_rf_get_features,
    _cached_rf_load_model,
    _cached_weekly_setup,
    _get_rf_conn,
    _tradier_token,
    find_spread_finder_friday_exp,
)

log = logging.getLogger(__name__)

_TXT = COLORS.get("text_secondary", "#cbd5e1")
_MUT = COLORS.get("text_muted", "#93a1b2")

_STATUS_STYLE = {
    "green": (SF_BULL, "PASS"),
    "yellow": (SF_WARN, "WARN"),
    "red": (SF_BEAR, "FLAG"),
    "na": (_MUT, "N/A"),
}
_VERDICT_STYLE = {
    "GREEN": (SF_BULL, "rgba(61,220,151,.10)", "All gates pass."),
    "YELLOW": (SF_WARN, "rgba(255,180,84,.10)", "Warnings present — read them before submitting."),
    "RED": (SF_BEAR, "rgba(255,105,110,.10)", "Hard flag raised — the tool confronts, you decide."),
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached market lookups
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _cached_pf_expirations(tradier_token: str, ticker: str) -> list[str]:
    try:
        return TradierDataClient(tradier_token).get_expirations(ticker)
    except Exception as e:
        log.warning(f"preflight expirations failed for {ticker}: {e}")
        return []


@st.cache_data(ttl=90, show_spinner=False)
def _cached_pf_chain(tradier_token: str, ticker: str, exp: str, run_key: str) -> dict:
    out: dict = {"quote_map": {}, "spot": None, "error": None}
    try:
        client = TradierDataClient(tradier_token)
        entry = client.get_chain_cached(ticker, exp)
        if entry.get("status") == "ok":
            out["quote_map"] = build_leg_quote_map(entry["calls"], entry["puts"])
        else:
            out["error"] = entry.get("error") or "chain fetch failed"
        spot = float(client.get_spot_price(ticker) or 0.0)
        out["spot"] = spot if spot > 0 else None
    except Exception as e:
        out["error"] = str(e)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _cached_category_stats(_conn, category: str) -> dict | None:
    """Behavioral stats for one category key; None until history is synced
    (the gate reads that as "na"). The Sync button clears this cache so
    fresh fills show immediately."""
    try:
        from public_api.stats import load_category_stats
    except ImportError:
        return None
    try:
        return load_category_stats(_conn, category)
    except Exception as e:
        log.warning(f"category stats load failed: {e}")
        return None


def _load_behavioral_stats(conn, category: str) -> dict | None:
    return _cached_category_stats(conn, category)


def _render_public_history_panel(conn) -> None:
    """Sync + status for the fill history that feeds the behavioral gate.
    Rendered whether or not a trade is loaded — you sync BEFORE trading."""
    with st.expander("Public.com fill history (behavioral gate data)",
                     expanded=False):
        from public_api.client import PublicClient, get_public_credentials
        from public_api.store import history_summary, load_review_queue
        from public_api.sync import sync_public_history

        secret, account = get_public_credentials()
        summ = history_summary(conn, account or None)
        st.caption(f"{summ['strategies']} strategies ({summ['closed']} closed) "
                   f"· {summ['review']} fills in review · last sync "
                   f"{(summ['last_sync'] or '—')[:19]}")
        if secret and account:
            if st.button("⟳ Sync fills from Public", key="pf_sync"):
                with st.spinner("Pulling fill history…"):
                    res = sync_public_history(conn, PublicClient(secret, account))
                if res is None:
                    st.warning("Public API unreachable or unauthorized — "
                               "nothing synced.")
                else:
                    _cached_category_stats.clear()
                    st.success(
                        f"Synced {res.strategies} strategies ({res.closed} "
                        f"closed) from {res.trade_fills} fills — coverage "
                        f"{res.coverage:.1%}, {res.ambiguous} ambiguous.")
                    for err in res.errors:
                        st.caption(f"⚠ {err}")
        else:
            st.caption("`PUBLIC_API_SECRET` / `PUBLIC_ACCOUNT_ID` are not in "
                       "secrets — the sync button appears once they are. "
                       "Already-synced stats still feed the gate.")
        review = load_review_queue(conn)
        if review:
            st.caption("Ambiguous fills awaiting review (never force-grouped):")
            st.dataframe(pd.DataFrame(review), use_container_width=True,
                         hide_index=True)


def _pf_anchor(conn, ticker: str, week_start: str, run_now) -> float | None:
    """Frozen Monday anchor for the trade's ticker (session freeze → DB row).
    No self-heal here — a missing anchor turns the model gate 'na', which is
    the honest answer for a ticker the Monday cron never anchored."""
    if st.session_state.get(f"sf_monday_open_week_{ticker}") == run_now.isocalendar()[1]:
        v = st.session_state.get(f"sf_monday_open_{ticker}")
        if v:
            return float(v)
    try:
        row = _cached_weekly_setup(conn, week_start, ticker)
        if row and row[0]:
            return float(row[0])
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# div+CSS renderers
# ─────────────────────────────────────────────────────────────────────────────

def _trade_summary_html(payload: dict) -> str:
    legs = payload.get("legs") or []
    leg_bits = " · ".join(
        f'<span style="color:{SF_BEAR if l["direction"] == "sell" else SF_BULL};">'
        f'{l["direction"].upper()} {l["option_type"].upper()} {l["strike"]:g}</span>'
        for l in legs)
    src = payload.get("source", "manual")
    return (
        f'<div style="background:{SF_CARD};border:1px solid rgba(147,161,178,.25);'
        f'border-radius:10px;padding:10px 14px;margin:.3rem 0;">'
        f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.08em;color:{_MUT};">'
        f'TRADE UNDER REVIEW · from {esc(str(src))}</div>'
        f'<div style="color:{_TXT};font-size:.95rem;margin-top:3px;">'
        f'<b>{esc(str(payload.get("ticker", "?")))}</b> · exp {esc(str(payload.get("expiration", "?")))}'
        f' · qty {payload.get("quantity", 1)}<br>{leg_bits}</div></div>'
    )


def _verdict_banner_html(report: PreflightReport) -> str:
    accent, bg, sub = _VERDICT_STYLE.get(report.verdict, (_MUT, "transparent", ""))
    chips = "".join(
        f'<span style="display:inline-block;margin-right:8px;padding:1px 8px;'
        f'border:1px solid {_STATUS_STYLE[g.status][0]};border-radius:5px;'
        f'font-size:.7rem;font-weight:700;color:{_STATUS_STYLE[g.status][0]};">'
        f'{g.name.upper()} {_STATUS_STYLE[g.status][1]}</span>'
        for g in report.gates)
    return (
        f'<div style="border:1px solid {accent};border-left:6px solid {accent};'
        f'background:{bg};border-radius:10px;padding:12px 16px;margin:.3rem 0 .6rem 0;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:{accent};letter-spacing:.05em;">'
        f'{report.verdict} <span style="font-size:.85rem;font-weight:600;color:{_TXT};">'
        f'· {esc(report.category)}</span></div>'
        f'<div style="color:{_MUT};font-size:.88rem;margin:3px 0 6px 0;">{esc(sub)}</div>'
        f'<div>{chips}</div></div>'
    )


def _gate_card_html(g) -> str:
    color, tag = _STATUS_STYLE.get(g.status, (_MUT, g.status))
    items = "".join(
        f'<li style="margin:.22rem 0;color:{_TXT};font-size:.84rem;line-height:1.35;">'
        f'{esc(r)}</li>' for r in g.reasons)
    return (
        f'<div style="background:{SF_CARD};border:1px solid {color};border-radius:10px;'
        f'padding:10px 14px;height:100%;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:.72rem;font-weight:700;letter-spacing:.08em;color:{_MUT};">'
        f'{g.name.upper()} GATE</span>'
        f'<span style="font-size:.7rem;font-weight:800;color:{color};border:1px solid {color};'
        f'border-radius:5px;padding:0 6px;">{tag}</span></div>'
        f'<ul style="list-style:none;padding-left:0;margin:.35rem 0 0 0;">{items}</ul></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trade input paths
# ─────────────────────────────────────────────────────────────────────────────

def _manual_trade_form(default_exp: str) -> dict | None:
    """Structure-preset entry form → normalized trade payload on submit."""
    with st.form("pf_manual_trade", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1, 1.2, 0.8, 1.4])
        with c1:
            ticker = st.text_input("Ticker", value=COCKPIT_TICKER, key="pf_ticker")
        with c2:
            exp = st.date_input("Expiration", value=date.fromisoformat(default_exp),
                                key="pf_exp")
        with c3:
            qty = st.number_input("Qty", min_value=1, value=1, key="pf_qty")
        with c4:
            preset = st.selectbox("Structure", [
                "Iron condor", "Vertical (put credit)", "Vertical (call credit)",
                "Vertical (put debit)", "Vertical (call debit)",
                "Single leg (buy call)", "Single leg (buy put)",
                "Single leg (sell call)", "Single leg (sell put)",
            ], key="pf_preset")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            k1 = st.number_input("Strike A", value=0.0, step=0.5, key="pf_k1",
                                 help="Condor: long put · Vertical/single: first strike")
        with s2:
            k2 = st.number_input("Strike B", value=0.0, step=0.5, key="pf_k2",
                                 help="Condor: short put · Vertical: second strike")
        with s3:
            k3 = st.number_input("Strike C", value=0.0, step=0.5, key="pf_k3",
                                 help="Condor: short call")
        with s4:
            k4 = st.number_input("Strike D", value=0.0, step=0.5, key="pf_k4",
                                 help="Condor: long call")
        submitted = st.form_submit_button("Load trade")

    if not submitted:
        return None
    t = (ticker or "").strip().upper()
    legs: list[dict] = []
    if preset == "Iron condor":
        if not all(v > 0 for v in (k1, k2, k3, k4)):
            st.warning("An iron condor needs all four strikes (A–D).")
            return None
        legs = [{"option_type": "put", "direction": "buy", "strike": k1},
                {"option_type": "put", "direction": "sell", "strike": k2},
                {"option_type": "call", "direction": "sell", "strike": k3},
                {"option_type": "call", "direction": "buy", "strike": k4}]
    elif preset.startswith("Vertical"):
        if not (k1 > 0 and k2 > 0):
            st.warning("A vertical needs strikes A and B.")
            return None
        opt = "put" if "put" in preset else "call"
        credit = "credit" in preset
        if opt == "call":
            sell_k, buy_k = (min(k1, k2), max(k1, k2)) if credit else (max(k1, k2), min(k1, k2))
        else:
            sell_k, buy_k = (max(k1, k2), min(k1, k2)) if credit else (min(k1, k2), max(k1, k2))
        legs = [{"option_type": opt, "direction": "sell", "strike": sell_k},
                {"option_type": opt, "direction": "buy", "strike": buy_k}]
    else:  # single legs
        if not k1 > 0:
            st.warning("A single leg needs Strike A.")
            return None
        opt = "put" if "put" in preset else "call"
        direction = "sell" if "sell" in preset else "buy"
        legs = [{"option_type": opt, "direction": direction, "strike": k1}]

    return {"source": "manual", "ticker": t, "expiration": exp.isoformat(),
            "quantity": int(qty), "legs": legs}


# ─────────────────────────────────────────────────────────────────────────────
# The tab
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment
def _render_preflight_tab(
    spot: float,
    levels: dict,
    regime: dict,
    data: GEXData,
    ticker: str = "SPX",
    weekly_em: dict | None = None,
) -> None:
    run_now = now_ny()
    week_start, _monday = _planning_week(run_now)
    conn = _get_rf_conn()
    token = _tradier_token()

    # ── Entry paths ──
    payload = st.session_state.get("preflight_trade")
    top = st.columns([1, 1, 2])
    with top[0]:
        if st.button("↩ Load last cockpit proposal", key="pf_load_last"):
            try:
                last = load_last_proposal(conn, COCKPIT_TICKER)
            except Exception as e:
                last = None
                st.warning(f"Could not load: {e}")
            if last and last.get("proposal"):
                p = last["proposal"]
                st.session_state["preflight_trade"] = {
                    "source": f"cockpit log ({last.get('week_start')})",
                    "ticker": COCKPIT_TICKER,
                    "expiration": p.get("expiration"),
                    "quantity": 1,
                    "legs": [
                        {"option_type": "put", "direction": "buy", "strike": p.get("put_long")},
                        {"option_type": "put", "direction": "sell", "strike": p.get("put_short")},
                        {"option_type": "call", "direction": "sell", "strike": p.get("call_short")},
                        {"option_type": "call", "direction": "buy", "strike": p.get("call_long")},
                    ],
                }
                payload = st.session_state["preflight_trade"]
            elif last is None:
                st.info("No persisted cockpit proposal found yet.")
    with top[1]:
        if payload and st.button("✕ Clear trade", key="pf_clear"):
            st.session_state.pop("preflight_trade", None)
            st.session_state.pop("preflight_last_report", None)
            payload = None

    default_friday = (find_spread_finder_friday_exp(
        _cached_pf_expirations(token, COCKPIT_TICKER), run_now.date())
        or (run_now.date() + timedelta(days=(4 - run_now.weekday()) % 7)).isoformat())
    with st.expander("Manual trade entry", expanded=payload is None):
        manual = _manual_trade_form(default_friday)
        if manual:
            st.session_state["preflight_trade"] = manual
            payload = manual

    _render_public_history_panel(conn)

    if not payload:
        st.info("Send a proposal from the **Monday Cockpit**, load the last "
                "persisted one, or enter a trade manually — then run Pre-Flight.")
        return

    st.html(_trade_summary_html(payload))

    # ── Evaluate on demand (each run logs one row — volume stays honest) ──
    if st.button("🛫 Run Pre-Flight", key="pf_run", type="primary"):
        t = str(payload.get("ticker") or "").upper()
        exp = str(payload.get("expiration") or "")
        legs = tuple(ProposedLeg(option_type=l["option_type"],
                                 direction=l["direction"],
                                 strike=float(l["strike"]))
                     for l in payload.get("legs") or [])
        trade = ProposedTrade(ticker=t, expiration=exp,
                              quantity=int(payload.get("quantity") or 1), legs=legs)
        try:
            dte = (date.fromisoformat(exp) - run_now.date()).days
        except ValueError:
            dte = None

        friday_exp = find_spread_finder_friday_exp(
            _cached_pf_expirations(token, t), run_now.date())
        chain = _cached_pf_chain(token, t, exp, data.run_time) if exp else {}

        anchor = None
        forecast = None
        if friday_exp and exp == friday_exp:
            anchor = _pf_anchor(conn, t, week_start, run_now)
            if anchor:
                try:
                    src = feature_source_ticker(t)
                    payload_model = _cached_rf_load_model(COCKPIT_MODEL_SPEC, src)
                    df_feat = _cached_rf_get_features(conn, src)
                    wk_ts = pd.Timestamp(week_start)
                    row = (df_feat.loc[wk_ts] if wk_ts in df_feat.index
                           else df_feat.iloc[-1])
                    forecast = forecast_next_week(
                        payload_model["result"], row,
                        payload_model["feature_cols"], anchor)
                except Exception as e:
                    log.warning(f"preflight forecast unavailable for {t}: {e}")

        em_pts = None
        try:
            snap = get_em_snapshot(get_weekly_em_date_key(run_now), t, "weekly")
            if snap and snap.get("expected_move_pts"):
                em_pts = float(snap["expected_move_pts"])
        except Exception:
            pass

        cfg = CockpitConfig()
        category = category_key(trade, dte)
        stats = _load_behavioral_stats(conn, category)

        report = run_preflight(
            trade,
            stats=stats,
            forecast=forecast,
            anchor=anchor,
            week_friday_exp=friday_exp,
            chain_quotes=chain.get("quote_map"),
            em_pts=em_pts,
            public_response=None,   # broker preflight wired in a later step
            cfg=cfg,
            dte=dte,
        )
        st.session_state["preflight_last_report"] = report
        try:
            save_preflight_log(conn, report_to_row(report, week_start))
        except Exception as e:
            log.warning(f"preflight log failed: {e}")
        if chain.get("error"):
            st.caption(f"⚠ chain: {chain['error']}")

    report = st.session_state.get("preflight_last_report")
    if isinstance(report, PreflightReport):
        st.html(_verdict_banner_html(report))
        cols = st.columns(2)
        for i, g in enumerate(report.gates):
            with cols[i % 2]:
                st.html(_gate_card_html(g))
        st.caption(f"category `{report.category}` · DTE {report.dte} · "
                   f"evaluated {report.checked_at[:19]} UTC · logged as "
                   f"{report.input_hash}")
