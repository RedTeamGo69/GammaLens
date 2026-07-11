"""
Monday Cockpit tab — weekly XSP iron-condor decision surface.

One screen answers "should I trade this week, and if so, at what strikes?":
the weekly HAR forecast band vs the straddle-implied move (VRP hard gate),
the Friday-expiry GEX walls (outward-only strike snapping), the week's macro
events (skip/widen policy), and a priced condor proposal — rolled into a
single TRADE/SKIP verdict with reasons always shown. Every rendered verdict
is hash-dedupe logged to ``cockpit_verdicts`` so the VRP gate can be audited
against realized outcomes later (the 0DTE banner lesson).

The Cockpit is XSP-fixed by design (``cockpit_config.COCKPIT_TICKER``): it
fetches its own Friday-expiry XSP bundle regardless of which ticker the rest
of the dashboard is showing. All heavy lifting lives in pure modules —
``range_finder.cockpit_engine`` decides, this file only assembles inputs and
renders div+CSS HTML (no SVG: st.html strips it; no new Plotly).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import timedelta

import pandas as pd
import streamlit as st

from theme import COLORS, SF_BEAR, SF_BULL, SF_CARD, SF_NEUT, SF_WARN
from models import GEXData
from phase1.data_client import TradierDataClient
from phase1.expected_move import compute_expected_move, find_atm_straddle
from phase1.gex_history import get_em_snapshot, get_weekly_em_date_key
from phase1.market_clock import (
    compute_time_to_expiry_years,
    now_ny,
    trading_days_between,
)
from phase1.ticker_config import get_config
# gex_engine re-exports find_key_levels / get_gamma_regime_text from the
# split modules; importing THROUGH it (never phase1.key_levels directly)
# avoids the key_levels → zero_gamma → gex_engine import cycle.
import phase1.gex_engine as gex_engine
from phase1.gex_engine import find_key_levels, get_gamma_regime_text
from range_finder.cockpit_config import (
    COCKPIT_MODEL_SPEC,
    COCKPIT_TICKER,
    CockpitConfig,
)
from range_finder.cockpit_engine import (
    CockpitVerdict,
    build_leg_quote_map,
    evaluate_cockpit,
)
from range_finder.conformal import maybe_apply_conformal
from range_finder.data_collector import capture_and_save_monday_anchor
from range_finder.event_calendars import events_for_week
from range_finder.gex_bridge import reconcile_gex_warnings
from range_finder.har_model import forecast_next_week
from range_finder.verdict_persistence import load_recent_verdicts, save_cockpit_verdict
from ui_theme import esc
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
_DIM = COLORS.get("text_dim", "#5b6878")
_BLU = COLORS.get("blue", "#6ea8ff")


# ─────────────────────────────────────────────────────────────────────────────
# Cached data assembly
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=90, show_spinner=False)
def _cached_cockpit_bundle(tradier_token: str, run_key: str, rfr: float) -> dict:
    """XSP Friday-expiry bundle: spot, chain quote-map, live straddle EM, GEX
    walls + regime. Keyed to the main pipeline's run_time (run_key) so its
    freshness tracks the dashboard's 90-second refresh cycle. Every fetch
    degrades gracefully — the engine turns missing pieces into named SKIPs.
    """
    out: dict = {"spot": None, "spot_source": "", "friday_exp": None,
                 "quote_map": {}, "gex_levels": None, "regime": None,
                 "live_em": None, "errors": [], "run_key": run_key}
    try:
        client = TradierDataClient(tradier_token)
    except Exception as e:  # pragma: no cover - constructor is trivial
        out["errors"].append(f"client: {e}")
        return out

    try:
        exps = client.get_expirations(COCKPIT_TICKER)
    except Exception as e:
        exps = []
        out["errors"].append(f"expirations: {e}")
    friday_exp = find_spread_finder_friday_exp(exps, now_ny().date()) if exps else None
    out["friday_exp"] = friday_exp

    spot = 0.0
    try:
        spot = float(client.get_spot_price(COCKPIT_TICKER) or 0.0)
    except Exception as e:
        out["errors"].append(f"quote: {e}")
    if spot > 0:
        out["spot"], out["spot_source"] = round(spot, 2), "Tradier XSP"
    else:
        # XSP quotes can be flaky far from RTH — derive from SPX/10, labeled.
        try:
            spx = float(client.get_spot_price("SPX") or 0.0)
            if spx > 0:
                out["spot"], out["spot_source"] = round(spx / 10.0, 2), "SPX/10 est."
        except Exception as e:
            out["errors"].append(f"spx fallback: {e}")
    spot = out["spot"] or 0.0

    if friday_exp and spot > 0:
        try:
            entry = client.get_chain_cached(COCKPIT_TICKER, friday_exp)
            if entry.get("status") == "ok":
                out["quote_map"] = build_leg_quote_map(entry["calls"], entry["puts"])
                straddle = find_atm_straddle(entry["calls"], entry["puts"], spot)
                out["live_em"] = compute_expected_move(straddle, spot)
            else:
                out["errors"].append(f"chain: {entry.get('error') or 'failed'}")
        except Exception as e:
            out["errors"].append(f"chain: {e}")

        try:
            gex_df, _stats, all_options, *_ = gex_engine.calculate_all(
                client, COCKPIT_TICKER, [friday_exp], spot, r=rfr)
            if gex_df is not None and not gex_df.empty and all_options:
                levels = find_key_levels(gex_df, spot, all_options=all_options, r=rfr)
                out["gex_levels"] = levels
                zg = levels.get("zero_gamma")
                if zg:
                    out["regime"] = get_gamma_regime_text(spot, zg)
        except Exception as e:
            out["errors"].append(f"gex: {e}")

    return out


@st.cache_data(ttl=900, show_spinner=False)
def _cached_xsp_weekly_em(date_key: str) -> dict | None:
    """Monday-frozen weekly EM snapshot for XSP (written by the 9:28 cron)."""
    return get_em_snapshot(date_key, COCKPIT_TICKER, "weekly")


def _planning_week(run_now) -> tuple:
    """Week the cockpit is deciding for: Mon–Thu → this week, Fri–Sun → next.
    Mirrors ui_spread_finder._spread_finder_target_friday so the two tabs
    always plan the same week."""
    today = run_now.date()
    wd = today.weekday()
    monday = today - timedelta(days=wd) if wd <= 3 else today + timedelta(days=7 - wd)
    return monday.strftime("%Y-%m-%d"), monday


def _resolve_anchor(conn, week_start: str, monday, run_now) -> tuple[float | None, str]:
    """Frozen Monday-open anchor for XSP: session freeze → weekly_setup row →
    retroactive self-heal capture. Never a live price — mid-week drift is the
    exact failure mode the Monday lock exists to prevent."""
    if st.session_state.get(f"sf_monday_open_week_{COCKPIT_TICKER}") == run_now.isocalendar()[1]:
        v = st.session_state.get(f"sf_monday_open_{COCKPIT_TICKER}")
        if v:
            return float(v), "Mon open (session freeze)"
    try:
        row = _cached_weekly_setup(conn, week_start, COCKPIT_TICKER)
        if row and row[0]:
            return float(row[0]), "Mon open (from DB)"
    except Exception:
        pass
    if monday <= run_now.date():
        try:
            healed_open, _vix, _src = capture_and_save_monday_anchor(
                conn, COCKPIT_TICKER, week_start, monday,
                spot_fallback=None, live_vix_fallback=None,
                cfg=get_config(COCKPIT_TICKER),
            )
            if healed_open:
                _cached_weekly_setup.clear()
                return float(healed_open), "Mon open (self-healed)"
        except Exception:
            pass
    return None, "unavailable"


def _resolve_straddle(ticker: str, weekly_em: dict | None, bundle: dict,
                      run_now) -> tuple[float | None, str]:
    """ATM straddle EM in points, Monday-frozen snapshot preferred over live."""
    if ticker == COCKPIT_TICKER and weekly_em and weekly_em.get("expected_move_pts"):
        return float(weekly_em["expected_move_pts"]), "weekly EM snapshot"
    try:
        snap = _cached_xsp_weekly_em(get_weekly_em_date_key(run_now))
    except Exception:
        snap = None
    if snap and snap.get("expected_move_pts"):
        return float(snap["expected_move_pts"]), "weekly EM snapshot"
    live = bundle.get("live_em") or {}
    if live.get("expected_move_pts"):
        return float(live["expected_move_pts"]), "live ATM straddle"
    return None, "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# div+CSS renderers (st.html strips SVG — everything is positioned divs)
# ─────────────────────────────────────────────────────────────────────────────

_SEV_STYLE = {"gate": (SF_BEAR, "GATE"), "warning": (SF_WARN, "WARN"),
              "info": (_BLU, "INFO")}


def _verdict_banner_html(verdict: CockpitVerdict, week_start: str) -> str:
    if verdict.verdict == "TRADE":
        accent, bg = SF_BULL, "rgba(61, 220, 151, .10)"
        label, sub = "TRADE", "All gates passed — proposal below."
    else:
        accent, bg = _MUT, "rgba(147, 161, 178, .10)"
        n = sum(1 for r in verdict.reasons if r.severity == "gate")
        label = "SKIP"
        sub = (f"{n} gate{'s' if n != 1 else ''} failed — reasons below. "
               f"Skipping is a result, not a failure.")
    return (
        f'<div style="border:1px solid {accent};border-left:6px solid {accent};'
        f'background:{bg};border-radius:10px;padding:12px 16px;margin:.2rem 0 .6rem 0;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:{accent};letter-spacing:.05em;">'
        f'{label} <span style="font-size:.9rem;font-weight:600;color:{_TXT};">'
        f'· {COCKPIT_TICKER} iron condor · week of {esc(week_start)}</span></div>'
        f'<div style="color:{_MUT};font-size:.9rem;margin-top:2px;">{esc(sub)}</div>'
        f'</div>'
    )


def _reasons_html(verdict: CockpitVerdict) -> str:
    if not verdict.reasons:
        return (f'<div style="color:{_MUT};font-size:.88rem;margin:.3rem 0;">'
                f'No warnings — clean pass on every gate.</div>')
    items = []
    order = {"gate": 0, "warning": 1, "info": 2}
    for r in sorted(verdict.reasons, key=lambda x: order.get(x.severity, 3)):
        color, tag = _SEV_STYLE.get(r.severity, (_DIM, r.severity.upper()))
        items.append(
            f'<li style="margin:.28rem 0;color:{_TXT};font-size:.88rem;line-height:1.35;">'
            f'<span style="display:inline-block;min-width:46px;font-size:.66rem;'
            f'font-weight:700;color:{color};border:1px solid {color};border-radius:4px;'
            f'padding:0 5px;text-align:center;margin-right:8px;vertical-align:1px;">{tag}</span>'
            f'{esc(r.text)}</li>')
    return (f'<div style="margin:.35rem 0;"><div style="color:{_MUT};font-size:.72rem;'
            f'font-weight:700;letter-spacing:.08em;">REASONS</div>'
            f'<ul style="list-style:none;padding-left:0;margin:.2rem 0;">{"".join(items)}</ul></div>')


def _chip(label: str, value: str, color: str = _TXT, border: str | None = None) -> str:
    b = border or "rgba(147,161,178,.25)"
    return (f'<div style="background:{SF_CARD};border:1px solid {b};border-radius:8px;'
            f'padding:6px 10px;">'
            f'<div style="font-size:.64rem;font-weight:700;letter-spacing:.08em;'
            f'color:{_MUT};">{esc(label)}</div>'
            f'<div style="font-size:.95rem;font-weight:700;color:{color};">{value}</div></div>')


def _chip_row(chips: list[str]) -> str:
    return (f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:.35rem 0;">'
            f'{"".join(chips)}</div>')


def _band_gauge_html(anchor: float, spot: float | None, forecast: dict,
                     straddle_em_pts: float | None, verdict: CockpitVerdict,
                     gex_levels: dict | None) -> str:
    """Horizontal strike-axis gauge: HAR point/PI bands vs the implied band,
    with the proposal strikes, GEX walls, anchor and spot as positioned divs."""
    point_half = anchor * float(forecast.get("point_pct") or 0) / 2
    upper_half = anchor * float(forecast.get("upper_pct") or 0) / 2
    marks = [anchor - upper_half, anchor + upper_half,
             anchor - point_half, anchor + point_half]
    if straddle_em_pts:
        marks += [anchor - straddle_em_pts, anchor + straddle_em_pts]
    p = verdict.proposal
    if p:
        marks += [p.put_long, p.call_long]
    for w in ((gex_levels or {}).get("call_wall"), (gex_levels or {}).get("put_wall")):
        if w:
            marks.append(w)
    if spot:
        marks.append(spot)
    lo, hi = min(marks), max(marks)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad

    def pct(x: float) -> float:
        return max(0.5, min(99.5, (x - lo) / (hi - lo) * 100.0))

    def band(a: float, b: float, color: str, top: int, h: int, extra: str = "") -> str:
        left, right = pct(min(a, b)), pct(max(a, b))
        return (f'<div style="position:absolute;left:{left:.2f}%;'
                f'width:{right - left:.2f}%;top:{top}px;height:{h}px;'
                f'background:{color};border-radius:3px;{extra}"></div>')

    def tick(x: float, color: str, w: int = 2, dashed: bool = False) -> str:
        style = f'border-left:{w}px {"dashed" if dashed else "solid"} {color};'
        return (f'<div style="position:absolute;left:{pct(x):.2f}%;top:0;bottom:0;'
                f'{style}"></div>')

    parts = [band(anchor - upper_half, anchor + upper_half,
                  "rgba(110,168,255,.14)", 8, 48),
             band(anchor - point_half, anchor + point_half,
                  "rgba(110,168,255,.30)", 20, 24)]
    if straddle_em_pts:
        parts.append(band(anchor - straddle_em_pts, anchor + straddle_em_pts,
                          "transparent", 14, 36,
                          extra=f"border:1px dashed {SF_WARN};"))
    cw, pw = (gex_levels or {}).get("call_wall"), (gex_levels or {}).get("put_wall")
    if cw:
        parts.append(tick(cw, SF_BEAR, dashed=True))
    if pw:
        parts.append(tick(pw, SF_BULL, dashed=True))
    if p:
        for k in (p.call_short, p.put_short):
            parts.append(tick(k, COLORS.get("yellow", "#f5c542"), w=3))
        for k in (p.call_long, p.put_long):
            parts.append(tick(k, _DIM, w=2))
    parts.append(tick(anchor, "#e7edf5", w=2))
    if spot:
        parts.append(tick(spot, _BLU, w=2, dashed=True))

    legend_bits = [
        f'<span style="color:{_BLU};">■ HAR point/PI band</span>',
        f'<span style="color:{SF_WARN};">▢ implied (straddle) band</span>',
        f'<span style="color:{COLORS.get("yellow", "#f5c542")};">| short strikes</span>',
        f'<span style="color:{_DIM};">| wings</span>',
        f'<span style="color:{SF_BEAR};">┆ call wall</span>',
        f'<span style="color:{SF_BULL};">┆ put wall</span>',
        f'<span style="color:#e7edf5;">| anchor</span>',
        f'<span style="color:{_BLU};">┆ spot</span>',
    ]
    return (
        f'<div style="background:{SF_CARD};border:1px solid rgba(147,161,178,.2);'
        f'border-radius:10px;padding:10px 14px;margin:.35rem 0;">'
        f'<div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;'
        f'color:{_MUT};margin-bottom:6px;">EXPECTED MOVE vs IMPLIED · strike axis '
        f'{lo:.0f} – {hi:.0f}</div>'
        f'<div style="position:relative;height:64px;background:rgba(11,15,22,.5);'
        f'border-radius:6px;overflow:hidden;">{"".join(parts)}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:.72rem;'
        f'color:{_MUT};margin-top:6px;">{"".join(legend_bits)}</div></div>'
    )


def _events_html(events: list) -> str:
    if not events:
        return _chip_row([_chip("MACRO EVENTS", "none this week", _MUT)])
    chips = []
    for e in events:
        t1 = e.tier == 1
        color = SF_BEAR if t1 else SF_WARN
        when = e.date + (f" · {e.release_time_et} ET" if e.release_time_et else "")
        chips.append(_chip(f"TIER {e.tier} · {e.name.upper()}", esc(when),
                           color, border=color))
    return _chip_row(chips)


def _proposal_card_html(verdict: CockpitVerdict) -> str:
    p = verdict.proposal
    if p is None:
        return (f'<div style="background:{SF_CARD};border:1px solid rgba(147,161,178,.2);'
                f'border-radius:10px;padding:12px 16px;margin:.35rem 0;color:{_MUT};">'
                f'No condor could be structured from the current inputs — see reasons.</div>')

    pop_txt = f"{p.pop * 100:.0f}% <span style='color:{_MUT};font-size:.7rem;'>({esc(p.pop_source)})</span>" \
        if p.pop is not None else f"—<span style='color:{_MUT};font-size:.7rem;'> (no delta)</span>"
    stats = _chip_row([
        _chip("NET CREDIT", f"{p.credit:.2f} pts (${p.max_profit:.0f})", SF_BULL),
        _chip("MAX LOSS", f"${p.max_loss:.0f}", SF_BEAR),
        _chip("CREDIT / WIDTH", f"{p.credit_ratio:.2f}",
              SF_BULL if p.credit_ratio >= 0.20 else SF_WARN),
        _chip("POP", pop_txt),
        _chip("BREAKEVENS", f"{p.breakeven_dn:.2f} / {p.breakeven_up:.2f}"),
        _chip("BE vs EM", f"{(verdict.proposal.breakeven_vs_em or {}).get('dn', '—')}× / "
                          f"{(verdict.proposal.breakeven_vs_em or {}).get('up', '—')}×", _MUT),
    ])

    rows = []
    for key, label in (("put_long", "BUY PUT"), ("put_short", "SELL PUT"),
                       ("call_short", "SELL CALL"), ("call_long", "BUY CALL")):
        leg = p.leg_quotes.get(key) or {}
        short = "short" in key
        ba = leg.get("ba_pct")
        rows.append(
            f'<tr style="color:{_TXT};">'
            f'<td style="padding:3px 10px;color:{SF_BEAR if short else SF_BULL};'
            f'font-weight:700;">{label}</td>'
            f'<td style="padding:3px 10px;font-weight:700;">{leg.get("strike", "—")}</td>'
            f'<td style="padding:3px 10px;">{leg.get("bid", 0):.2f} / {leg.get("ask", 0):.2f}</td>'
            f'<td style="padding:3px 10px;">{leg.get("mid", 0):.2f}</td>'
            f'<td style="padding:3px 10px;">{leg.get("oi", 0):.0f}</td>'
            f'<td style="padding:3px 10px;">{f"{ba:.1f}%" if ba is not None else "—"}</td></tr>')
    snap_bits = "".join(
        f'<div style="color:{_BLU};font-size:.76rem;margin-top:3px;">⤷ {esc(n)}</div>'
        for n in p.snap_notes)

    return (
        f'<div style="background:{SF_CARD};border:1px solid rgba(147,161,178,.2);'
        f'border-radius:10px;padding:12px 16px;margin:.35rem 0;">'
        f'<div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;color:{_MUT};">'
        f'PROPOSED CONDOR · {esc(p.expiration)} · wing {p.wing_width:g} · '
        f'k={p.k_used:g} · EM ±{p.em_pts:.2f} pts off anchor {p.anchor:.2f}</div>'
        f'{stats}'
        f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;'
        f'font-size:.84rem;margin-top:2px;">'
        f'<tr style="color:{_MUT};font-size:.68rem;letter-spacing:.06em;">'
        f'<th style="text-align:left;padding:3px 10px;">LEG</th>'
        f'<th style="text-align:left;padding:3px 10px;">STRIKE</th>'
        f'<th style="text-align:left;padding:3px 10px;">BID/ASK</th>'
        f'<th style="text-align:left;padding:3px 10px;">MID</th>'
        f'<th style="text-align:left;padding:3px 10px;">OI</th>'
        f'<th style="text-align:left;padding:3px 10px;">B/A %</th></tr>'
        f'{"".join(rows)}</table></div>{snap_bits}</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# The tab
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment
def _render_cockpit_tab(
    spot: float,
    levels: dict,
    regime: dict,
    data: GEXData,
    ticker: str = "SPX",
    weekly_em: dict | None = None,
) -> None:
    run_now = now_ny()
    week_start, monday = _planning_week(run_now)
    xsp_cfg = get_config(COCKPIT_TICKER)

    # ── In-tab controls → CockpitConfig (the pure engine reads only this) ──
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        k = st.slider("k × expected move (short-strike distance)", 0.75, 1.50,
                      1.00, 0.05, key="ckpt_k")
    with c2:
        wing_opts = ["auto"] + [str(w) for w in xsp_cfg.get("wing_widths", [5, 10])]
        wing_sel = st.selectbox("Wing width", wing_opts, key="ckpt_wing")
    with c3:
        policy = st.segmented_control("Tier-1 event policy", ["skip", "widen"],
                                      default="skip", key="ckpt_event_policy")
    cfg = CockpitConfig(
        k_expected_move=float(k),
        event_policy=policy or "skip",
        wing_width=None if wing_sel == "auto" else float(wing_sel),
    )

    # ── Assemble inputs (each degrades to a named SKIP, never a guess) ──
    bundle = _cached_cockpit_bundle(_tradier_token(), data.run_time,
                                    round(float(data.rfr or 0.045), 4))
    friday_exp = bundle.get("friday_exp")
    xsp_spot = bundle.get("spot")

    conn = _get_rf_conn()
    anchor, anchor_source = _resolve_anchor(conn, week_start, monday, run_now)
    straddle_em_pts, straddle_source = _resolve_straddle(ticker, weekly_em, bundle, run_now)

    forecast = None
    feature_row = None
    feat_note = None
    model_fitted_at = None
    try:
        payload = _cached_rf_load_model(COCKPIT_MODEL_SPEC, "SPX")
        model_fitted_at = payload.get("fitted_at")
        df_feat = _cached_rf_get_features(conn, "SPX")
        if df_feat is not None and len(df_feat):
            wk_ts = pd.Timestamp(week_start)
            if wk_ts in df_feat.index:
                feature_row = df_feat.loc[wk_ts]
            else:
                feature_row = df_feat.iloc[-1]
                feat_note = (f"features are from "
                             f"{getattr(df_feat.index[-1], 'date', lambda: df_feat.index[-1])()}"
                             f" — run Weekly Setup on the Spread Finder tab for this week's row")
        ref_for_levels = anchor or xsp_spot or 0.0
        if feature_row is not None and ref_for_levels > 0:
            forecast = forecast_next_week(payload["result"], feature_row,
                                          payload["feature_cols"], ref_for_levels)
            forecast = maybe_apply_conformal(forecast, conn, "SPX", COCKPIT_MODEL_SPEC)
    except FileNotFoundError:
        st.info(f"No saved **{COCKPIT_MODEL_SPEC}** fit yet — run **Weekly Setup** "
                f"on the Spread Finder tab to fit it.")
    except Exception as e:
        st.warning(f"Weekly HAR model unavailable: {e}")

    events = events_for_week(week_start)
    sessions = None
    t_years = None
    if friday_exp:
        try:
            sessions = trading_days_between(run_now.date(), friday_exp)
            t_years, _close = compute_time_to_expiry_years(friday_exp, ts=run_now)
        except Exception:
            pass

    verdict = evaluate_cockpit(
        ticker=COCKPIT_TICKER,
        anchor=anchor,
        spot=xsp_spot,
        friday_exp=friday_exp,
        forecast=forecast,
        straddle_em_pts=straddle_em_pts,
        gex_levels=bundle.get("gex_levels"),
        events=events,
        chain_quotes=bundle.get("quote_map") or {},
        cfg=cfg,
        strike_increment=float(xsp_cfg.get("strike_increment", 1)),
        wing_widths=[float(w) for w in xsp_cfg.get("wing_widths", [5, 10])],
        sessions_to_expiry=sessions,
        t_years=t_years,
    )

    try:
        save_cockpit_verdict(conn, verdict, ticker=COCKPIT_TICKER,
                             week_start=week_start, model_name=COCKPIT_MODEL_SPEC)
    except Exception as e:
        log.warning(f"cockpit verdict log failed: {e}")

    # ── Render ──
    if ticker != COCKPIT_TICKER:
        st.caption(f"The Cockpit is {COCKPIT_TICKER}-fixed — it fetched its own "
                   f"{COCKPIT_TICKER} data (dashboard is on {ticker}).")

    st.html(_verdict_banner_html(verdict, week_start))

    # VRP strip
    vrp = verdict.vrp
    ratio = vrp.get("ratio")
    ratio_color = SF_BULL if (ratio and ratio >= cfg.vrp_min_ratio) else SF_WARN
    vrp_chips = [
        _chip("HAR WEEK RANGE (POINT)",
              f"{(vrp.get('forecast_point_pct') or 0) * 100:.2f}%"
              if vrp.get("forecast_point_pct") else "—", _BLU),
        _chip("IMPLIED WEEK RANGE",
              f"{(vrp.get('implied_range_pct') or 0) * 100:.2f}%"
              if vrp.get("implied_range_pct") else "—", SF_WARN),
        _chip(f"VRP RATIO (gate ≥ {cfg.vrp_min_ratio:.2f})",
              f"{ratio:.2f}×" if ratio else "—", ratio_color),
        _chip("STRADDLE", f"{straddle_em_pts:.2f} pts" if straddle_em_pts else "—",
              _TXT),
        _chip("ANCHOR", f"{anchor:.2f}" if anchor else "—", "#e7edf5"),
        _chip("SPOT", f"{xsp_spot:.2f}" if xsp_spot else "—", _BLU),
    ]
    st.html(_chip_row(vrp_chips))
    if sessions is not None and monday < run_now.date():
        st.caption(f"{5 - min(sessions, 5)} of 5 sessions elapsed — the VRP "
                   f"comparison is Monday-calibrated; the live straddle prices "
                   f"only the remaining window.")

    if forecast and anchor:
        st.html(_band_gauge_html(anchor, xsp_spot, forecast, straddle_em_pts,
                                 verdict, bundle.get("gex_levels")))

    # GEX strip (+ the single coherent anchored-vs-live regime line)
    gl = bundle.get("gex_levels") or {}
    rg = bundle.get("regime") or {}
    gex_chips = [
        _chip("CALL WALL", f"{gl.get('call_wall'):.0f}" if gl.get("call_wall") else "—", SF_BEAR),
        _chip("ZERO GAMMA", f"{gl.get('zero_gamma'):.2f}" if gl.get("zero_gamma") else "—", _TXT),
        _chip("PUT WALL", f"{gl.get('put_wall'):.0f}" if gl.get("put_wall") else "—", SF_BULL),
        _chip("REGIME", esc(rg.get("regime", "—")),
              SF_BULL if "ositive" in str(rg.get("regime", "")) else
              (SF_BEAR if "egative" in str(rg.get("regime", "")) else SF_NEUT)),
    ]
    st.html(_chip_row(gex_chips))
    try:
        flag_raw = feature_row.get("gex_flag") if feature_row is not None else None
        flag = int(flag_raw) if flag_raw is not None and flag_raw == flag_raw else None
        for line in reconcile_gex_warnings([], [], flag, rg.get("regime")):
            st.caption(f"⚖️ {line}")
    except Exception:
        pass

    st.html(_events_html(events))
    st.html(_proposal_card_html(verdict))
    st.html(_reasons_html(verdict))

    # ── Handoff to Pre-Flight ──
    p = verdict.proposal
    if p is not None:
        if st.button("📤 Send to Pre-Flight", key="ckpt_send_preflight",
                     type="primary" if verdict.verdict == "TRADE" else "secondary"):
            st.session_state["preflight_trade"] = {
                "source": "cockpit",
                "ticker": COCKPIT_TICKER,
                "expiration": p.expiration,
                "quantity": 1,
                "legs": [
                    {"option_type": "call", "direction": "sell", "strike": p.call_short},
                    {"option_type": "call", "direction": "buy", "strike": p.call_long},
                    {"option_type": "put", "direction": "sell", "strike": p.put_short},
                    {"option_type": "put", "direction": "buy", "strike": p.put_long},
                ],
                "proposal": asdict(p),
                "week_start": week_start,
                "cockpit_verdict": verdict.verdict,
            }
            st.session_state["_tab_last"] = "preflight"
            st.session_state.pop("tab_seg", None)
            try:
                st.rerun(scope="app")
            except TypeError:  # older Streamlit without fragment scopes
                st.rerun()

    # ── Staleness / provenance footer ──
    bits = [f"model {COCKPIT_MODEL_SPEC}" + (f" fitted {model_fitted_at[:16]}" if model_fitted_at else ""),
            f"anchor: {anchor_source}",
            f"straddle: {straddle_source}",
            f"spot: {bundle.get('spot_source') or '—'}",
            f"GEX bundle keyed to {bundle.get('run_key', '—')}"]
    if feat_note:
        bits.append(f"⚠ {feat_note}")
    st.caption(" · ".join(bits))
    for err in bundle.get("errors") or []:
        st.caption(f"⚠ data: {err}")

    with st.expander("Verdict log (audit trail)", expanded=False):
        try:
            rows = load_recent_verdicts(conn, COCKPIT_TICKER, limit=12)
        except Exception as e:
            rows = []
            st.caption(f"log unavailable: {e}")
        if rows:
            df = pd.DataFrame(rows)[[
                "week_start", "verdict", "vrp_ratio", "call_short", "put_short",
                "credit", "credit_ratio", "pop", "logged_at", "updated_at",
            ]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No verdicts logged yet.")
