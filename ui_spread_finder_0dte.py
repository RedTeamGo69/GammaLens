"""
ui_spread_finder_0dte.py
Same-day (0DTE) spread finder tab.

Sibling to ``ui_spread_finder.py`` (weekly). Reuses the cadence-agnostic
spread translation (``build_spread_plan(dte=0)``), GEX adjustment, and a
few presentational helpers (strike table, GEX context panel) from the
weekly module — but loads its own daily-cadence HAR forecast and live
^VIX1D quote, and surfaces a Variance Risk Premium (VRP) banner.

Ticker scope: SPX and XSP only. Other tickers see an explicit error
banner and the tab body short-circuits before any data fetch or model
load. XSP reuses the SPX-trained daily HAR fit (XSP = SPX / 10).
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from theme import SF_BULL, SF_BEAR, SF_WARN
from models import GEXData

from range_finder.gex_bridge import (
    extract_gex_context, adjust_spread_with_gex,
)
from range_finder.spread_levels import (
    build_spread_plan as rf_build_spread_plan,
    TICKER_CONFIG as RF_TICKER_CONFIG,
    SpreadPlan,
)
# NB: ``log_daily_spread_plan`` is imported lazily inside the render function
# below, NOT here at module top. Reason: Streamlit Cloud's hot-reload does not
# re-execute already-loaded Python modules when files change on disk — it just
# clears st.cache_* and re-runs the script. So if a deploy lands AFTER
# ``range_finder.spread_persistence`` (or its re-export host ``spread_levels``)
# was loaded by the weekly tab, the new symbol ``log_daily_spread_plan`` won't
# be visible in the cached module object even though it exists in the file on
# disk. Importing it at module top would crash this whole tab with an
# ``ImportError`` on first click; importing lazily lets the rest of the tab
# render and degrades the logging step to a logged warning until the user
# reboots the app (Manage app → Reboot).
from range_finder.feature_builder_daily import get_daily_features
from range_finder.har_model_daily import (
    MODEL_SPECS_DAILY,
    forecast_next_session,
)

from ui_spread_finder import (
    _get_rf_conn,
    _cached_rf_load_model,
    _render_gex_context_panel,
    _render_sf_spread_table,
)

log = logging.getLogger(__name__)

# Tickers supported by the 0DTE finder. SPX and XSP share ^VIX1D and XSP =
# SPX / 10, so a single SPX-trained daily HAR drives both. Single-stock
# tickers (AMZN, AMD) don't have deep 0DTE markets and aren't supported;
# QQQ has 0DTE listings but the daily HAR isn't fit on QQQ data here.
SUPPORTED_0DTE_TICKERS = {"SPX", "XSP"}

# Wing widths suitable for 0DTE chains (which thin out beyond ±25 pts on SPX).
WING_WIDTHS_0DTE_SPX = [5, 10, 15, 20, 25]
WING_WIDTHS_0DTE_XSP = [1, 2, 3, 5]  # XSP strikes are 1-pt increments

# VRP banner thresholds — daily-range fractions (unitless).
# These will be re-tuned once we have historical vrp_daily data in the
# `daily_model_features` table to back-test against realised outcomes.
_VRP_RICH_THRESHOLD = 0.0015   # vrp_daily >= 15 bps: implied >> realised
_VRP_THIN_THRESHOLD = 0.0005   # vrp_daily < 5 bps: implied barely above realised

# How long to cache a live ^VIX1D quote (seconds). Theta on 0DTE is the
# whole point, so we want fresh data but not on every widget interaction.
_LIVE_VIX1D_TTL_SECONDS = 60


def _today_iso(ref_dt: datetime = None) -> str:
    return (ref_dt or datetime.now()).strftime("%Y-%m-%d")


@st.cache_data(ttl=300, show_spinner=False)
def _cached_daily_features(_conn, ticker: str = "SPX"):
    """Daily feature matrix for one ticker, cached for 5 minutes.

    The 0DTE finder re-renders on every widget interaction (live VIX1D,
    GEX overlay), and a fresh SELECT on daily_model_features each time
    would dominate the Neon CU-hour bill — same problem the weekly
    ``_cached_rf_get_features`` solves. 5-minute TTL is well under the
    daily cadence at which rows actually change.
    """
    return get_daily_features(_conn, ticker=ticker)


def _fetch_live_vix1d_cached() -> tuple[float | None, str]:
    """Return (vix1d_quote, label) using a session-state TTL cache.

    Avoids hammering yfinance on every fragment rerun while still giving
    the 0DTE tab a near-live VIX1D number. Returns the cached quote and
    a "captured X seconds ago" label for display.
    """
    cache_key = "_sf0_live_vix1d"
    ts_key    = "_sf0_live_vix1d_ts"

    now = time.monotonic()
    cached_val = st.session_state.get(cache_key)
    cached_ts  = st.session_state.get(ts_key)
    if cached_val is not None and cached_ts is not None \
       and (now - cached_ts) < _LIVE_VIX1D_TTL_SECONDS:
        age = int(now - cached_ts)
        return cached_val, f"cached {age}s ago"

    # Refresh via the data_collector helper (single yfinance quote).
    # Lazy import + AttributeError catch: defends against a stale
    # range_finder.data_collector module object in sys.modules (Streamlit
    # Cloud hot-reload doesn't fully invalidate Python's import cache when
    # a deploy adds new symbols to an existing module). If the cached
    # module lacks fetch_live_vix1d, fall through to a no-quote state and
    # let the user reboot the app.
    try:
        from range_finder.data_collector import fetch_live_vix1d
        val = fetch_live_vix1d()
    except (ImportError, AttributeError):
        val = None
    st.session_state[cache_key] = val
    st.session_state[ts_key]    = now
    return val, "live"


def _build_chain_quotes_for_0dte(data: GEXData, ticker: str) -> tuple[dict, str | None]:
    """Strike → {call/put bid/ask} for today's 0DTE expiration.

    Mirrors ``_build_chain_quotes_for_spreads`` from the weekly UI but
    resolves the expiration as ``data.dte0_exp`` (i.e. today, when a 0DTE
    is actually listed) rather than the upcoming Friday. Returns ``({},
    None)`` when no 0DTE is available — the caller surfaces a clear
    message and ``build_spread_plan`` falls back to BSM credit estimates.
    """
    if not data.chain_cache or not getattr(data, "dte0_exp", None):
        return {}, None

    today_iso = _today_iso()
    if data.dte0_exp != today_iso:
        # ``dte0_exp`` falls back to "nearest listed expiration" when no
        # actual 0DTE is listed (weekends, holidays, days when chains
        # haven't been rolled). Don't silently price next-day options as
        # "0DTE" — return empty and let the caller surface that fact.
        return {}, None

    entry = data.chain_cache.get((ticker, data.dte0_exp))
    if not entry or entry.get("status") != "ok":
        return {}, None

    quotes: dict = {}
    for opt in entry.get("calls", []):
        K = opt["strike"]
        quotes.setdefault(K, {})
        quotes[K]["call_bid"] = opt.get("bid", 0.0) or 0.0
        quotes[K]["call_ask"] = opt.get("ask", 0.0) or 0.0
    for opt in entry.get("puts", []):
        K = opt["strike"]
        quotes.setdefault(K, {})
        quotes[K]["put_bid"] = opt.get("bid", 0.0) or 0.0
        quotes[K]["put_ask"] = opt.get("ask", 0.0) or 0.0

    return quotes, data.dte0_exp


def _vrp_banner(vrp_daily: float | None) -> None:
    """Render a color-coded VRP banner. Warning-only — never blocks the plan."""
    if vrp_daily is None or (isinstance(vrp_daily, float) and math.isnan(vrp_daily)):
        st.info(
            "**VRP unavailable** — need both VIX1D-implied range and prior-day "
            "realised range. The plan below uses the model forecast without a "
            "VRP context check."
        )
        return

    vrp_bps = vrp_daily * 100  # convert fraction → percent
    if vrp_daily >= _VRP_RICH_THRESHOLD:
        st.markdown(
            f"<div style='background:{SF_BULL}22;border-left:4px solid {SF_BULL};"
            f"padding:10px 14px;border-radius:6px;'>"
            f"<b style='color:{SF_BULL}'>🟢 VRP elevated ({vrp_bps:+.2f}%)</b> — "
            f"VIX1D-implied range exceeds prior-day realised by a comfortable margin. "
            f"Favorable day to sell premium."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif vrp_daily >= _VRP_THIN_THRESHOLD:
        st.markdown(
            f"<div style='background:{SF_WARN}22;border-left:4px solid {SF_WARN};"
            f"padding:10px 14px;border-radius:6px;'>"
            f"<b style='color:{SF_WARN}'>🟡 VRP moderate ({vrp_bps:+.2f}%)</b> — "
            f"Implied above realised but not by much. Standard sizing; "
            f"watch for an IV expansion that widens the edge."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:{SF_BEAR}22;border-left:4px solid {SF_BEAR};"
            f"padding:10px 14px;border-radius:6px;'>"
            f"<b style='color:{SF_BEAR}'>🔴 VRP thin ({vrp_bps:+.2f}%)</b> — "
            f"Implied range is at or below prior-day realised. Premium-selling "
            f"edge is weak today; consider sizing down or skipping."
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_metric_row(forecast: dict, plan: SpreadPlan, spot: float,
                       live_vix1d: float | None, vrp_daily: float | None) -> None:
    """Top metric row — spot, VIX1D, point/upper PI, VRP, buffer."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot", f"${spot:,.2f}")
    c2.metric("VIX1D", f"{live_vix1d:.2f}" if live_vix1d is not None else "—")
    c3.metric("HAR Point Range",  f"{forecast['point_pct']*100:.2f}%")
    c4.metric(f"PI Upper ({forecast['confidence_level']}%)",
              f"{forecast['upper_pct']*100:.2f}%")
    vrp_str = f"{vrp_daily*100:+.2f}%" if vrp_daily is not None else "—"
    c5.metric("VRP (IV-RV)", vrp_str)


def _compute_inference_vrp(live_vix1d: float | None,
                            feature_row: pd.Series | None) -> float | None:
    """Re-compute VRP using live VIX1D for the *current* moment.

    The persisted ``vrp_daily`` in ``daily_model_features`` was computed
    against yesterday's VIX1D close. During the trading day we want the
    live ^VIX1D quote vs. yesterday's realised range (``har_d1_daily``).
    """
    if live_vix1d is None or live_vix1d <= 0:
        return None
    if feature_row is None:
        return None
    har_d1 = feature_row.get("har_d1_daily")
    if har_d1 is None or (isinstance(har_d1, float) and math.isnan(har_d1)):
        return None
    bm_factor = 2.0 * math.sqrt(2.0 / math.pi)
    vix1d_implied = (float(live_vix1d) / math.sqrt(252)) / 100 * bm_factor
    return vix1d_implied - float(har_d1)


@st.fragment
def _render_0dte_spread_finder_tab(
    spot: float,
    levels: dict,
    regime: dict,
    data: GEXData,
    ticker: str = "SPX",
):
    """Render the 0DTE Spread Finder tab.

    Wrapped in ``@st.fragment`` so widget interactions within the tab
    (model dropdown, recompute button, wing-width selector) only rerun
    the tab body — the heavy GEX chart in the sibling tab and the
    sidebar refresh logic stay untouched.
    """
    # ── Ticker guard ────────────────────────────────────────────────
    # First action: SPX and XSP only. Single-stock and QQQ tickers exit
    # before any data fetch or model load — the daily HAR isn't trained
    # on them, and 0DTE option chains aren't deep enough to support the
    # strategy. Users can still use the weekly tab for those.
    if ticker not in SUPPORTED_0DTE_TICKERS:
        st.error(
            f"⚠ 0DTE Spread Finder is currently SPX/XSP only — "
            f"**{ticker}** is not supported. "
            "Use the Weekly Spread Finder tab for this ticker."
        )
        st.caption(
            "0DTE markets for single-stock tickers are illiquid and the "
            "daily HAR model is trained on SPX only. XSP rides on the SPX "
            "fit (XSP = SPX / 10)."
        )
        return

    st.markdown("### ⚡ 0DTE Spread Finder")
    st.caption(
        "Same-day expiration credit spreads driven by a daily-cadence HAR forecast "
        "and live ^VIX1D. Strikes refresh on every interaction so intraday GEX "
        "shifts move the buffer with you."
    )

    today_iso = _today_iso()
    conn = _get_rf_conn()

    ticker_cfg = RF_TICKER_CONFIG.get(ticker, RF_TICKER_CONFIG["SPX"])

    # XSP reuses the SPX-trained daily HAR fit (XSP = SPX / 10, same ^VIX1D).
    # All inference loads come from ticker="SPX" rows in saved_models /
    # daily_model_features regardless of the active dashboard ticker.
    model_ticker = "SPX"

    # ── Top controls ────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        ref_key = f"sf0_ref_price_{ticker}"
        spx_ref_input = st.number_input(
            f"{ticker} reference (today's open or live spot)",
            min_value=10.0, max_value=20000.0,
            value=float(st.session_state.get(ref_key, spot)),
            step=float(ticker_cfg["strike_increment"]),
            help="Reference price for strike placement. Defaults to live spot; "
                 "override with today's open if you prefer a fixed anchor.",
            key=ref_key,
        )
    with col_b:
        live_vix1d, vix1d_label = _fetch_live_vix1d_cached()
        vix1d_key = f"sf0_vix1d_level_{ticker}"
        default_vix1d = float(live_vix1d) if live_vix1d is not None else 15.0
        vix1d_input = st.number_input(
            f"VIX1D ({vix1d_label})",
            min_value=1.0, max_value=200.0,
            value=float(st.session_state.get(vix1d_key, default_vix1d)),
            step=0.1,
            help="Live ^VIX1D quote; override if you want to stress-test a level. "
                 f"Cache TTL is {_LIVE_VIX1D_TTL_SECONDS}s.",
            key=vix1d_key,
        )
    with col_c:
        model_choice = st.selectbox(
            "Daily model",
            options=list(MODEL_SPECS_DAILY.keys()),
            index=1,  # default M2_daily_vix
            help="M1 = HAR core only; M2 adds VIX/VIX1D; M3 adds events + HV5.",
            key=f"sf0_model_choice_{ticker}",
        )

    if st.button("Recompute now", key=f"sf0_recompute_{ticker}",
                 help="Force a refresh of live VIX1D and re-derive the plan."):
        st.session_state.pop("_sf0_live_vix1d", None)
        st.session_state.pop("_sf0_live_vix1d_ts", None)
        _cached_daily_features.clear()
        st.rerun(scope="fragment")

    st.markdown("---")

    # ── Load daily features ────────────────────────────────────────
    try:
        df_dfeat = _cached_daily_features(conn, ticker=model_ticker)
    except Exception:
        df_dfeat = pd.DataFrame()

    if df_dfeat.empty:
        st.warning(
            "No daily feature data found. Run "
            "`python bootstrap_range_finder.py --daily-only` "
            "(or the full bootstrap) to populate `daily_spx` and "
            "`daily_model_features` first."
        )
        _render_gex_context_panel(extract_gex_context(levels, spot, regime), spot)
        return

    # Most recent persisted feature row — used as the lag-1 feature vector
    # for today's forecast.
    feature_row = df_dfeat.iloc[-1]

    # ── Load daily HAR model ───────────────────────────────────────
    try:
        payload = _cached_rf_load_model(model_choice, model_ticker)
    except Exception as e:
        st.error(
            f"Couldn't load daily model **{model_choice}** for "
            f"`ticker={model_ticker}`: {e}. "
            "Run `python bootstrap_range_finder.py --daily-only` to fit and "
            "save the daily HAR specs."
        )
        return

    result = payload["result"]
    feature_cols = payload["feature_cols"]

    # ── Forecast ────────────────────────────────────────────────────
    try:
        forecast = forecast_next_session(
            result, feature_row, feature_cols, spx_ref_input,
        )
    except Exception as e:
        st.error(f"Daily forecast failed: {e}")
        return

    # ── Live-VRP recompute using the user-visible VIX1D input ───────
    # Persisted vrp_daily was computed against yesterday's VIX1D close. The
    # banner displays the live-VIX1D version so the trader sees the gap
    # they actually face right now.
    vrp_daily_live = _compute_inference_vrp(vix1d_input, feature_row)

    # ── Build spread plan (dte=0) ──────────────────────────────────
    gex_ctx = extract_gex_context(levels, spot, regime)
    chain_quotes, chain_exp = _build_chain_quotes_for_0dte(data, ticker)

    wing_widths = WING_WIDTHS_0DTE_SPX if ticker == "SPX" else WING_WIDTHS_0DTE_XSP

    try:
        plan = rf_build_spread_plan(
            forecast=forecast,
            feature_row=feature_row,
            week_start=today_iso,        # spread_log keying — repurposed for 0DTE
            wing_widths=wing_widths,
            vix_level=vix1d_input,       # VIX1D drives BSM credit on 0DTE
            spx_open=spx_ref_input,
            dte=0,
            ticker=ticker,
            chain_quotes=chain_quotes,
        )
        # adjust_spread_with_gex returns an *annotation* dict (call/put-wall
        # distance, regime notes) — NOT a modified SpreadPlan. Keep `plan`
        # as the dataclass and capture the annotations separately so the
        # warnings list below can pick them up.
        gex_adj = adjust_spread_with_gex(plan, gex_ctx)
    except Exception as e:
        st.error(f"Build spread plan failed: {e}")
        return

    # Surface the GEX annotation notes as plan warnings so the existing
    # "Warnings & context" expander renders them.
    for note in (gex_adj.get("gex_adjustment_notes") or []):
        if note not in plan.warnings:
            plan.warnings.append(note)

    # ── VRP banner ──────────────────────────────────────────────────
    _vrp_banner(vrp_daily_live if vrp_daily_live is not None
                else feature_row.get("vrp_daily"))

    # ── Top metric row ──────────────────────────────────────────────
    _render_metric_row(forecast, plan, spot, vix1d_input, vrp_daily_live)

    # ── Chain provenance ────────────────────────────────────────────
    if chain_quotes:
        st.caption(
            f"📈 Pricing from live chain — expiration **{chain_exp}** "
            f"({len(chain_quotes)} strikes loaded)."
        )
    else:
        # With dte=0 the BSM fallback collapses to intrinsic value (T=0), so
        # OTM strikes price at $0.00. That's truthful but confusing; surface
        # the limitation explicitly so the trader knows the credits in the
        # table aren't meaningful until the chain refreshes.
        if getattr(data, "dte0_exp", None) and data.dte0_exp != today_iso:
            st.warning(
                f"⚠ No 0DTE listed today — nearest expiration in the cache is "
                f"**{data.dte0_exp}**. The strike map below is still valid, "
                f"but **credit estimates will show $0.00** for OTM strikes "
                f"because BSM at T=0 reduces to intrinsic value. Wait for the "
                f"chain to roll or check the weekly tab."
            )
        else:
            st.warning(
                "⚠ 0DTE chain not yet in cache — credit estimates use BSM "
                "with T=0 and **will show $0.00** for OTM strikes. Refresh "
                "the GEX dashboard for this ticker to pull live bid/ask."
            )

    # ── Spread tables ───────────────────────────────────────────────
    st.markdown("#### Call spreads")
    _render_sf_spread_table(plan.call_spreads, plan.recommended_width)

    st.markdown("#### Put spreads")
    _render_sf_spread_table(plan.put_spreads, plan.recommended_width)

    # ── Warnings / Context ──────────────────────────────────────────
    if plan.warnings:
        with st.expander("Warnings & context", expanded=True):
            for w in plan.warnings:
                st.markdown(f"- {w}")

    # ── GEX context panel ───────────────────────────────────────────
    st.markdown("---")
    cgc1, cgc2 = st.columns([1, 2])
    with cgc1:
        _render_gex_context_panel(gex_ctx, spot)
    with cgc2:
        st.markdown(
            "**Why GEX matters intraday for 0DTE**  \n"
            "Negative-gamma regimes amplify same-day moves — dealer hedging "
            "buys highs and sells lows. The buffer in this plan widens "
            "automatically when `gex_normalized` is negative. Re-click *Recompute "
            "now* after a big GEX shift to see strikes re-snap."
        )

    # ── Persist (once per session) ──────────────────────────────────
    # Log the day's plan to spread_log_daily so historical VRP-vs-outcome
    # analysis can join against actual high/low later. Guard against
    # duplicate logging in the same Streamlit session: each render would
    # otherwise upsert a fresh `generated_at` and clobber the morning's
    # first row.
    #
    # Lazy import (see module-top comment): defers the lookup of
    # log_daily_spread_plan to call time so a stale module cache from
    # Streamlit Cloud's hot-reload can't break the entire tab — it just
    # skips logging until the next reboot.
    log_key = f"_sf0_logged_{ticker}_{today_iso}"
    if not st.session_state.get(log_key):
        try:
            from range_finder.spread_persistence import log_daily_spread_plan
            log_daily_spread_plan(
                conn, plan, session_date=today_iso, ticker=ticker,
                vix1d_open=vix1d_input,
                vrp_at_open=vrp_daily_live,
            )
            st.session_state[log_key] = True
        except (ImportError, AttributeError) as e:
            log.warning(
                f"Daily spread logging unavailable — likely a stale "
                f"Streamlit Cloud module cache. Reboot the app via "
                f"Manage app → Reboot to enable logging. ({e})"
            )
        except Exception as e:
            log.warning(f"Failed to log daily spread plan: {e}")
