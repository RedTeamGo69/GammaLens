"""
ui_spread_finder_0dte.py
Same-day (0DTE) spread finder tab.

Sibling to ``ui_spread_finder.py`` (weekly). Reuses the cadence-agnostic
spread translation (``build_spread_plan(dte=0)``), GEX adjustment, and a
few presentational helpers (strike table, GEX context panel) from the
weekly module — but loads its own daily-cadence HAR forecast and live
^VIX1D quote, and surfaces a Variance Risk Premium (VRP) banner.

Ticker scope: SPX, XSP, and SPY. Other tickers see an explicit error
banner and the tab body short-circuits before any data fetch or model
load. XSP and SPY both reuse the SPX-trained daily HAR fit — all three
track the S&P 500 (XSP = SPX / 10, SPY ≈ SPX / 10) and share ^VIX1D, so
the scale-invariant range_pct forecast applies unchanged.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from models import GEXData

from range_finder.gex_bridge import (
    extract_gex_context, adjust_spread_with_gex,
)
from range_finder.spread_levels import (
    build_spread_plan as rf_build_spread_plan,
    TICKER_CONFIG as RF_TICKER_CONFIG,
    SpreadPlan,
)
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

# Tickers supported by the 0DTE finder. SPX, XSP, and SPY all track the
# S&P 500 and share ^VIX1D (XSP = SPX / 10, SPY ≈ SPX / 10), so a single
# SPX-trained daily HAR drives all three. The Nasdaq tickers (QQQ/NDX)
# are intentionally NOT supported here: there is no 1-day Nasdaq vol index
# (Cboe publishes no "VXN1D"), so the VIX1D-dependent VRP / 0DTE machinery
# has no clean analog. Single-stock tickers (AMZN, AMD) lack deep 0DTE
# markets. Those tickers can still use the Weekly Spread Finder tab.
SUPPORTED_0DTE_TICKERS = {"SPX", "XSP", "SPY"}

# Wing widths come from phase1.ticker_config (single source of truth shared
# with the weekly finder): SPX 100..500 by 100s, XSP 5/10/15/20.

# VRP banner thresholds — daily-range fractions (unitless).
# These will be re-tuned once we have historical vrp_daily data in the
# `daily_model_features` table to back-test against realised outcomes.
_VRP_RICH_THRESHOLD = 0.0015   # vrp_daily >= 15 bps: implied >> realised
_VRP_THIN_THRESHOLD = 0.0005   # vrp_daily < 5 bps: implied barely above realised

# How long to cache a live ^VIX1D quote (seconds). Theta on 0DTE is the
# whole point, so we want fresh data but not on every widget interaction.
_LIVE_VIX1D_TTL_SECONDS = 60


def _today_iso(ref_dt: datetime = None) -> str:
    """Trading-day 'today' in New York time. A naive datetime.now() on a
    UTC-hosted Streamlit instance rolls to tomorrow at 8 PM ET, which made
    _build_chain_quotes_for_0dte reject the perfectly valid same-day chain
    ('No 0DTE listed today — nearest expiration in the cache is <today>')."""
    from phase1.market_clock import now_ny
    return (ref_dt or now_ny()).strftime("%Y-%m-%d")


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
    # Honor the TTL even when the cached value is None — otherwise a
    # failed quote (weekend, yfinance hiccup) re-fires the ~1s blocking
    # fetch on every single widget interaction until it succeeds.
    if cached_ts is not None and (now - cached_ts) < _LIVE_VIX1D_TTL_SECONDS:
        age = int(now - cached_ts)
        label = f"cached {age}s ago" if cached_val is not None else "unavailable"
        return cached_val, label

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


def _mc(label: str, value: str, sub: str, value_color: str = "var(--text-primary)") -> str:
    """One terminal metric card (mirrors the weekly finder's ``.sf-cards`` grid)."""
    return (f'<div class="sf-card"><div class="cl">{label}</div>'
            f'<div class="cv" style="color:{value_color};">{value}</div>'
            f'<div class="cs">{sub}</div></div>')


def _vrp_banner(vrp_daily: float | None) -> None:
    """Render a color-coded VRP banner (terminal style). Warning-only — never blocks the plan."""
    if vrp_daily is None or (isinstance(vrp_daily, float) and math.isnan(vrp_daily)):
        st.html(
            '<div class="sf-note info" style="margin:2px 0 6px;">'
            '<b>VRP unavailable</b> — need both VIX1D-implied range and prior-day '
            'realised range. The plan below uses the model forecast without a '
            'VRP context check.</div>'
        )
        return

    vrp_bps = vrp_daily * 100  # convert fraction → percent
    if vrp_daily >= _VRP_RICH_THRESHOLD:
        color, dot, head, body = (
            "var(--green)", "🟢", f"VRP elevated ({vrp_bps:+.2f}%)",
            "VIX1D-implied range exceeds prior-day realised by a comfortable margin. "
            "Favorable day to sell premium.",
        )
    elif vrp_daily >= _VRP_THIN_THRESHOLD:
        color, dot, head, body = (
            "var(--amber)", "🟡", f"VRP moderate ({vrp_bps:+.2f}%)",
            "Implied above realised but not by much. Standard sizing; watch for an "
            "IV expansion that widens the edge.",
        )
    else:
        color, dot, head, body = (
            "var(--red)", "🔴", f"VRP thin ({vrp_bps:+.2f}%)",
            "Implied range is at or below prior-day realised. Premium-selling edge "
            "is weak today; consider sizing down or skipping.",
        )
    st.html(
        f'<div class="sf-vrp" style="background:color-mix(in srgb,{color} 12%,transparent);'
        f'border-left:4px solid {color};">'
        f'<b style="color:{color};">{dot} {head}</b> — {body}</div>'
    )


def _render_metric_row(forecast: dict, plan: SpreadPlan, spot: float,
                       live_vix1d: float | None, vrp_daily: float | None) -> None:
    """Top metric row — spot, VIX1D, point/upper PI, VRP — as a terminal card grid."""
    vrp_ok = vrp_daily is not None and not (isinstance(vrp_daily, float) and math.isnan(vrp_daily))
    vrp_str = f"{vrp_daily*100:+.2f}%" if vrp_ok else "—"
    vrp_color = (
        "var(--green)" if vrp_ok and vrp_daily >= _VRP_RICH_THRESHOLD else
        "var(--amber)" if vrp_ok and vrp_daily >= _VRP_THIN_THRESHOLD else
        "var(--red)" if vrp_ok else "var(--text-muted)"
    )
    st.html(
        '<div class="sf-cards">'
        + _mc("Spot", f"${spot:,.2f}", "live reference")
        + _mc("VIX1D", f"{live_vix1d:.2f}" if live_vix1d is not None else "—",
              "1-day implied vol")
        + _mc("HAR Point Range", f"{forecast['point_pct']*100:.2f}%",
              "model point estimate")
        + _mc(f"PI Upper ({forecast['confidence_level']}%)",
              f"{forecast['upper_pct']*100:.2f}%", "upper bound")
        + _mc("VRP (IV − RV)", vrp_str, "implied vs realised", vrp_color)
        + '</div>'
    )


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
    # First action: SPX, XSP, and SPY only. Every other ticker exits before
    # any data fetch or model load — the daily HAR is trained on the S&P 500
    # only, and Nasdaq/single-stock tickers either lack a 1-day vol index or
    # deep 0DTE markets. Users can still use the weekly tab for those.
    if ticker not in SUPPORTED_0DTE_TICKERS:
        st.error(
            f"⚠ 0DTE Spread Finder is currently SPX/XSP/SPY only — "
            f"**{ticker}** is not supported. "
            "Use the Weekly Spread Finder tab for this ticker."
        )
        st.caption(
            "The daily HAR model is trained on the S&P 500 only. XSP and SPY "
            "ride the SPX fit (both track the S&P 500 and share ^VIX1D; "
            "XSP = SPX / 10, SPY ≈ SPX / 10). The Nasdaq tickers have no "
            "1-day vol index, and single-stock 0DTE markets are illiquid."
        )
        return

    st.html(
        '<div class="sf-section-title">⚡ 0DTE Spread Finder</div>'
        '<div class="sf-section-sub">Same-day expiration credit spreads · daily-cadence '
        'HAR forecast + live ^VIX1D · strikes refresh on every interaction so intraday '
        'GEX shifts move the buffer with you · 💾 Neon Postgres</div>'
    )

    today_iso = _today_iso()
    conn = _get_rf_conn()

    ticker_cfg = RF_TICKER_CONFIG.get(ticker, RF_TICKER_CONFIG["SPX"])

    # SPX, XSP, and SPY all share the SPX-trained daily HAR fit: each tracks
    # the S&P 500 (XSP = SPX / 10, SPY ≈ SPX / 10) and uses ^VIX1D, and the
    # forecast is on scale-invariant range_pct anchored to the active ticker's
    # own reference price, so the SPX fit applies unchanged. All inference
    # loads come from ticker="SPX" rows in saved_models / daily_model_features
    # regardless of the active dashboard ticker.
    model_ticker = "SPX"

    # ── Top controls ────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        ref_key = f"sf0_ref_price_{ticker}"
        # Seed once, then let the widget own the key — passing `value=`
        # alongside an existing session-state key logs a Streamlit warning
        # on every rerun.
        if ref_key not in st.session_state:
            st.session_state[ref_key] = float(spot)
        # The key is sticky across reruns, so a value seeded from a broken
        # feed or a fat-fingered edit survives the session. 0DTE strikes
        # hug spot, so a ref >10% from live spot can only be corruption —
        # heal it before it anchors the ladder to the chain edge.
        if spot and spot > 0:
            try:
                _cur_ref0 = float(st.session_state[ref_key])
            except (TypeError, ValueError):
                _cur_ref0 = 0.0
            if _cur_ref0 <= 0 or abs(_cur_ref0 / spot - 1.0) > 0.10:
                st.session_state[ref_key] = float(spot)
        spx_ref_input = st.number_input(
            f"{ticker} reference (today's open or live spot)",
            min_value=10.0, max_value=20000.0,
            step=float(ticker_cfg["strike_increment"]),
            help="Reference price for strike placement. Defaults to live spot; "
                 "override with today's open if you prefer a fixed anchor.",
            key=ref_key,
        )
    with col_b:
        live_vix1d, vix1d_label = _fetch_live_vix1d_cached()
        vix1d_key = f"sf0_vix1d_level_{ticker}"
        if vix1d_key not in st.session_state:
            st.session_state[vix1d_key] = (
                float(live_vix1d) if live_vix1d is not None else 15.0
            )
        vix1d_input = st.number_input(
            f"VIX1D ({vix1d_label})",
            min_value=1.0, max_value=200.0,
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
    # A missing or schema-incompatible fit is recoverable in-app: the daily
    # features are already persisted (df_dfeat above), so run_daily_pipeline
    # just re-runs the OLS fit and saves the current-schema specs to Postgres.
    # This mirrors the weekly tab's "click Forecast to refit" and the cron's
    # self-heal, so a SCHEMA_VERSION bump no longer dead-ends the tab on a
    # manual shell command.
    try:
        payload = _cached_rf_load_model(model_choice, model_ticker)
    except Exception as e:
        from range_finder.model_persistence import IncompatibleModelError
        if isinstance(e, IncompatibleModelError):
            st.warning(
                f"Saved daily model **{model_choice}** is from an older feature "
                f"schema and can't be loaded safely — it needs a refit. ({e})"
            )
        else:
            st.info(
                f"No saved daily fit for **{model_choice}** yet "
                f"(`ticker={model_ticker}` — XSP/SPY reuse the SPX fit)."
            )
        st.caption(
            "Daily specs are normally refit by the Monday cron. You can refit "
            "now from the persisted daily features:"
        )
        if st.button(
            "Refit daily models now", key=f"sf0_refit_{ticker}", type="primary",
            help="Re-runs the daily HAR OLS over the saved feature rows and "
                 "writes the current-schema fits to Postgres.",
        ):
            with st.spinner("Refitting daily HAR specs…"):
                try:
                    from range_finder.har_model_daily import run_daily_pipeline
                    out = run_daily_pipeline(conn, preferred_model=model_choice)
                    _cached_rf_load_model.clear()
                    st.success(
                        "Refit complete — saved "
                        f"{sorted(out['results'].keys())}. Reloading…"
                    )
                    st.rerun(scope="fragment")
                except Exception as fit_err:
                    st.error(
                        f"Refit failed: {fit_err}. As a fallback, run "
                        "`python bootstrap_range_finder.py --daily-only`."
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

    wing_widths = ticker_cfg["wing_widths"]

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

    # ── VRP banner + metric row (same resolved value for both, so the
    # banner can't say "moderate" while the metric card shows "—") ────
    vrp_display = (vrp_daily_live if vrp_daily_live is not None
                   else feature_row.get("vrp_daily"))
    _vrp_banner(vrp_display)
    _render_metric_row(forecast, plan, spot, vix1d_input, vrp_display)

    # ── Chain provenance ────────────────────────────────────────────
    if chain_quotes:
        st.html(
            '<div class="sf-note ok" style="margin:6px 0;">'
            f'📈 Pricing from live chain — expiration <b>{chain_exp}</b> '
            f'({len(chain_quotes)} strikes loaded).</div>'
        )
    else:
        # With dte=0 the BSM fallback collapses to intrinsic value (T=0), so
        # OTM strikes price at $0.00. That's truthful but confusing; surface
        # the limitation explicitly so the trader knows the credits in the
        # table aren't meaningful until the chain refreshes.
        if getattr(data, "dte0_exp", None) and data.dte0_exp != today_iso:
            st.html(
                '<div class="sf-note warn" style="margin:6px 0;">'
                f'⚠ No 0DTE listed today — nearest expiration in the cache is '
                f'<b>{data.dte0_exp}</b>. The strike map below is still valid, '
                f'but <b>credit estimates will show $0.00</b> for OTM strikes '
                f'because BSM at T=0 reduces to intrinsic value. Wait for the '
                f'chain to roll or check the weekly tab.</div>'
            )
        else:
            st.html(
                '<div class="sf-note warn" style="margin:6px 0;">'
                '⚠ 0DTE chain not yet in cache — credit estimates use BSM '
                'with T=0 and <b>will show $0.00</b> for OTM strikes. Refresh '
                'the GEX dashboard for this ticker to pull live bid/ask.</div>'
            )

    # ── Spread tables ───────────────────────────────────────────────
    col_call, col_put = st.columns(2)
    with col_call:
        st.html('<div class="sf-eyebrow">Call Spreads</div>')
        _render_sf_spread_table(plan.call_spreads)
    with col_put:
        st.html('<div class="sf-eyebrow">Put Spreads</div>')
        _render_sf_spread_table(plan.put_spreads)

    # ── GEX context + warnings ──────────────────────────────────────
    st.markdown("---")
    cgc1, cgc2 = st.columns([1, 1])
    with cgc1:
        _render_gex_context_panel(gex_ctx, spot)
    with cgc2:
        st.html('<div class="sf-eyebrow">Warnings &amp; GEX Notes</div>')
        _esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if plan.warnings:
            st.html("".join(
                f'<div class="sf-note warn" style="margin-bottom:6px;">{_esc(w)}</div>'
                for w in plan.warnings
            ))
        else:
            st.html('<div class="sf-note ok">No warnings for this session.</div>')
        st.html(
            '<div class="sf-note info" style="margin-top:6px;">'
            '<b>Why GEX matters intraday for 0DTE</b> — negative-gamma regimes '
            'amplify same-day moves as dealer hedging buys highs and sells lows. '
            'The buffer in this plan widens automatically when <code>gex_normalized</code> '
            'is negative. Re-click <b>Recompute now</b> after a big GEX shift to see '
            'strikes re-snap.</div>'
        )

    # ── 0DTE plan logging removed ──
    # The day's plan used to be written to spread_log_daily here for a VRP-vs-
    # outcome backtest, but that log was write-only (nothing read it back), so
    # it was removed. The plan above is computed and displayed live on demand.
