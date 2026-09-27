"""
Spread Finder tab UI — weekly credit spread planning with forecast,
GEX context, and interactive strike maps.
Extracted from streamlit_app.py.
"""
from __future__ import annotations

from datetime import date as date_cls, datetime

import pandas as pd
import streamlit as st

from models import GEXData
from phase1 import credentials
from phase1.trading_week import planning_week

from range_finder.gex_bridge import (
    GEXContext, extract_gex_context, save_gex_to_range_finder,
    adjust_spread_with_gex, regime_to_gex_flag, reconcile_gex_warnings,
)
from range_finder.data_collector import (
    fetch_spx_vix as rf_fetch_spx_vix, save_spx_vix as rf_save_spx_vix,
    fetch_fred_macro as rf_fetch_fred_macro, save_fred_macro as rf_save_fred_macro,
    build_event_flags as rf_build_event_flags,
    get_weekly_spx as rf_get_weekly_spx,
    fred_key_status as rf_fred_key_status,
    live_vol_close as rf_live_vol_close,
)
from range_finder.feature_builder import (
    build_features as rf_build_features,
    get_features as rf_get_features,
    select_forecast_row as rf_select_forecast_row,
)
from range_finder.gex_policy import (
    GEX_LIVE_SPREAD_INFLUENCE_ENABLED,
    GEX_NORMALIZED_FEATURE,
    uses_disabled_gex_feature,
)
from range_finder.har_model import (
    MODEL_SPECS as RF_MODEL_SPECS,
    GEX_MIN_WEEKS_FOR_FIT as RF_GEX_MIN_WEEKS,
    feature_has_enough_data as rf_feature_has_enough_data,
    estimate_side_share_quantile as rf_estimate_side_share_quantile,
)
from range_finder.model_persistence import load_model as rf_load_model
from range_finder.spread_levels import (
    MIN_CREDIT_RATIO,
    SpreadPlan,
    SpreadTier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Spread Finder Tab — HAR placement with GEX retained as research context
# ─────────────────────────────────────────────────────────────────────────────

from range_finder.weekly_anchor import resolve_anchor as resolve_weekly_anchor
from range_finder.spread_finder_rules import check_reference, detect_regime_shift
from range_finder.forward_workbook import (
    FT_MAX_TICKERS, build_forward_test_workbook, ft_class,
)
from range_finder.recommendations import (
    build_recommendations, displayed_tier, tier_bands, chain_entry_to_quotes,
    FitReport, fit_and_save_specs, load_saved_fit,
)

from theme import SF_BULL, SF_BEAR, SF_NEUT


@st.cache_resource(ttl=3600)
def _get_rf_conn():
    """Get or create the range finder Postgres connection.

    1-hour TTL lets a long-idle Streamlit Cloud process drop its backend
    connection so Neon can auto-suspend the compute. Without a TTL the
    cached connection persists for the full process lifetime, holding
    the endpoint warm even when the user has no tab open. ``init_all_tables``
    is idempotent (CREATE TABLE IF NOT EXISTS + ALTER ... IF NOT EXISTS),
    so re-running it on each reconnect is safe and fast (<100 ms).
    """
    from range_finder.db import get_connection, init_all_tables
    conn = get_connection()
    init_all_tables(conn)
    return conn


@st.cache_data(ttl=600, show_spinner=False)
def _cached_pi_coverage(ticker: str) -> dict:
    """Empirical PI-coverage summary from scored spread_log rows.

    One small aggregate SELECT, cached 10 min — the underlying data only
    changes once a week (Monday cron logs the new plan and scores last
    week's), so reruns cost nothing. See range_finder/calibration.py.
    """
    from range_finder.calibration import weekly_pi_coverage
    return weekly_pi_coverage(_get_rf_conn(), ticker=ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_side_share_q(ticker: str) -> "float | None":
    """Empirical side-share quantile for per-side band placement, cached 1h
    (the weekly OHLC it derives from changes once a week). None → callers
    keep the legacy symmetric /2 split, so a DB hiccup degrades gracefully
    to pre-side-share behavior instead of erroring the tab."""
    try:
        from range_finder.feature_builder import _load_weekly_for_ticker
        est = rf_estimate_side_share_quantile(
            _load_weekly_for_ticker(_get_rf_conn(), ticker))
        return est["q"]
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _live_vol_close(proxy: str) -> float | None:
    """Latest vol-proxy (VIX/VXN) close, refreshed every 5 minutes.

    Returns None on an empty/failed fetch — the caller must treat None as
    "no live VIX" and NOT fabricate a number, because this value drives the
    VIX regime-shift circuit breaker: a stale or invented VIX silently
    disables the breaker exactly when a live spike should trip it. A 5-min
    TTL (vs the old cache-once-per-session behavior) means a mid-session
    spike is actually seen while still sparing the quote sources on every
    rerun. Source order (Tradier, then yfinance) lives in data_collector.
    """
    return rf_live_vol_close(proxy)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_rf_get_features(_conn, ticker: str = "SPX"):
    """model_features table for one ticker, cached for 10 minutes.

    Streamlit reruns the Spread Finder render every time any widget changes
    — dropdown, number input, slider, risk-tier button — and the raw
    ``rf_get_features`` helper runs a full ``SELECT *`` on ``model_features``
    each call. On Neon's free tier that single query dominated the
    CU-hour bill. The 10 min TTL is well under the weekly cadence at which
    features actually change; the Refresh/Rebuild/Weekly Setup paths call
    ``.clear()`` below to force a reload the moment new data lands.

    The ``_conn`` underscore tells Streamlit to skip hashing the connection
    object (psycopg2 connections aren't hashable).

    The window is pinned to har_model.TRAIN_WINDOW_YEARS (same as the cron
    fit path) so deeper weekly_spx backfills — e.g. the 10y history
    experiment — can't silently change what the UI trains or forecasts on.
    """
    from range_finder.har_model import train_window_min_date
    return rf_get_features(_conn, min_date=train_window_min_date(),
                           exclude_covid=True, ticker=ticker)


@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_rf_load_model(model_name: str, ticker: str):
    """Load a HAR fit from saved_models, cached for 1 hour across sessions.

    The session-state guards in the render path stop the unpickle from
    re-running within a single session, but every fresh session was
    re-fetching the multi-MB BYTEA blob from Neon — up to 8 SELECTs
    (4 specs × 2 tickers) per session lifetime. This wrapper caches the
    unpickled payload at the Streamlit-instance level so users sharing
    the same instance share the load.

    Uses ``@st.cache_resource`` (not ``cache_data``) because the unpickled
    statsmodels result wrapper is a non-trivial Python object that
    shouldn't be re-deserialized per call. The cache key is
    (model_name, ticker), so SPX and XSP fits stay independent — same as
    the underlying ``saved_models`` PK.

    Save paths in this module call ``_cached_rf_load_model.clear()``
    after a successful refit so the new fit is picked up immediately
    instead of waiting for the TTL to expire. The 1-hour TTL bridges
    the case where ``scheduled_snapshot.py`` (Mondays 9:30 AM ET) writes
    a new fit without any UI interaction.
    """
    from range_finder.db import get_connection
    # Open a short-lived connection rather than reusing the cached
    # Streamlit connection — _cached_rf_load_model is keyed on
    # (model_name, ticker), so passing in the cached `conn` would couple
    # cache invalidation to connection identity. The blob is pulled once
    # per (spec, ticker) per hour, so the extra connect cost is trivial.
    conn = get_connection()
    try:
        return rf_load_model(model_name, conn=conn, ticker=ticker)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@st.cache_data(ttl=900, show_spinner=False)
def _cached_weekly_setup(_conn, week_start: str, ticker: str):
    """Look up the weekly_setup row for (week_start, ticker), cached 15 min.

    Replaces a hand-rolled session-state miss-cache that still hit Neon
    on every fresh session. Cross-session caching means the SELECT only
    fires once per Streamlit instance per 15-minute window, regardless
    of how many users / tabs / reruns happen.

    Returns (monday_open, monday_vix) on hit, or None on miss. Both hits
    AND misses are cached — the bespoke logic this replaced cached only
    misses, so hits still re-queried on every fresh session.

    Save paths (``do_weekly`` / ``do_save_gex``) call ``.clear()`` to
    invalidate, mirroring the existing ``_cached_rf_get_features.clear()``
    pattern.
    """
    cur = _conn.cursor()
    cur.execute(
        "SELECT monday_open, monday_vix FROM weekly_setup WHERE week_start = ? AND ticker = ?",
        (week_start, ticker),
    )
    row = cur.fetchone()
    if row and row[0]:
        return (row[0], row[1])
    return None


def find_spread_finder_friday_exp(
    avail: "list[str]",
    ref_date: "date_cls | None" = None,
) -> "str | None":
    """Listed end-of-week expiration for the UI's planned trading week."""
    from phase1.market_clock import now_ny
    from phase1.trading_week import listed_week_expiration
    return listed_week_expiration(avail, planning_week(ref_date or now_ny()))


def spread_finder_weekly_em(avail, ref_date, *, weekly_exp, weekly_em_snap,
                            compute_em) -> dict:
    """Weekly EM for the week the Spread Finder is PLANNING.

    Mon–Thu (and weekends, where the app's weekly EM already rolls forward)
    the planned expiration is the app's weekly expiration, so the frozen
    weekly snapshot applies. On Friday the app's weekly EM still describes
    the expiring contract while the tab plans next week, so price the planned
    expiration's straddle live instead (its chain is pre-fetched). Empty
    rather than the wrong week's EM when that fails.
    """
    sf_exp = find_spread_finder_friday_exp(avail, ref_date)
    if sf_exp is None:
        return {}
    if sf_exp == weekly_exp:
        return weekly_em_snap or {}
    try:
        return compute_em(sf_exp) or {}
    except Exception:
        return {}


def _chain_entry_to_quotes(entry: dict) -> dict:
    return chain_entry_to_quotes(entry)


def _build_chain_quotes_for_spreads(
    data: GEXData,
    ticker: str,
    ref_date: "date_cls | None" = None,
) -> tuple[dict, str | None]:
    """Build a strike→{call_bid, call_ask, put_bid, put_ask} lookup from the
    Friday chain that matches the Spread Finder's planned week.

    The target expiration is anchored to *the week the spread finder is
    forecasting* (``phase1.trading_week.planning_week``), not to "whichever
    expiration the user happened to pick in the sidebar".  Before this was
    added, a user who had ``0DTE`` or ``Tomorrow`` selected would see the
    spread finder silently fall back to today's chain — producing $0.00
    credits for far-OTM weekly strikes because it was pricing 0-DTE puts
    instead of Friday weeklies.  The pre-fetch in ``fetch_all_data`` makes
    sure the right chain is always in ``data.chain_cache`` regardless of
    sidebar state, and this function just looks up that exact Friday.

    Returns (quotes_dict, selected_expiration_str_or_None).  When the
    correct Friday isn't available we return empty so ``build_spread_side``
    falls back cleanly to its BSM estimator (the UI caption tells the user
    we're on BSM rather than market quotes).
    """
    if not data.chain_cache:
        return {}, None

    # Resolve the expiration we SHOULD be looking at. Prefer the full
    # expiration universe from data.avail so holiday-shifted Fridays can
    # still match; fall back to whatever's already in the chain cache.
    avail = list(getattr(data, "avail", None) or [])
    if not avail:
        avail = sorted({exp for (t, exp) in data.chain_cache if t == ticker})

    target_exp = find_spread_finder_friday_exp(avail, ref_date=ref_date)
    if target_exp is None:
        return {}, None

    entry = data.chain_cache.get((ticker, target_exp))
    if not entry or entry.get("status") != "ok":
        # The right Friday isn't cached — don't silently substitute another
        # expiration (that's exactly how we used to end up pricing weekly
        # spreads off today's 0DTE chain).  Let the caller fall back to BSM.
        return {}, None

    return _chain_entry_to_quotes(entry), target_exp


def _export_chain_quotes(ticker: str, ref_date: "date_cls | None" = None) -> tuple[dict, str | None]:
    """Fetch the Spread-Finder-planned-Friday chain for ONE ticker and build the
    strike -> {bid/ask} lookup, so the multi-ticker Excel export snaps to the
    SAME real listed strikes the live tab does.

    Non-active export rows are built from persisted state and don't have the
    active ticker's ``data.chain_cache``, so without this they round to the
    nominal strike_increment — which drifts from the tradeable grid (e.g. AMD
    557.5 on its $2.5 far-OTM grid vs a nominal 556 that isn't even listed).

    Returns ``({}, None)`` on any failure (no token, no Friday listed, fetch
    error) so the caller degrades cleanly to nominal/BSM strikes — exactly
    today's behavior — instead of blocking the export.
    """
    from phase1 import credentials
    token = credentials.tradier_token()
    if not token:
        return {}, None
    try:
        from ui_market_data import get_expirations_cached
        avail = get_expirations_cached(token, ticker) or []
    except Exception:
        avail = []
    target_exp = find_spread_finder_friday_exp(avail, ref_date=ref_date)
    if not target_exp:
        return {}, None
    try:
        from phase1.data_client import TradierDataClient
        entry = TradierDataClient(token=token).get_chain_once(ticker, target_exp)
    except Exception:
        return {}, None
    if not entry or entry.get("status") != "ok":
        return {}, None
    quotes = _chain_entry_to_quotes(entry)
    return (quotes, target_exp) if quotes else ({}, None)


# Always-present defaults, in display order: the index/ETF defaults followed by
# the single-name defaults the Monday cron fits (NVDA/JPM/CAT). Mirrors
# QUICK_TICKERS in ui_theme — keep the two in sync. The active/searched ticker
# and any the user has added (session-state list under _FT_XLSX_EXTRA_KEY) are
# appended.
_FT_DEFAULT_TICKERS = ["SPX", "XSP", "SPY", "QQQ", "NDX", "NVDA", "JPM", "CAT"]
_FT_XLSX_EXTRA_KEY = "_sf_xlsx_extra"   # session-state list of user-added tickers


def _default_model_for_ticker(ticker: str) -> str:
    """Per-ticker default model spec.

    Walk-forward OOS validation showed M3_extended is the strongest forecaster
    on index products (SPX/XSP/QQQ/NDX/SPY...), while the simpler M2_vix wins on
    single names (AMZN/AMD/...), whose weekly range isn't well explained by the
    market-wide event-count / S&P return-lag features that M3 adds. So index/ETF
    products default to M3_extended and single stocks to M2_vix. The user can
    still override via the dropdown — the choice is remembered per ticker.
    """
    return "M2_vix" if ft_class(ticker) == "Stock" else "M3_extended"


import re as _re

# A valid exchange/OCC ticker root: 1-6 chars, A-Z 0-9 and a dot only. This
# is the primary defense against spreadsheet formula injection — the export
# tickers can arrive via URL state (ui_url_state), so a value like
# ``=cmd|'/c calc'!A1`` or ``@SUM(...)`` must never reach a cell. Anything
# outside this charset is rejected at the door.
_TICKER_RE = _re.compile(r"^[A-Z0-9.]{1,6}$")


def _valid_ticker_symbol(ticker: str) -> bool:
    return bool(_TICKER_RE.match((ticker or "").upper()))


def _xlsx_extra_list() -> list:
    """User-added export tickers (session-state; deduped, defaults excluded).

    Filtered through _valid_ticker_symbol on the way out so a tampered
    session/URL value can never reach the workbook builder even if it was
    persisted before validation tightened."""
    raw = st.session_state.setdefault(_FT_XLSX_EXTRA_KEY, [])
    return [t for t in raw if _valid_ticker_symbol(t)]


def _xlsx_add_extra(ticker: str) -> None:
    """Append a ticker to the export list (no-op for defaults / duplicates /
    anything that isn't a valid exchange symbol)."""
    t = (ticker or "").upper()
    if not t or t in _FT_DEFAULT_TICKERS or not _valid_ticker_symbol(t):
        return
    lst = st.session_state.setdefault(_FT_XLSX_EXTRA_KEY, [])
    if t not in lst:
        lst.append(t)


def _xlsx_remove_extra(ticker: str) -> None:
    """Drop a ticker from the export list."""
    lst = st.session_state.get(_FT_XLSX_EXTRA_KEY, [])
    if ticker in lst:
        lst.remove(ticker)


def _cb_remove_extra_pill() -> None:
    """Pills on_change: remove the added-export ticker whose chip was clicked,
    then reset the selection so each chip acts as a one-shot ✕ remove."""
    picked = st.session_state.get("_sf_xlsx_rm_pills")
    if picked:
        _xlsx_remove_extra(str(picked).split()[0])
    st.session_state["_sf_xlsx_rm_pills"] = None


def _tier_bands_from_tiers(spread_tiers) -> dict:
    return tier_bands(spread_tiers)


def _prior_week_close(conn, ticker: str, week_start: str):
    """Last weekly close strictly before week_start (XSP scaled to /10).

    Reads weekly_spx (SPX/XSP) or weekly_underlying (own-HAR tickers).
    The old exporter read a `spx_close` column off model_features — which
    doesn't exist in that table — so Prev Close was silently blank on
    every export.
    """
    from phase1.ticker_config import uses_own_har, price_scale_divisor
    try:
        if uses_own_har(ticker):
            from range_finder.data_collector import get_weekly_underlying
            wk = get_weekly_underlying(conn, ticker=ticker)
            col = "close"
        else:
            wk = rf_get_weekly_spx(conn)
            col = "spx_close"
        if wk.empty or col not in wk.columns:
            return None
        prior = wk.loc[wk.index < pd.Timestamp(week_start), col].dropna()
        if prior.empty:
            return None
        val = float(prior.iloc[-1])
        val /= price_scale_divisor(ticker)
        return round(val, 2)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _cached_prior_week_close(week_start: str, ticker: str):
    """Cross-session cache so the export build (which runs on every rerun
    because st.download_button materializes its payload eagerly) doesn't
    re-SELECT weekly history each interaction."""
    return _prior_week_close(_get_rf_conn(), ticker, week_start)


def _collect_week_bands_for_ticker(ticker: str, model_choice: str, week_start: str) -> dict:
    """Forecast + tier strikes for one NON-ACTIVE ticker from persisted
    state only: the saved HAR fit, the cron's Monday-open capture
    (weekly_setup), and the DB weekly-EM snapshot. Returns a plain dict
    so st.cache_data can pickle it; failures degrade to an error note on
    that ticker's row instead of sinking the whole export.
    """
    from phase1.ticker_config import feature_source_ticker
    from phase1.gex_history import get_em_snapshot

    out = {"ticker": ticker, "ref": None, "prev_close": None,
           "bands": {}, "notes": [], "error": None}
    try:
        conn = _get_rf_conn()

        # A scaled mini (XSP→SPX) reads its parent's shared features.
        _src = feature_source_ticker(ticker)
        df_feat = _cached_rf_get_features(conn, ticker=_src)
        if df_feat.empty:
            out["error"] = "no feature data — run Weekly Setup on this ticker"
            return out

        # Saved fit for the active spec (a scaled mini rides its parent's fit
        # only when it has none of its own).
        try:
            payload, _fit_src = load_saved_fit(_cached_rf_load_model, model_choice, ticker)
        except FileNotFoundError:
            out["error"] = f"no saved {model_choice} fit — run Weekly Setup on {ticker}"
            return out
        except Exception as e:
            out["error"] = f"saved {model_choice} fit unusable ({e}) — run Weekly Setup on {ticker}"
            return out
        if _fit_src != ticker:
            out["notes"].append(f"{ticker} via {_fit_src} fit")

        if uses_disabled_gex_feature(payload["feature_cols"]):
            out["error"] = (
                f"saved {model_choice} fit uses disabled GEX calibration — "
                f"run Weekly Setup on {ticker}"
            )
            return out

        feature_row, blocked = rf_select_forecast_row(df_feat, week_start)
        if blocked is not None:
            out["error"] = f"forecast blocked: {blocked.describe()}"
            return out

        # Reference price: Monday-open capture, else prior weekly close.
        ref = None
        vix = None
        setup = _cached_weekly_setup(conn, week_start, ticker)
        if setup:
            ref = float(setup[0]) if setup[0] else None
            vix = float(setup[1]) if setup[1] else None
        prev_close = _cached_prior_week_close(week_start, ticker)
        if ref is None:
            ref = prev_close
            if ref is not None:
                out["notes"].append("ref = prior close (no Mon-open capture)")
        if ref is None:
            out["error"] = "no reference price (weekly_setup empty)"
            return out
        if vix is None:
            try:
                _v = feature_row.get("vix_close")
                vix = float(_v) if _v is not None and _v == _v else 18.0
            except (TypeError, ValueError):
                vix = 18.0

        try:
            wem = get_em_snapshot(week_start, ticker=ticker, em_type="weekly")
        except Exception:
            wem = None

        # Live Friday chain so the export snaps to the SAME real listed strikes
        # the on-screen tab does — nominal increment rounding otherwise drifts
        # from the tradeable grid (e.g. AMD 556 nominal vs 557.5 on its $2.5
        # far-OTM grid). Degrades to nominal strikes on any failure.
        chain_quotes, chain_exp = _export_chain_quotes(
            ticker, pd.Timestamp(week_start).date()
        )

        _side_q = _cached_side_share_q(ticker)
        forecast, plan, tiers = build_recommendations(
            result=payload["result"], feature_row=feature_row,
            feature_cols=payload["feature_cols"], reference=ref, vix=vix,
            week_start=week_start, ticker=ticker, side_share_q=_side_q,
            chain_quotes=chain_quotes or None, weekly_em=wem,
            conn=conn, model_name=model_choice,
        )

        out["ref"] = round(float(ref), 2)
        out["prev_close"] = prev_close
        out["bands"] = _tier_bands_from_tiers(tiers)

        bits = [model_choice]
        _events = [n for n, f in (("FOMC", plan.has_fomc), ("CPI", plan.has_cpi),
                                  ("NFP", plan.has_nfp), ("OPEX", plan.has_opex)) if f]
        if _events:
            bits.append("/".join(_events))
        if plan.buffer_pct:
            bits.append(f"buf {plan.buffer_pct * 100:.2f}%")
        if plan.recommended_width:
            bits.append(f"wing {plan.recommended_width:g}")
        bits.append(f"chain {chain_exp}" if chain_exp else "nominal strikes")
        out["notes"] = bits + out["notes"]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


@st.cache_data(ttl=600, show_spinner=False)
def _cached_nonactive_week_bands(week_start: str, model_choice: str, tickers: tuple) -> list[dict]:
    return [_collect_week_bands_for_ticker(t, model_choice, week_start) for t in tickers]


def _auto_warm_up_spread_model(conn, ticker: str, ticker_cfg: dict) -> bool:
    """Cold-start the weekly Spread Finder for a freshly-looked-up ticker.

    Mirrors the Refresh → Rebuild → Forecast button chain but runs
    automatically the first time an own-HAR ticker has no feature matrix and
    no fitted model. Pulls ~6yr of weekly OHLC from yfinance, builds the
    feature matrix, then fits + saves every model spec to Postgres so later
    loads are instant.

    Returns True when a usable feature matrix now exists; False (caller shows a
    graceful "unavailable" note) when yfinance has no history for the symbol.
    """
    from phase1.ticker_config import uses_own_har as _uses_own_har
    from phase1.ticker_config import feature_source_ticker as _feature_source

    feat_ticker = _feature_source(ticker)

    # 1) Per-ticker weekly OHLC (own-HAR only; a scaled mini rides its parent's
    #    rows, which the parent's own warm-up/cron populates).
    if _uses_own_har(ticker):
        try:
            from range_finder.data_collector import (
                fetch_underlying_weekly, save_underlying_weekly,
            )
            df_t = fetch_underlying_weekly(
                ticker=ticker,
                yf_symbol=ticker_cfg["yf_symbol"],
                vol_proxy_yf=ticker_cfg.get("vol_proxy_yf", "^VIX"),
                years=6,
            )
            if df_t is None or len(df_t) == 0:
                return False
            save_underlying_weekly(conn, ticker, df_t)
        except Exception as e:
            st.caption(f"⚠ Could not load price history for {ticker}: {e}")
            return False

    # 2) Build the feature matrix.
    try:
        rf_build_features(conn, ticker=feat_ticker)
        _cached_rf_get_features.clear()
    except Exception as e:
        st.caption(f"⚠ Feature build failed for {ticker}: {e}")
        return False

    # 3) Fit + save every spec (same logic as the Weekly Setup path).
    try:
        df_feat = _cached_rf_get_features(conn, ticker=feat_ticker)
    except Exception:
        df_feat = pd.DataFrame()
    if df_feat is None or df_feat.empty:
        return False

    report = fit_and_save_specs(conn, ticker, features=df_feat)
    _cached_rf_load_model.clear()
    return bool(report.fits)


@st.fragment
def _render_spread_finder_tab(spot: float, levels: dict, regime: dict, data, ticker: str = "SPX", weekly_em: dict = None):
    """Render HAR spread placement with GEX kept as research context.

    Wrapped in @st.fragment so widget interactions inside the tab (horizon
    slider, model-spec dropdown, credit width, etc.) only rerun this tab
    instead of triggering a full-page rerun that rebuilds the GEX chart,
    the sidebar, and re-fetches weekly/monthly EM."""
    from phase1.market_clock import now_ny

    # resolve_config returns the chain-derived config registered by the main
    # app for an arbitrary ticker (or the curated entry for the known five) —
    # never SPX's 5-point grid as a silent fallback for an unknown symbol.
    from phase1.ticker_config import resolve_config as _resolve_config
    ticker_cfg = _resolve_config(ticker)

    # ── Earnings-week warning (single-stock tickers only) ──
    # AMZN/AMD set has_single_name_earnings=True in ticker_config; QQQ/SPX/XSP
    # don't. When the upcoming week is flagged in earnings_flags, surface a
    # warning banner at the top of the tab but still let the user generate
    # a plan — single-stock IV is structurally elevated pre-earnings, which
    # makes weekly OTM credit spreads look juicy but their tail risk is
    # much higher than the HAR predicts. ("Warn but allow" mode.)
    from phase1.ticker_config import has_single_name_earnings as _has_sne
    if _has_sne(ticker):
        try:
            from phase1.market_clock import now_ny as _now_ny_e
            # Check the SAME week the spread finder is planning for —
            # this week's Monday on Mon-Thu, next Monday on Fri-Sun
            # (phase1.trading_week.planning_week, same as week_start below).
            # The old check always looked at NEXT Monday, so a Tuesday
            # user planning this week's spreads never saw the warning
            # for earnings landing on Wednesday.
            #
            # Use the EXCHANGE clock (now_ny), not server-local: a UTC-hosted
            # Streamlit Cloud instance rolls to the next calendar day at
            # 20:00 ET, so a late-evening ET user would otherwise be shown
            # the wrong planning week (and miss/false-fire the warning) for
            # 4 hours every day. Everything else in the tab keys off ET.
            _plan_week = planning_week(_now_ny_e()).key
            _conn_e = _get_rf_conn()
            _cur_e = _conn_e.cursor()
            _cur_e.execute(
                "SELECT has_earnings, earnings_date FROM earnings_flags "
                "WHERE ticker = ? AND week_start = ?",
                (ticker, _plan_week),
            )
            _row_e = _cur_e.fetchone()
            if _row_e and _row_e[0]:
                _ed = _row_e[1] or _plan_week
                st.warning(
                    f"⚠ **{ticker} reports earnings the week of {_plan_week}** "
                    f"(scheduled {_ed}). Single-stock IV is structurally elevated "
                    "pre-earnings — weekly OTM credit spreads will look richer "
                    "than usual, but realised tail risk is much higher than the "
                    "HAR model predicts. Verify chain liquidity, and consider "
                    "sizing down or skipping this week."
                )
        except Exception:
            # Don't let an earnings-flag query failure block the tab render.
            pass

    # ── Fetch latest vol-proxy close (5-min TTL, refreshed across reruns) ──
    # Per-ticker vol proxy: VIX for SPX/XSP/AMZN/AMD, VXN for QQQ.
    #   live_vix_raw : the genuine live close, or None when unavailable. This
    #                  is the ONLY value the regime-shift breaker may read — a
    #                  fabricated fallback would silently disable it.
    #   live_vix     : display/default fallback for the VIX input, where a
    #                  concrete number is required so the widget renders.
    _vol_proxy = ticker_cfg.get("vol_proxy_yf", "^VIX")
    live_vix_raw = _live_vol_close(_vol_proxy)
    live_vix = live_vix_raw if live_vix_raw is not None else 18.0

    # ── Weekly anchor (range_finder.weekly_anchor) ──
    # Mon–Thu strikes lock to the planning week's first-session daily Open:
    # the persisted weekly_setup row wins; a missing row is captured (and
    # persisted) from the true daily bar once the first session has opened.
    # Fri–Sun the tab plans next week, whose open doesn't exist yet, so the
    # latest price stands in.
    run_now = now_ny()
    anchor = resolve_weekly_anchor(
        _get_rf_conn(), ticker, run_now,
        read_setup=_cached_weekly_setup, live_vix=live_vix, cfg=ticker_cfg,
    )
    if anchor.captured:
        _cached_weekly_setup.clear()
    self_heal_msg = None
    if anchor.self_healed:
        self_heal_msg = (
            f"ℹ️ No Monday-open capture existed for the week of "
            f"{anchor.week_start} (the Monday setup job didn't run for "
            f"{ticker}), so the anchor was captured retroactively "
            f"from Monday's daily bar — strikes are now locked to it "
            f"for the week. Run **Weekly Setup** if this recurs."
        )

    if anchor.locked:
        default_ref = anchor.open
        default_vix = anchor.vix or live_vix
        ref_source = {"db": "Mon open (from DB)", "captured": "Mon open"}[anchor.status]
        if anchor.self_healed:
            ref_source = "Mon open (self-healed)"
    else:
        default_ref = round(spot, 2)
        default_vix = live_vix
        if run_now.weekday() >= 5:
            ref_source = "Fri close (next-week plan)"
        elif run_now.weekday() == 4:
            ref_source = "live spot (next-week plan)"
        else:
            ref_source = "live spot"

    # ── Reference-price sanity guard ──
    # The ref is sticky (keyed session state, frozen Monday captures, a DB
    # restore, manual edits), so a single bad write anchors every strike
    # in the tab to a bogus level for the rest of the session — e.g. an
    # SPX ref of 100 against a ~7,400 spot walks the whole ladder to the
    # chain edge and produces nonsense spreads. Live spot is always in
    # hand here, so validate against it: hard-correct values that can only
    # be data corruption (no weekly market move is that big), and surface
    # a warning for large-but-conceivable gaps so the user double-checks.
    # Thresholds and the judgement live in range_finder.spread_finder_rules.
    ref_guard_msg = None
    if check_reference(default_ref, spot).status == "reset":
        default_ref = round(spot, 2)
        ref_source = "live spot (auto-corrected)"

    # ── Auto-update reference price and VIX when ticker changes ──
    # Also evict the ticker we're *leaving*'s cached HAR fit so the
    # incoming ticker lands on a clean load-from-Postgres path. Without
    # this, switching SPX→XSP→(GEX tab)→SPX could leave the outgoing
    # ticker's `_mdl_result_{ticker}` / `_mdl_name_{ticker}` session
    # entries in place; the name check at the bottom of this function
    # then sees cached_name == current dropdown choice and skips the
    # reload, freezing the tab on the last-displayed model.
    ref_key = f"sf_ref_price_{ticker}"
    vix_key = f"sf_vix_level_{ticker}"
    prev_ticker = st.session_state.get("_sf_prev_ticker")
    if prev_ticker != ticker:
        if prev_ticker:
            for _suffix in ("sf_model_result_", "sf_model_features_",
                            "sf_model_metrics_", "sf_model_name_"):
                st.session_state.pop(f"{_suffix}{prev_ticker}", None)
        st.session_state["_sf_prev_ticker"] = ticker

    # Seed the inputs whenever the anchor they came from changes — ticker
    # switch, first render, the week rolling, or the anchor locking (a
    # session opened before Monday's open must pick up the lock). A manual
    # override survives until then; an unlocked anchor's seed (open=None)
    # doesn't move with live spot, so the input doesn't chase ticks.
    seed_key = f"sf_ref_seed_{ticker}"
    _seed = (anchor.week_start, anchor.open)
    if (prev_ticker != ticker or ref_key not in st.session_state
            or st.session_state.get(seed_key) != _seed):
        st.session_state[ref_key] = default_ref
        st.session_state[vix_key] = default_vix
        st.session_state[seed_key] = _seed
    if vix_key not in st.session_state:
        st.session_state[vix_key] = default_vix

    # Validate whatever the session is actually carrying (stale seed from a
    # broken feed, fat-fingered edit, corrupt weekly_setup restore) — the
    # default-ref heal above can't see a value written on an earlier rerun.
    _cur_ref = st.session_state.get(ref_key)
    _ref_check = check_reference(_cur_ref, spot)
    _dev_cur = _ref_check.deviation
    if _ref_check.status == "reset":
        ref_guard_msg = (
            f"⚠️ The stored {ticker} reference (`{_cur_ref}`) was "
            f"{'unusable' if _dev_cur is None else f'{_dev_cur:.0%} away from live spot'} "
            f"(spot ≈ {spot:,.2f}) — strikes built from it would be nonsense, "
            f"so it was reset to **{default_ref:,.2f}** ({ref_source}). "
            "If this keeps happening, the Monday-open capture in "
            "`weekly_setup` for this week is bad — re-run **Weekly Setup** "
            "to overwrite it."
        )
        st.session_state[ref_key] = default_ref
    elif _ref_check.status == "warn":
        ref_guard_msg = (
            f"⚠️ The {ticker} reference ({float(_cur_ref):,.2f}) is "
            f"{_dev_cur:.0%} away from live spot ({spot:,.2f}). That's an "
            "extreme gap for a weekly anchor — verify the reference before "
            "trusting the strikes below."
        )

    st.html(
        f'<div class="sf-section-title">{ticker} Weekly Credit Spread Finder</div>'
        '<div class="sf-section-sub">HAR regression range forecast · GEX calibration '
        'pending — not applied to live buffer or strikes · 💾 Neon Postgres</div>'
    )
    if ref_guard_msg:
        st.warning(ref_guard_msg)
    if self_heal_msg:
        st.caption(self_heal_msg)

    # Hardcoded macro calendars (FOMC/CPI/NFP/…) lapse yearly; when they do,
    # event weeks silently flag event-free and the buffer stops widening.
    # The log-only warning is invisible on Streamlit Cloud, so surface it.
    try:
        from range_finder.event_calendars import calendar_staleness_warnings
        _cal_warnings = calendar_staleness_warnings()
        if _cal_warnings:
            st.warning("⚠ Event calendar maintenance needed:\n\n"
                       + "\n\n".join(f"• {m}" for m in _cal_warnings))
    except Exception:
        pass

    # ── Extract GEX context from current dashboard data ──
    gex_ctx = extract_gex_context(levels, spot, regime)

    # ── Sidebar-like controls within the tab ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        step_size = ticker_cfg["strike_increment"]
        # No `value=` — the key is always pre-seeded in session state above,
        # and passing both makes Streamlit log a "widget created with a
        # default value but also had its value set via Session State"
        # warning on every rerun.
        # Wide bounds on purpose: a tight min_value made the frontend clamp
        # any out-of-range value to a plausible-looking bound (e.g. 100 for
        # SPX) instead of letting the sanity guard above see — and heal —
        # the real garbage value.
        spx_close_input = st.number_input(
            f"{ticker} Reference ({ref_source})",
            min_value=1.0, max_value=100000.0, step=float(step_size),
            help=f"Reference price for range calculation. Source: {ref_source}. "
                 "Locked to this week's Monday open Mon-Thu (frozen live Monday, "
                 "restored from the weekly_setup table, or captured retroactively) "
                 "so strikes stay fixed all week; editable if you need to override.",
            key=ref_key,
        )

    with col_ctrl2:
        vix_source = "Mon open" if (anchor.locked and anchor.vix) else "last close"
        vix_input = st.number_input(
            f"VIX Level ({vix_source})",
            min_value=5.0, max_value=100.0, step=0.5,
            help=f"VIX level for BSM credit estimation. Source: {vix_source}. "
                 "Frozen at Monday's open alongside the reference price.",
            key=vix_key,
        )

    with col_ctrl3:
        _spec_keys = list(RF_MODEL_SPECS.keys())
        # Heal a stale per-ticker selection: a session that still has a removed
        # spec (e.g. M6_regime) stored under this key would otherwise crash
        # st.selectbox ("default value not in options"). Drop it so the
        # per-ticker default below applies cleanly.
        _mc_key = f"sf_model_choice_{ticker}"
        if _mc_key in st.session_state and st.session_state[_mc_key] not in _spec_keys:
            del st.session_state[_mc_key]
        _default_model = _default_model_for_ticker(ticker)
        _default_idx = _spec_keys.index(_default_model) if _default_model in _spec_keys else 0
        model_choice = st.selectbox(
            "Model Spec",
            options=_spec_keys,
            index=_default_idx,
            help=(
                "Default is set per ticker from walk-forward OOS: M3_extended for "
                "indices/ETFs and M2_vix for single names. GEX remains excluded "
                "from every live fit while BUG-06 calibration is pending. Your "
                "pick is remembered per ticker."
            ),
            key=f"sf_model_choice_{ticker}",
        )

    # Action buttons
    col_btn_main, col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.3, 1, 1, 1, 1])

    with col_btn_main:
        do_weekly = st.button("Weekly Setup", key=f"sf_weekly_{ticker}", type="primary", use_container_width=True,
                              help="Run all steps: Refresh → Rebuild → Save GEX for research → Forecast")
    with col_btn1:
        do_refresh = st.button("Refresh Data", key=f"sf_refresh_{ticker}", use_container_width=True)
    with col_btn2:
        do_rebuild = st.button("Rebuild Features", key=f"sf_rebuild_{ticker}", use_container_width=True)
    with col_btn3:
        do_save_gex = st.button("Save GEX", key=f"sf_save_gex_{ticker}", use_container_width=True)
    with col_btn4:
        do_forecast = st.button("Forecast", key=f"sf_forecast_{ticker}", use_container_width=True)

    # Weekly Setup runs all four steps in sequence
    if do_weekly:
        do_refresh = do_rebuild = do_save_gex = do_forecast = True
        # Weekly Setup may land a fresh weekly_setup row (via the cron
        # path that mirrors Monday's open). Drop the cached lookup so
        # the next render picks it up immediately instead of waiting
        # for the 15 min TTL to expire.
        _cached_weekly_setup.clear()

    conn = _get_rf_conn()
    from phase1.ticker_config import uses_own_har as _uses_own_har
    from phase1.ticker_config import feature_source_ticker as _feature_source

    # ── Step 1: Refresh market data ──
    if do_refresh:
        with st.spinner("1/4 — Fetching SPX / VIX weekly data..."):
            try:
                df_spx = rf_fetch_spx_vix(years=6)
                rows_written = rf_save_spx_vix(conn, df_spx)
                if len(df_spx) == 0:
                    st.success("SPX/VIX data already up to date")
                else:
                    st.success(f"SPX/VIX data refreshed — {len(df_spx)} weeks fetched, {rows_written} new")
            except Exception as e:
                if "empty" in str(e).lower() and datetime.today().weekday() >= 4:
                    st.warning(f"SPX/VIX fetch returned empty data (expected on weekends/holidays). Existing data is still valid.")
                else:
                    st.error(f"SPX/VIX fetch failed: {e}")

        # Own-HAR tickers (QQQ/AMZN/AMD) model their own weekly OHLC +
        # vol proxy, not SPX's. The cron always refreshed both; this
        # button used to refresh only SPX/VIX, so Weekly Setup on those
        # tickers fit against stale weekly_underlying rows.
        if _uses_own_har(ticker):
            with st.spinner(f"1/4 — Fetching {ticker} weekly data..."):
                try:
                    from range_finder.data_collector import (
                        fetch_underlying_weekly, save_underlying_weekly,
                    )
                    df_t = fetch_underlying_weekly(
                        ticker=ticker,
                        yf_symbol=ticker_cfg["yf_symbol"],
                        vol_proxy_yf=ticker_cfg["vol_proxy_yf"],
                        years=6,
                    )
                    _n_t = save_underlying_weekly(conn, ticker, df_t)
                    st.success(f"{ticker} weekly data refreshed — {len(df_t)} weeks fetched, {_n_t} upserted")
                except Exception as e:
                    st.error(f"{ticker} weekly fetch failed: {e}")

        with st.spinner("1/4 — Fetching FRED macro data..."):
            try:
                df_macro = rf_fetch_fred_macro(years=6)
                rf_save_fred_macro(conn, df_macro)
                st.success(f"FRED macro data refreshed — {len(df_macro)} rows")
            except Exception as e:
                # Distinguish "no key" from "FRED returned an error" — the
                # old message lumped them together and blamed the user for
                # a missing key whenever FRED itself had a 500. Also
                # surface the key status so you can eyeball whether
                # Streamlit actually picked up the secret.
                if not credentials.fred_api_key():
                    st.warning(
                        "FRED fetch skipped: FRED_API_KEY is not set. "
                        "Add it under Streamlit Cloud → Manage app → Secrets, "
                        "or export FRED_API_KEY in your local env."
                    )
                else:
                    st.warning(
                        f"FRED fetch failed — {e}. "
                        f"Key status: {rf_fred_key_status()}. "
                        "Existing macro data in the DB is still valid; "
                        "the rest of the pipeline will keep running. "
                        "Try again in a minute — FRED's API occasionally "
                        "returns 500s during their maintenance windows."
                    )

        rf_build_event_flags(conn)

    # ── Step 2: Rebuild features ──
    if do_rebuild:
        with st.spinner("2/4 — Computing feature matrix..."):
            try:
                # Same per-ticker routing as the Monday cron: a scaled mini
                # (XSP→SPX) shares its parent's feature rows; own-HAR
                # tickers build their own. (This used to always build SPX, so
                # Rebuild on QQQ/AMZN/AMD never touched the rows the fit reads.)
                rf_build_features(conn, ticker=_feature_source(ticker))
                # Drop the cached feature frame so the next read reflects
                # the rebuild — otherwise the UI would keep serving the
                # pre-rebuild rows until the 10-minute TTL expires.
                _cached_rf_get_features.clear()
                st.success("Features rebuilt")
            except Exception as e:
                st.error(f"Feature rebuild failed: {e}")

    # ── Step 3: Save live GEX ──
    if do_save_gex:
        try:
            gex_flag = save_gex_to_range_finder(gex_ctx, conn, ticker=ticker)
            regime_label = {1: "positive", 0: "neutral", -1: "negative"}.get(gex_flag, "unknown")
            # Invalidate the weekly_setup cache so a follow-up render sees
            # any concurrently-saved row right away (defensive — Save GEX
            # writes gex_inputs not weekly_setup, but the click is the
            # canonical "I'm actively setting up this week" signal).
            _cached_weekly_setup.clear()
            st.success(
                f"GEX research capture saved: regime={regime_label}, "
                f"flag={gex_flag} (ticker={ticker})"
            )
        except Exception as e:
            st.error(f"GEX save failed: {e}")

    st.markdown("---")

    # ── Check data availability (reload after rebuild if needed) ──
    # A scaled mini (XSP→SPX) rides its parent's HAR features at
    # 1/divisor scale (see ticker_config shares_har_with). Own-HAR tickers
    # (QQQ / SPY / NDX / AMZN / AMD) read their own per-ticker feature rows.
    from phase1.ticker_config import feature_source_ticker, get_config as _get_cfg
    _features_ticker = feature_source_ticker(ticker)
    try:
        df_feat = _cached_rf_get_features(conn, ticker=_features_ticker)
    except Exception:
        df_feat = pd.DataFrame()

    if df_feat.empty:
        # Own-HAR tickers cold-start automatically the first time they're
        # looked up: pull history → build
        # features → fit every spec. SPX (the base) keeps the manual nudge — an
        # empty SPX matrix means the whole app is uninitialized, a deploy-time
        # bootstrap step, not a per-ticker warm-up.
        _warm_attempted = st.session_state.setdefault("_sf_warmup_attempted", set())
        if _features_ticker != "SPX" and _features_ticker not in _warm_attempted:
            _warm_attempted.add(_features_ticker)
            with st.spinner(
                f"Preparing spread model for {ticker} (one-time, ~10s) — "
                "pulling history and fitting all specs…"
            ):
                _ok = _auto_warm_up_spread_model(
                    conn, _features_ticker, _get_cfg(_features_ticker)
                )
            if _ok:
                try:
                    df_feat = _cached_rf_get_features(conn, ticker=_features_ticker)
                except Exception:
                    df_feat = pd.DataFrame()

        if df_feat.empty:
            if _features_ticker != "SPX":
                st.warning(
                    f"📉 Spread Finder isn't available for **{ticker}** — no usable "
                    "multi-year price history was found for it (the GEX view above "
                    "still works). This typically means Yahoo Finance has no history "
                    "under the same symbol. Try **Refresh Market Data → Rebuild "
                    "Features → Forecast** to retry."
                )
            else:
                st.info(
                    "No feature data found. Click **Refresh Market Data** then **Rebuild Features** "
                    "to initialize the range prediction model (requires FRED API key in environment)."
                )
            # Still show GEX context even without model data
            _render_gex_context_panel(gex_ctx, spot)
            return

    # ── Fit model or load from cache ──
    # ── Step 4: Fit model and forecast ──
    # Session-state layout: we keep the fit result, features, metrics and
    # the *name* of the spec that produced them.  Tracking the name lets
    # the dropdown actually work — without it, switching M3→M4 would
    # silently keep using the old M3 fit (metric cards and strikes both)
    # because the rest of the code just reads `sf_model_result_{ticker}`
    # regardless of what the selectbox currently says.
    _mdl_result_key  = f"sf_model_result_{ticker}"
    _mdl_feat_key    = f"sf_model_features_{ticker}"
    _mdl_metrics_key = f"sf_model_metrics_{ticker}"
    _mdl_name_key    = f"sf_model_name_{ticker}"

    if do_forecast:
        # Weekly Setup fits every spec (same as the Monday cron) so a
        # user can switch the model dropdown afterwards without tripping
        # the "click Forecast to fit this spec" prompt. A standalone
        # Forecast click still fits only the currently-selected spec —
        # that's the fast path for iterating on one model.
        _specs_to_fit = list(RF_MODEL_SPECS.keys()) if do_weekly else [model_choice]

        # A shared pair (SPX+XSP) reuses one scale-invariant fit, so a
        # Weekly Setup click on either member populates the whole group — every
        # curated ticker that resolves to the same feature source. Own-HAR
        # tickers (QQQ / SPY / AMZN / AMD) stand alone. A plain Forecast click
        # always saves only to the active ticker.
        from phase1.ticker_config import feature_source_ticker as _fsrc, all_tickers as _all_t
        if do_weekly:
            _src = _fsrc(ticker)
            _tickers_to_save = [t for t in _all_t() if _fsrc(t) == _src] or [ticker]
        else:
            _tickers_to_save = [ticker]

        _spinner_label = (
            f"4/4 — Fitting {len(_specs_to_fit)} specs..."
            if len(_specs_to_fit) > 1 else f"4/4 — Fitting {model_choice}..."
        )

        # GEX calibration status for the spec the user is looking at. Live
        # fits never consume GEX while the policy switch is off (BUG-06);
        # production_feature_columns applies the same switch inside the fit.
        if model_choice == "M4_full":
            gex_col = GEX_NORMALIZED_FEATURE
            _weeks = (int(df_feat[gex_col].notna().sum())
                      if gex_col in df_feat.columns else 0)
            if not GEX_LIVE_SPREAD_INFLUENCE_ENABLED:
                st.caption(
                    f"GEX calibration: Pending — {_weeks} stored weeks "
                    "remain available for research, but GEX is not "
                    "applied to live fits, buffers, or strikes."
                )
            elif rf_feature_has_enough_data(df_feat, gex_col):
                st.caption(
                    f"ℹ️ M4_full: using {_weeks} weeks of stored GEX "
                    "history as a training feature."
                )
            else:
                st.caption(
                    f"ℹ️ M4_full: only {_weeks} weeks of GEX history — need >{RF_GEX_MIN_WEEKS} to "
                    f"fold `gex_normalized` into the fit. Keep clicking **Save GEX** "
                    f"each week; in the meantime M4 runs without the GEX feature."
                )

        with st.spinner(_spinner_label):
            try:
                _report = fit_and_save_specs(
                    conn, ticker, specs=_specs_to_fit,
                    save_tickers=_tickers_to_save, features=df_feat,
                )
            except Exception as e:     # a save failed mid-way (DB)
                st.error(f"Saving fits failed: {e}")
                _report = FitReport()
        # New fits landed in Postgres — drop cached unpickles so a later tab
        # switch sees the fresh weights instead of the previous fit until the
        # 1-hour TTL expires.
        _cached_rf_load_model.clear()
        for _spec, _reason in _report.skipped.items():
            if _reason.startswith("fit failed"):
                st.error(f"{_spec} fitting failed: {_reason}")
            elif _spec == model_choice:
                st.warning(f"{_spec}: {_reason} — skipped")
        _selected = _report.fits.get(model_choice)
        _selected_result = _selected.result if _selected else None
        _selected_avail = _selected.feature_cols if _selected else None
        _selected_metrics = _selected.metrics if _selected else None

        # Summary line for the Weekly Setup path so the user can see which
        # specs landed in Postgres at a glance.
        if do_weekly:
            _ticker_note = " × ".join(_tickers_to_save)
            st.success(
                f"Fitted {len(_report.fits)}/{len(_specs_to_fit)} specs for "
                f"{_ticker_note} ({len(_report.fits) * len(_tickers_to_save)} rows saved)"
            )

        # Prime session state with the currently-selected spec's fit so
        # the rest of this render uses it without falling through to the
        # load-from-Postgres path (same behavior as the previous single-
        # spec code).
        if _selected_result is not None:
            st.session_state[_mdl_result_key]  = _selected_result
            st.session_state[_mdl_feat_key]    = _selected_avail
            st.session_state[_mdl_metrics_key] = _selected_metrics
            st.session_state[_mdl_name_key]    = model_choice
            st.success(
                f"Model fitted | {model_choice} | "
                f"OOS R² = {_selected_metrics['oos_r2']:.4f}"
            )

    # If the user toggled the model dropdown, the cached fit in session
    # state belongs to a different spec — evict it so the load block
    # below pulls the right saved fit for the newly-selected spec from
    # Postgres (or shows the "click Forecast" nudge if that spec has
    # never been fitted yet).
    _cached_mdl_name = st.session_state.get(_mdl_name_key)
    if _cached_mdl_name is not None and _cached_mdl_name != model_choice:
        for _k in (_mdl_result_key, _mdl_feat_key, _mdl_metrics_key, _mdl_name_key):
            st.session_state.pop(_k, None)

    # Try to load model from session or disk
    if _mdl_result_key not in st.session_state:
        try:
            payload, _fit_src = load_saved_fit(_cached_rf_load_model, model_choice, ticker)
            st.session_state[_mdl_result_key]  = payload["result"]
            st.session_state[_mdl_feat_key]    = payload["feature_cols"]
            st.session_state[_mdl_metrics_key] = payload["metrics"]
            st.session_state[_mdl_name_key]    = model_choice
        except FileNotFoundError:
            st.info(f"No saved fit for **{model_choice}** yet. Click **Forecast** to fit it for the first time.")
            _render_gex_context_panel(gex_ctx, spot)
            return
        except Exception as e:
            st.warning(f"Saved {model_choice} model incompatible: {e}. Click **Forecast** to refit.")
            _render_gex_context_panel(gex_ctx, spot)
            return

    result       = st.session_state[_mdl_result_key]
    feat_cols    = st.session_state[_mdl_feat_key]
    metrics      = st.session_state[_mdl_metrics_key]
    active_model = st.session_state.get(_mdl_name_key, model_choice)

    if uses_disabled_gex_feature(feat_cols):
        for _k in (_mdl_result_key, _mdl_feat_key, _mdl_metrics_key, _mdl_name_key):
            st.session_state.pop(_k, None)
        st.warning(
            f"Saved {model_choice} fit predates the BUG-06 mitigation and "
            "contains GEX. Click Forecast to refit it without GEX; no live "
            "recommendation will be shown from the old fit."
        )
        _render_gex_context_panel(gex_ctx, spot)
        return

    # ── Determine week start ──
    # Anchored to NY wall clock (run_now = now_ny() above) so that the
    # week_start convention here matches the Friday chain pre-fetch in
    # streamlit_app.fetch_all_data — otherwise a UTC-hosted server could
    # roll into "tomorrow" a few hours before NY does and end up looking
    # at a different expiration than the one the pre-fetch cached.
    #
    # Mon-Thu: plan THIS week's spreads; Fri-Sun: roll to next week
    # (phase1.trading_week.planning_week — the export, chain pre-fetch and
    # earnings check ask the same question).
    week_start = planning_week(run_now).key
    sf_ref_date = run_now.date()

    # ── Get feature row ──
    # Look up the row in the already-cached `df_feat` instead of issuing a
    # fresh `SELECT * FROM model_features WHERE week_start = ?` on every
    # render. `df_feat` is loaded by `_cached_rf_get_features` (10 min TTL)
    # and indexed by `week_start` (DatetimeIndex), so this is a pure
    # in-memory lookup and saves one Neon roundtrip per Spread Finder pass.
    feature_row, _blocked = rf_select_forecast_row(df_feat, week_start)
    if _blocked is not None and _blocked.reason == "missing":
        _newest = (_blocked.newest_week.strftime("%Y-%m-%d")
                   if _blocked.newest_week is not None else "none")
        st.error(
            f"⚠️ **Forecast blocked:** no feature row exists for {week_start}. "
            f"The newest persisted row is {_newest}. Run **Weekly "
            "Setup** (or **Refresh Data** + **Rebuild Features**) before "
            "using this week's strikes."
        )
        _render_gex_context_panel(gex_ctx, spot)
        return
    if _blocked is not None:
        _source_label = (
            _blocked.source_week.strftime("%Y-%m-%d")
            if _blocked.source_week is not None else "missing"
        )
        st.error(
            f"⚠️ **Forecast blocked:** the {week_start} row uses weekly path "
            f"inputs from **{_source_label}**, but it requires "
            f"**{_blocked.required_week:%Y-%m-%d}**. A provisional next-week row "
            "created before Friday's close cannot be reused after the week "
            "rolls. Run **Weekly Setup** (or **Refresh Data** + **Rebuild "
            "Features**) to refresh the HAR inputs. The saved Monday-open "
            "strike anchor will remain unchanged."
        )
        _render_gex_context_panel(gex_ctx, spot)
        return

    # ── Regime-shift circuit breaker ──────────────────────────────────────
    # HAR features are lagged by one week — vix_close in feature_row is
    # last Friday's close. When IV spikes overnight (e.g., VIX 15 → 40 on
    # a news shock), the model's input features still reflect the pre-spike
    # world for a full week, so the forecast's PI is anchored to the wrong
    # vol regime and the Spread Finder will place strikes dangerously
    # close to spot. The live weekly expected move (from the straddle) is
    # drawn on the strike map as a reference band and shorts inside it are
    # flagged, but that doesn't stop the user from trusting the "model says
    # range will be 2%" read.
    #
    # Compare the live VIX to the trailing VIX already in the feature row
    # (range_finder.spread_finder_rules). live_vix_raw — the genuine fetch,
    # NOT the 18.0 display default — or the breaker would judge an invented
    # number.
    regime_shift = detect_regime_shift(live_vix_raw, feature_row.get("vix_close"))
    if regime_shift is not None:
        st.error(
            f"⚠️ **VIX regime shift detected ({regime_shift.severity})** — "
            f"live VIX **{regime_shift.live_vix:.1f}** vs trailing "
            f"feature VIX **{regime_shift.trailing_vix:.1f}** "
            f"(**{regime_shift.ratio:.2f}×**).\n\n"
            f"The HAR model's features lag by one week, so the forecast below "
            f"is anchored to the pre-spike vol regime. Short strikes sized "
            f"against this forecast are likely **too narrow**. Check the live "
            f"weekly expected-move band on the strike map (it reflects the "
            f"current straddle) and treat any short strike inside it as too "
            f"risky — or, better, skip the trade until features catch up."
        )
    # ── Build forecast → plan → tiers from the latest GEX refresh ──
    # We intentionally DO NOT cache these on a session-state key any more.
    # Everything below is cheap arithmetic on top of the already-loaded HAR
    # model (the expensive validation + production fits are gated behind the
    # "Forecast" button and cached separately via rf_load_model), so
    # recomputing on every page rerun lets the spread finder pick up fresh
    # chain bid/ask as soon as fetch_all_data refreshes data.chain_cache
    # (i.e. on auto-refresh, "Refresh Now", or any normal rerun — no need
    # to click "Forecast" again to get updated credits).
    #
    # Risk-tier switching stays snappy because the _risk_tier_fragment
    # below is wrapped in @st.fragment and only re-reads the spread_tiers
    # we stash in session_state — the outer recompute doesn't happen on
    # tier toggles.
    # Per-side band share: empirical side-share quantile (falls back to the
    # legacy /2 split when weekly history is unavailable). Cached 1h.
    side_share_q = _cached_side_share_q(ticker)

    chain_quotes, chain_exp = _build_chain_quotes_for_spreads(
        data, ticker, ref_date=sf_ref_date,
    )
    forecast, plan, spread_tiers = build_recommendations(
        result=result, feature_row=feature_row, feature_cols=feat_cols,
        reference=spx_close_input, vix=vix_input, week_start=week_start,
        ticker=ticker, side_share_q=side_share_q, chain_quotes=chain_quotes,
        weekly_em=weekly_em, conn=conn, model_name=model_choice,
    )

    gex_adj = adjust_spread_with_gex(plan, gex_ctx)

    # =========================================================================
    # METRIC CARDS
    # =========================================================================

    # Five metric cards. The PI card holds both tier bounds (Lower ↔ Upper)
    # so all four risk tiers named in the spec — Lower PI, Point, Upper PI,
    # Effective — stay visible without cramping the row into 6 columns
    # (which clips labels on typical laptop widths).
    # Flag it when the dropdown's selection and the fit we're actually
    # rendering don't agree (the OOS card should tell the truth, not lie).
    _mdl_label = active_model if active_model == model_choice else f"{active_model} ⚠"
    _flag = gex_adj['gex_regime_flag']
    _reg_color = ("var(--green)" if _flag > 0 else
                  "var(--red)" if _flag < 0 else "var(--text-muted)")

    def _mc(label, value, sub, value_color="var(--text-primary)"):
        return (f'<div class="sf-card"><div class="cl">{label}</div>'
                f'<div class="cv" style="color:{value_color};">{value}</div>'
                f'<div class="cs">{sub}</div></div>')

    st.html(
        '<div class="sf-cards">'
        + _mc("Point Estimate", f"{forecast['point_pct']*100:.2f}%",
              f"vs VIX {forecast['model_vs_vix']*100:+.2f}%")
        + _mc(f"{forecast['confidence_level']}% PI Range",
              f"{forecast['lower_pct']*100:.2f}–{forecast['upper_pct']*100:.2f}%",
              "aggressive ↔ moderate")
        + _mc("Effective Range", f"{plan.effective_range_pct*100:.2f}%",
              f"conservative · buffer +{plan.buffer_pct*100:.2f}%")
        + _mc("GEX Regime", gex_ctx.gamma_regime.title(),
              "research · not applied", _reg_color)
        + _mc(f"OOS R² · {_mdl_label}", f"{metrics['oos_r2']:.4f}",
              f"MAE {metrics['mae_pct']*100:.2f}%"
              + (f" · dir {metrics['direction_acc']:.0%}"
                 if metrics.get("direction_acc") is not None else ""))
        + '</div>'
    )

    # ── Empirical PI-coverage caption (the calibration audit, in one line) ──
    # Reads scored spread_log rows via calibration.weekly_pi_coverage; the
    # Monday cron logs a plan and scores last week's, so this accumulates one
    # observation per week. Cached so reruns don't hit Neon.
    try:
        _cov = _cached_pi_coverage(ticker)
        if _cov["n"] == 0:
            _cov_txt = "PI calibration: no scored weeks yet — accumulating"
        elif not _cov["sufficient"]:
            _cov_txt = (f"PI calibration: {_cov['one_sided']:.0%} of {_cov['n']} "
                        f"scored weeks inside the PI upper — accumulating "
                        f"(needs 15+ for conclusions)")
        else:
            _buf = _cov.get("buffer", {})
            _buf_txt = (f" · buffer breach {_buf['breach_rate']:.0%}"
                        if _buf.get("n") else "")
            _cov_txt = (f"PI calibration: {_cov['one_sided']:.0%} one-sided over "
                        f"{_cov['n']} scored weeks (nominal "
                        f"{_cov['nominal_one_sided']:.0%}){_buf_txt}")
        st.caption(_cov_txt)
    except Exception:
        pass   # calibration display is never worth breaking the finder over

    # ── Excel export — ALL tickers, one week-named tab ──
    # The active ticker's row reuses the exact tiers rendered above (chain-
    # snapped strikes, live weekly-EM floor); the other four instruments are
    # rebuilt from persisted state (saved fits, Monday-open captures, DB EM
    # snapshots) via a 10-min cross-session cache so the eager download-button
    # payload build doesn't hammer Neon on every rerun.
    # ── Managed export-ticker list ──────────────────────────────────
    # Always-present defaults + whatever the user has added. The active ticker
    # is included only when it's a default or has been explicitly added, so
    # nothing gets silently swept in/out when switching tickers.
    _extras = [t for t in _xlsx_extra_list() if t not in _FT_DEFAULT_TICKERS]
    _export_tickers = _FT_DEFAULT_TICKERS + _extras
    if len(_export_tickers) > FT_MAX_TICKERS:
        # The Scoreboard scans a fixed 40-row window per week tab, so the
        # workbook builder truncates silently — surface the drop here instead.
        st.caption(
            f"⚠ The export caps at {FT_MAX_TICKERS} tickers — dropped: "
            f"{', '.join(_export_tickers[FT_MAX_TICKERS:])}. Remove some "
            "added tickers to include them."
        )

    st.html('<div class="sf-eyebrow">Tickers in this export</div>')
    st.markdown(
        '<div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">'
        'Always included: <span style="font-family:var(--mono);color:var(--text-secondary);">'
        + ' · '.join(_FT_DEFAULT_TICKERS)
        + '</span></div>',
        unsafe_allow_html=True,
    )
    # Added tickers as small removable chips — click a chip (✕) to drop it.
    if _extras:
        st.markdown(
            '<div style="font-size:9.5px;color:var(--text-dim);text-transform:uppercase;'
            'letter-spacing:.04em;margin:2px 0;">Added (click ✕ to remove)</div>',
            unsafe_allow_html=True,
        )
        st.pills(
            "Added export tickers",
            [f"{t}  ✕" for t in _extras],
            selection_mode="single",
            key="_sf_xlsx_rm_pills",
            label_visibility="collapsed",
            on_change=_cb_remove_extra_pill,
        )
    # Add the active ticker — compact button, only when it isn't already in.
    if ticker not in _export_tickers:
        st.button(
            f"➕ Add {ticker} to export",
            key=f"_sf_xlsx_add_{ticker}",
            on_click=_xlsx_add_extra, args=(ticker,),
        )
    elif ticker in _extras:
        st.caption(f"✓ {ticker} is in the export")

    _xlsx_col, _ = st.columns([1, 3])
    with _xlsx_col:
        try:
            _active_notes = [active_model]
            if gex_ctx.gamma_regime:
                _gflag = gex_adj.get("gex_regime_flag")
                _active_notes.append(
                    f"GEX research {gex_ctx.gamma_regime}"
                    + (f" ({_gflag:+d})" if isinstance(_gflag, int) else "")
                    + " — not applied"
                )
            _active_events = [n for n, f in (("FOMC", plan.has_fomc), ("CPI", plan.has_cpi),
                                             ("NFP", plan.has_nfp), ("OPEX", plan.has_opex)) if f]
            if _active_events:
                _active_notes.append("/".join(_active_events))
            if plan.buffer_pct:
                _active_notes.append(f"buf {plan.buffer_pct * 100:.2f}%")
            if plan.recommended_width:
                _active_notes.append(f"wing {plan.recommended_width:g}")
            if chain_exp:
                _active_notes.append(f"chain {chain_exp}")

            _active_row = {
                "ticker": ticker,
                "ref": round(float(spx_close_input), 2),
                "prev_close": _cached_prior_week_close(week_start, ticker),
                "bands": _tier_bands_from_tiers(spread_tiers),
                "notes": _active_notes,
                "error": None,
            }
            # Rebuild every NON-active export ticker from persisted state; slot
            # the live active row in if the active ticker is in the export list.
            _other_tickers = tuple(t for t in _export_tickers if t != ticker)
            _other_rows = {
                r["ticker"]: r
                for r in _cached_nonactive_week_bands(week_start, active_model, _other_tickers)
            }
            _ft_rows = [
                _active_row if t == ticker else _other_rows.get(
                    t, {"ticker": t, "error": "data collection failed",
                        "bands": {}, "notes": [], "ref": None, "prev_close": None},
                )
                for t in _export_tickers
            ]

            # Memoize the workbook on a content signature so an unrelated
            # widget rerun (ticker chip toggle, tier switch, auto-refresh
            # tick) doesn't rebuild the whole openpyxl workbook + formula
            # grid every pass — only an actual change to the rows/bands does.
            import hashlib as _hl, json as _json
            _sig = _hl.sha1(
                _json.dumps(_ft_rows, sort_keys=True, default=str).encode()
            ).hexdigest()
            _xlsx_sig_key = f"_sf_xlsx_sig_{week_start}_{active_model}"
            _xlsx_bytes_key = f"_sf_xlsx_bytes_{week_start}_{active_model}"
            if st.session_state.get(_xlsx_sig_key) != _sig:
                st.session_state[_xlsx_bytes_key] = build_forward_test_workbook(
                    week_start=week_start, model_choice=active_model, rows=_ft_rows,
                )
                st.session_state[_xlsx_sig_key] = _sig
            _xlsx_bytes = st.session_state[_xlsx_bytes_key]
            st.download_button(
                label       = "Export weekly workbook",
                data        = _xlsx_bytes,
                file_name   = f"har_forward_test_{week_start}_{active_model}.xlsx",
                mime        = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key         = f"_sf_xlsx_{week_start}_{active_model}",
                help        = (
                    "One tab named for this week's Monday with each instrument's "
                    "POINT / PI / EFFECTIVE bands, plus a ticker-keyed Scoreboard: "
                    "week tabs are matched by Instrument name, so the ticker mix "
                    "and row order can differ week to week. First download = your "
                    "master workbook; each later week, Move/Copy the new week tab "
                    "into it (any position). Re-importing a week already in the "
                    "master? Delete its old tab first — Excel silently renames "
                    "the paste to '(2)', which the Scoreboard ignores. Fill "
                    "'Weekly Close' after Friday and everything scores itself. "
                    "Upgrading an old master: delete its Scoreboard and bookend "
                    "tabs, Move/Copy this file's Scoreboard + FTBlank sheets in "
                    "(Ctrl+click both), and re-type any old tickers into blank "
                    "slots — history re-scores retroactively."
                ),
            )
            _missing = [r["ticker"] for r in _ft_rows if r.get("error")]
            if _missing:
                st.caption(
                    f"⚠ No bands for {', '.join(_missing)} — see the Notes column "
                    "in the export. Usually fixed by running Weekly Setup on that "
                    "ticker (needs a saved fit + Monday-open capture); the new "
                    "defaults populate after their first cron run."
                )
        except Exception as _xlsx_err:
            st.caption(f"Excel export unavailable: {_xlsx_err}")

    st.markdown("---")

    # =========================================================================
    # RISK TIER SELECTOR + DEPENDENT UI (wrapped in fragment for fast switching)
    # =========================================================================

    # Store everything the fragment needs in session_state so it doesn't
    # rely on closure variables that become stale across fragment reruns.
    st.session_state["_rtf_spread_tiers"] = spread_tiers
    st.session_state["_rtf_forecast"]     = forecast
    st.session_state["_rtf_plan"]         = plan
    st.session_state["_rtf_spx_close"]    = spx_close_input
    st.session_state["_rtf_gex_ctx"]      = gex_ctx
    st.session_state["_rtf_ticker"]       = ticker
    st.session_state["_rtf_weekly_em"]    = weekly_em
    st.session_state["_rtf_chain_exp"]    = chain_exp
    st.session_state["_rtf_spot"]         = spot
    st.session_state["_rtf_gex_adj"]      = gex_adj

    # Previously wrapped in @st.fragment to keep risk-tier clicks from
    # recomputing the plan. That saved ~100ms per tier toggle but the
    # cost was a nested-fragment pattern: outer @st.fragment on
    # _render_spread_finder_tab plus inner @st.fragment on
    # _risk_tier_fragment. Nested fragments schedule reruns
    # independently, and when the inner read `_rtf_*` session-state
    # that the outer had written, rapid ticker/tab switching could
    # leave the inner rendering against a stale snapshot — strikes
    # and widths would stop updating until a hard refresh.  Flattened
    # to a plain nested function: tier toggles now rerun the outer
    # fragment only (still tab-isolated, no full-app rerun), which
    # reuses the cached model fit and is fast enough (~100ms).
    def _risk_tier_fragment():
        _TIER_COLORS = {
            "aggressive":   "#ff4b4b",
            "moderate":     "#ffa726",
            "conservative": "#66bb6a",
        }

        # Read from session_state to avoid stale closure references
        _spread_tiers  = st.session_state["_rtf_spread_tiers"]
        _forecast      = st.session_state["_rtf_forecast"]
        _plan          = st.session_state["_rtf_plan"]
        _spx_close_inp = st.session_state["_rtf_spx_close"]
        _gex_ctx       = st.session_state["_rtf_gex_ctx"]
        _ticker        = st.session_state["_rtf_ticker"]
        _weekly_em     = st.session_state["_rtf_weekly_em"]
        _chain_exp     = st.session_state["_rtf_chain_exp"]
        _spot          = st.session_state["_rtf_spot"]
        _gex_adj       = st.session_state["_rtf_gex_adj"]

        tier_labels = [t.label for t in _spread_tiers]
        default_idx = len(tier_labels) - 1
        st.html('<div class="sf-eyebrow">Risk Tier</div>')
        # One full-width button per tier (st.columns → equal widths, no blank
        # gap), like the action buttons above. The active tier is the primary
        # (green) button; selection persists in session_state.
        _tier_state_key = f"sf_risk_tier_idx_{_ticker}"
        if _tier_state_key not in st.session_state:
            st.session_state[_tier_state_key] = default_idx
        for _i, _col in enumerate(st.columns(len(tier_labels))):
            with _col:
                st.button(
                    f"{tier_labels[_i]} ({_spread_tiers[_i].range_pct*100:.1f}%)",
                    key=f"sf_tier_btn_{_ticker}_{_i}",
                    type=("primary" if _i == st.session_state[_tier_state_key] else "secondary"),
                    use_container_width=True,
                    on_click=lambda idx=_i: st.session_state.__setitem__(_tier_state_key, idx),
                )
        selected_tier_idx = st.session_state[_tier_state_key]

        selected_tier = _spread_tiers[selected_tier_idx]

        # Present the HAR model's ORIGINAL strikes (before any weekly-EM floor).
        # The tier carries both: `.call_short`/`.put_short` are the EM-floored
        # (widened) strikes; `.model_call_short`/`.model_put_short` hold the
        # pre-floor strikes, but ONLY when the floor actually moved them (None
        # otherwise). Per design we no longer snap the short strike out to the
        # EM boundary — we show the model's own strike and flag it when it sits
        # inside the weekly expected move (warning below + EM band on the map).
        # Falling back to the floored value when the model field is None is
        # exact: None means the floor never moved that strike.
        displayed = displayed_tier(selected_tier)
        disp_call_short = displayed["call_short"]
        disp_put_short = displayed["put_short"]
        disp_call_spreads = displayed["call_spreads"]
        disp_put_spreads = displayed["put_spreads"]

        tier_color = _TIER_COLORS.get(selected_tier.risk_level, "#888")
        st.markdown(
            f"<span style='color:{tier_color};font-size:18px;font-weight:bold;'>"
            f"{selected_tier.risk_level.upper()}</span>"
            f" &nbsp;—&nbsp; Range: {selected_tier.range_pct*100:.2f}%"
            f" &nbsp;|&nbsp; Calls above `{_fmt_strike(disp_call_short)}`"
            f" &nbsp;|&nbsp; Puts below `{_fmt_strike(disp_put_short)}`",
            unsafe_allow_html=True,
        )

        # =====================================================================
        # RANGE GAUGE + STRIKE MAP (side by side)
        # =====================================================================

        col_gauge, col_strikes = st.columns([1, 1])

        with col_gauge:
            st.markdown("**Forecast Range**")
            _render_sf_range_gauge(
                _forecast, _plan, _spx_close_inp,
                tier_label=selected_tier.label,
                tier_range_pct=selected_tier.range_pct,
                tier_risk_level=selected_tier.risk_level,
            )

        with col_strikes:
            st.markdown("**Strike Map with GEX Walls**")
            _render_sf_strike_map_tier(
                selected_tier, _plan, _spx_close_inp, _gex_ctx,
                _plan.recommended_width, ticker=_ticker,
                weekly_em=_weekly_em,
            )

        st.markdown("---")

        st.markdown(f"**Spread Parameters — {selected_tier.label}**")

        col_call, col_put = st.columns(2)

        with col_call:
            st.markdown(f"Call Spreads — short above `{_fmt_strike(disp_call_short)}`")
            _render_sf_spread_table(disp_call_spreads)

        with col_put:
            st.markdown(f"Put Spreads — short below `{_fmt_strike(disp_put_short)}`")
            _render_sf_spread_table(disp_put_spreads)

        # Show credit source note with chain expiration
        all_tier_spreads = disp_call_spreads + disp_put_spreads
        has_market = any(getattr(s, "credit_source", "bsm") == "market" for s in all_tier_spreads)
        has_bsm = any(getattr(s, "credit_source", "bsm") == "bsm" for s in all_tier_spreads)
        exp_note = f" Chain: {_chain_exp}" if _chain_exp else ""
        if has_market and has_bsm:
            st.caption(f"Credits from Friday chain bid/ask.{exp_note} &nbsp;|&nbsp; * = BSM estimate (strike not in chain).")
        elif has_market:
            st.caption(f"Credits from Friday chain bid/ask (short bid - long ask).{exp_note}")
        else:
            st.caption("Credits are BSM estimates (no Friday chain data available). Verify with broker before trading.")

        # =====================================================================
        # GEX CONTEXT + WARNINGS
        # =====================================================================

        st.markdown("---")

        col_gex, col_warn = st.columns([1, 1])

        with col_gex:
            _render_gex_context_panel(_gex_ctx, _spot)

        with col_warn:
            st.html('<div class="sf-eyebrow">Warnings &amp; GEX Research Notes</div>')

            # One coherent GEX story: strips the anchored-regime warning and
            # composes a single anchored-vs-live message (flip-aware) instead
            # of concatenating two independent — and possibly contradictory —
            # regime signals.
            all_warnings = reconcile_gex_warnings(
                list(_plan.warnings),
                _gex_adj.get("gex_adjustment_notes", []),
                anchored_gex_flag=_plan.gex_flag,
                live_regime=_gex_adj.get("gex_regime"),
                anchor_label="Monday anchor",
            )

            # Inside-the-expected-move flags. The model no longer snaps the
            # short strike out to the EM boundary (that "EM floor" feature was
            # removed); instead we keep the model's own strike and warn when it
            # sits inside the weekly expected move. The model marks that case by
            # populating `model_*_short` (only set when the original strike was
            # inside the EM), so it's the authoritative inside-EM signal.
            _em_upper = (_weekly_em or {}).get("upper_level", 0) or 0
            _em_lower = (_weekly_em or {}).get("lower_level", 0) or 0
            if _em_upper > 0 and _em_lower > 0:
                if selected_tier.model_call_short is not None:
                    all_warnings.append(
                        f"Call short {_fmt_strike(disp_call_short)} is within the weekly "
                        f"expected move (EM upper {_em_upper:,.0f})."
                    )
                if selected_tier.model_put_short is not None:
                    all_warnings.append(
                        f"Put short {_fmt_strike(disp_put_short)} is within the weekly "
                        f"expected move (EM lower {_em_lower:,.0f})."
                    )

            _esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if all_warnings:
                st.html("".join(
                    f'<div class="sf-note warn" style="margin-bottom:6px;">{_esc(w)}</div>'
                    for w in all_warnings
                ))
            else:
                st.html('<div class="sf-note ok">No warnings for this week.</div>')

            # Event flags
            events = {"FOMC": _plan.has_fomc, "CPI": _plan.has_cpi, "NFP": _plan.has_nfp, "OPEX": _plan.has_opex}
            active = [k for k, v in events.items() if v]
            if active:
                st.html('<div class="sf-note info" style="margin-top:6px;">Events this week: '
                        f'<b style="color:var(--text-secondary);">{", ".join(active)}</b></div>')
            else:
                st.html('<div class="sf-note" style="margin-top:6px;">No major events this week</div>')

    _risk_tier_fragment()


def _render_gex_context_panel(gex_ctx: GEXContext, spot: float):
    """Render live GEX as research context, never a spread input."""
    gex_flag = regime_to_gex_flag(gex_ctx.gamma_regime)
    regime_color = SF_BULL if gex_flag == 1 else SF_BEAR if gex_flag == -1 else SF_NEUT
    zg_pct = (abs(spot - gex_ctx.zero_gamma) / spot * 100) if spot else 0.0

    st.html(
        '<div class="sf-gex">'
        '<div class="sf-eyebrow">Live GEX Context</div>'
        f'<div class="reg" style="color:{regime_color};">{gex_ctx.gamma_regime.title()}</div>'
        '<div class="row"><span class="k">Zero-Gamma</span>'
        f'<span class="v" style="color:var(--cyan);">${gex_ctx.zero_gamma:,.0f} '
        f'<span class="sub">({zg_pct:.2f}% from spot)</span></span></div>'
        '<div class="row"><span class="k">Call Wall</span>'
        f'<span class="v" style="color:var(--green);">${gex_ctx.call_wall:,.0f}</span></div>'
        '<div class="row"><span class="k">Put Wall</span>'
        f'<span class="v" style="color:var(--red);">${gex_ctx.put_wall:,.0f}</span></div>'
        '<div class="row"><span class="k">Spread Influence</span>'
        '<span class="v" style="color:var(--text-secondary);">Research / disabled</span></div>'
        '<div class="row"><span class="k">Spot</span>'
        f'<span class="v">${spot:,.2f}</span></div>'
        '</div>'
    )


def _render_sf_range_gauge(
    forecast: dict,
    plan: SpreadPlan,
    spx_ref: float,
    tier_label: str = None,
    tier_range_pct: float = None,
    tier_risk_level: str = None,
):
    """Horizontal number-line gauge of the weekly range forecast.

    The old view stacked four ascending bars (Lower PI, Point, Upper PI,
    Effective) which read like four independent predictions climbing in
    magnitude. They aren't — they're all positions on the same weekly
    range-% axis: Point is the model's central forecast, Lower/Upper PI
    are the 10th/90th percentiles of its predictive distribution, and
    Effective is Upper PI plus a fixed safety buffer. This view lays
    them out on one horizontal axis:

      - Amber band  = 80% prediction interval (Lower PI → Upper PI)
      - Red band    = buffer extension (Upper PI → Effective)
      - Bull dot    = Point Estimate (the center of the distribution)
      - Dashed tick = VIX-implied range for reference
      - White caret = the Risk Tier currently driving strike placement

    Because the forecast is fit on log(range), the back-transformed
    distribution is asymmetric — Point will usually sit below the
    midpoint of the PI band. That's statistically correct, not a bug,
    and the horizontal layout actually makes it visible (whereas the
    old four-bar chart hid it entirely).
    """
    lower_pct     = forecast["lower_pct"]         * 100
    point_pct     = forecast["point_pct"]         * 100
    upper_pct     = forecast["upper_pct"]         * 100
    effective_pct = plan.effective_range_pct      * 100
    vix_pct       = forecast["vix_implied_pct"]   * 100
    confidence    = forecast["confidence_level"]

    # Axis extends a touch past the biggest value so the right-most marker
    # doesn't collide with the track edge.
    axis_max = max(effective_pct, vix_pct, point_pct) * 1.18 + 0.4
    if axis_max <= 0:
        axis_max = 1.0

    def _p(x: float) -> float:  # value → % position on the axis, clamped
        return max(0.0, min(100.0, x / axis_max * 100.0))

    tier_colors = {"aggressive": "var(--red)", "moderate": "var(--amber)",
                   "conservative": "var(--green)"}
    tier_color = tier_colors.get((tier_risk_level or "").lower(), "#fff")

    lo, up, eff, pt, vx = (_p(lower_pct), _p(upper_pct), _p(effective_pct),
                           _p(point_pct), _p(vix_pct))

    caret = ""
    if tier_range_pct is not None:
        caret = (f'<div class="caret" style="left:{_p(tier_range_pct * 100):.2f}%;'
                 f'color:{tier_color};font-weight:700;">▼ {tier_label or "Selected"}</div>')

    st.html(
        '<div class="sf-gauge">'
        '<div class="track">'
        f'<div class="band" style="left:{lo:.2f}%;width:{max(0.0, up - lo):.2f}%;'
        'background:rgba(255,180,84,.55);border-radius:3px;"></div>'
        f'<div class="band" style="left:{up:.2f}%;width:{max(0.0, eff - up):.2f}%;'
        'background:rgba(255,77,104,.5);"></div>'
        f'<div class="vix" style="left:{vx:.2f}%;"></div>'
        f'<div class="dot" style="left:{pt:.2f}%;"></div>'
        f'{caret}'
        '</div>'
        '<div class="axis"><span>0%</span>'
        f'<span>{axis_max / 2:.1f}%</span><span>{axis_max:.1f}%</span></div>'
        '<div style="display:flex;flex-wrap:wrap;gap:10px 14px;margin-top:10px;'
        'font-family:var(--mono);font-size:10px;">'
        f'<span style="color:var(--amber);">Lower PI {lower_pct:.2f}%</span>'
        f'<span style="color:var(--green);">Point {point_pct:.2f}%</span>'
        f'<span style="color:var(--amber);">Upper PI {upper_pct:.2f}%</span>'
        f'<span style="color:var(--red);">Effective {effective_pct:.2f}%</span>'
        f'<span style="color:var(--text-dim);">VIX {vix_pct:.2f}% · {confidence}% PI</span>'
        '</div>'
        '</div>'
    )


def _render_sf_strike_map_tier(
    tier: SpreadTier, plan: SpreadPlan, spx_ref: float,
    gex_ctx: GEXContext, selected_width: float = 25, ticker: str = "SPX",
    weekly_em: dict = None,
):
    """Swim-lane strike map grouped by semantic role.

    The old layout sorted every level (walls, shorts, spot, EM bounds,
    etc.) onto its own Y-row — clean for avoiding label overlap but
    forced the reader to mentally re-group "which of these is my trade
    vs structural GEX vs a forecast marker?" every time they scanned.

    This version keeps one shared price axis on X but splits the Y into
    three labeled lanes:

      TRADE   — Put Long / Put Short / Ref / Spot / Call Short / Call Long
      RANGE   — Effective bounds + EM bounds
      GEX     — Put Wall / Zero-Gamma / Call Wall

    Background shading:
      - tier-colored band between Put Short and Call Short (the "safe"
        window for the selected risk tier — where the spread profits)
      - translucent blue band across the weekly EM envelope

    Label collisions within a lane are avoided by alternating marker
    text positions top/bottom after sorting each lane's markers by
    price.
    """
    tier_colors = {"aggressive": "var(--red)", "moderate": "var(--amber)",
                   "conservative": "var(--green)"}
    tier_color = tier_colors.get(tier.risk_level, "var(--text-muted)")

    # Get strikes from the selected tier. We show the model's ORIGINAL strikes
    # (pre weekly-EM floor) — see the render tab for rationale. `.model_*` is
    # populated only when the floor moved the strike; fall back to the
    # (unchanged) floored value otherwise so the map matches the tables.
    call_short = tier.model_call_short if tier.model_call_short is not None else tier.call_short
    put_short  = tier.model_put_short  if tier.model_put_short  is not None else tier.put_short
    call_spreads = tier.model_call_spreads or tier.call_spreads
    put_spreads  = tier.model_put_spreads  or tier.put_spreads

    # Default long strikes from first spread
    call_long = call_spreads[0].long_strike if call_spreads else call_short + 25
    put_long  = put_spreads[0].long_strike  if put_spreads  else put_short - 25

    # Draw the same spread the table stars (best qualifying per side);
    # fall back to the model-recommended width when nothing qualifies.
    _best_c = _best_spread_idx(call_spreads)
    if _best_c is not None:
        call_long = call_spreads[_best_c].long_strike
    else:
        for s in call_spreads:
            if s.wing_width == selected_width:
                call_long = s.long_strike
    _best_p = _best_spread_idx(put_spreads)
    if _best_p is not None:
        put_long = put_spreads[_best_p].long_strike
    else:
        for s in put_spreads:
            if s.wing_width == selected_width:
                put_long = s.long_strike

    # Weekly expected move from Friday straddle (already computed by GEX engine)
    em_upper = (weekly_em or {}).get("upper_level", 0)
    em_lower = (weekly_em or {}).get("lower_level", 0)
    has_em = em_upper > 0 and em_lower > 0
    em_color = "#29b6f6"  # light blue

    # ── Swim-lane geometry (price → % position on one shared X axis) ──
    trade = [
        (put_long,      "Put Long",      "var(--red)",   "◄"),
        (put_short,     "Put Short",     tier_color,     "◆"),
        (spx_ref,       f"{ticker} Ref", "#ffffff",      "★"),
        (gex_ctx.spot,  "Spot",          "var(--amber)", "●"),
        (call_short,    "Call Short",    tier_color,     "◆"),
        (call_long,     "Call Long",     "var(--red)",   "►"),
    ]
    rng = [
        (plan.effective_lower_px, "Eff Lo", "var(--amber)", "▲"),
        (plan.effective_upper_px, "Eff Hi", "var(--amber)", "▲"),
    ]
    if has_em:
        rng += [(em_lower, "EM Lo", "var(--blue)", "▲"),
                (em_upper, "EM Hi", "var(--blue)", "▲")]
    gex = [
        (gex_ctx.put_wall,   "Put Wall",  "var(--red)",   "■"),
        (gex_ctx.zero_gamma, "Zero-Γ",    "var(--cyan)",  "✕"),
        (gex_ctx.call_wall,  "Call Wall", "var(--green)", "■"),
    ]

    all_px = [m[0] for m in trade + rng + gex]
    pmin, pmax = min(all_px), max(all_px)
    margin = (pmax - pmin) * 0.12 or 5.0
    axis_min, axis_max = pmin - margin, pmax + margin
    span = (axis_max - axis_min) or 1.0

    def _xp(px: float) -> float:
        return max(0.0, min(100.0, (px - axis_min) / span * 100.0))

    ticks = [round((axis_min + span * f) / 5) * 5 for f in (0, 0.25, 0.5, 0.75, 1.0)]

    def _lane(markers, fmt=lambda px: f"{px:,.0f}") -> str:
        out = []
        for i, (px, _lbl, color, glyph) in enumerate(sorted(markers, key=lambda m: m[0])):
            lbl_pos = "bottom:15px;" if i % 2 == 0 else "top:15px;"
            out.append(
                f'<div class="mk" style="left:{_xp(px):.2f}%;color:{color};">{glyph}'
                f'<span class="lbl" style="position:absolute;left:50%;transform:translateX(-50%);'
                f'{lbl_pos}color:{color};">{_lbl} {fmt(px)}</span></div>'
            )
        return "".join(out)

    grid = "".join(f'<div class="grid" style="left:{_xp(t):.2f}%;"></div>' for t in ticks)
    refs = (
        f'<div class="grid" style="left:{_xp(spx_ref):.2f}%;border-left:2px solid rgba(231,237,245,.4);"></div>'
        f'<div class="grid" style="left:{_xp(gex_ctx.spot):.2f}%;border-left:1.5px dashed var(--amber);"></div>'
    )
    safe = (
        f'<div class="safe" style="left:{_xp(put_short):.2f}%;'
        f'width:{max(0.0, _xp(call_short) - _xp(put_short)):.2f}%;'
        f'background:color-mix(in srgb,{tier_color} 12%,transparent);'
        f'border-left:1px solid {tier_color};border-right:1px solid {tier_color};"></div>'
    )
    em_band = ""
    if has_em:
        em_band = (
            f'<div class="safe" style="left:{_xp(em_lower):.2f}%;'
            f'width:{max(0.0, _xp(em_upper) - _xp(em_lower)):.2f}%;'
            'background:rgba(110,168,255,.06);border-left:1px dashed var(--blue);'
            'border-right:1px dashed var(--blue);"></div>'
        )
    xaxis = "".join(
        f'<span style="position:absolute;left:{_xp(t):.2f}%;transform:translateX(-50%);">{t:,.0f}</span>'
        for t in ticks
    )

    st.html(
        '<div class="sf-map"><div class="yax">'
        '<div class="lane">TRADE</div><div class="lane">RANGE</div><div class="lane">GEX</div>'
        '</div><div class="area">'
        f'{safe}{em_band}{grid}{refs}'
        f'<div class="lane-row">{_lane(trade, _fmt_strike)}</div>'
        f'<div class="lane-row">{_lane(rng)}</div>'
        f'<div class="lane-row">{_lane(gex)}</div>'
        '</div></div>'
        f'<div class="sf-map-x" style="position:relative;height:14px;">{xaxis}</div>'
        f'<div class="sf-map-foot">{ticker} PRICE LEVEL</div>'
    )


def _best_spread_idx(spreads) -> "int | None":
    """Index of the 'best' spread in a width ladder, or None.

    With the short strike fixed across the ladder, credit RATIO is always
    highest on the narrowest width (the marginal credit per extra point of
    wing decays), so 'max ratio' would just re-create an arbitrary
    always-first-row highlight. Instead: the WIDEST spread whose credit
    still clears MIN_CREDIT_RATIO and the event minimum width — i.e. the
    most absolute premium and breach headroom you can take while still
    being paid at least the floor per point of risk. In high-IV tape more
    rungs qualify and the highlight walks wider; on dead/thin chains
    (0.00 credits) nothing qualifies and nothing is highlighted.
    """
    best = None
    for i, s in enumerate(spreads):
        if not getattr(s, "meets_min_credit", False):
            continue
        if getattr(s, "below_min_width", False):
            continue
        if best is None or s.wing_width > spreads[best].wing_width:
            best = i
    return best


def _fmt_strike(x) -> str:
    """Format a strike / wing width / level, preserving genuine fractional
    strikes (e.g. NVDA's 207.5 on the $2.5 grid) instead of rounding them to a
    misleading integer, while keeping whole values clean (208, not 208.0).

    Without this the table formatted strikes with ``:,.0f``, rounding 207.5 → 208
    so a true 2.5-wide spread looked 2-wide next to its "W = 2.5pt" label."""
    if x is None:
        return "—"
    x = float(x)
    if x == round(x):
        return f"{int(round(x)):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _render_sf_spread_table(spreads):
    """Render the spread ladder as an 8-column terminal table, highlighting the
    best qualifying spread (see _best_spread_idx). Presentation only — every
    value comes straight off the (untouched) SpreadOption objects."""
    if not spreads:
        st.html('<div class="sf-note">No spreads available.</div>')
        return

    best_idx = _best_spread_idx(spreads)

    body = []
    for i, s in enumerate(spreads):
        wlabel = f"{_fmt_strike(s.wing_width)}pt"
        if getattr(s, "below_min_width", False):
            wlabel += "*"
        if i == best_idx:
            wlabel = "★ " + wlabel
        cr = f"{s.estimated_credit:.2f}" + ("*" if getattr(s, "credit_source", "bsm") == "bsm" else "")
        ok = "Y" if s.meets_min_credit else "N"
        ok_cls = "oky" if s.meets_min_credit else "okn"
        body.append(
            f'<tr class="{"best" if i == best_idx else ""}">'
            f'<td>{wlabel}</td>'
            f'<td class="short">{_fmt_strike(s.short_strike)}</td>'
            f'<td class="long">{_fmt_strike(s.long_strike)}</td>'
            f'<td class="cr">{cr}</td>'
            f'<td>${s.max_loss:,.0f}</td>'
            f'<td>{s.breakeven:,.0f}</td>'
            f'<td>{s.credit_ratio:.1%}</td>'
            f'<td class="{ok_cls}">{ok}</td>'
            '</tr>'
        )

    st.html(
        '<div class="sf-table-wrap"><table class="sf-table"><thead><tr>'
        '<th>W</th><th>Short</th><th>Long</th><th>CR</th><th>MaxLoss</th>'
        '<th>BE</th><th>R%</th><th>OK</th>'
        '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>'
    )
    if best_idx is not None:
        st.caption(
            f"★ Best = widest wing still paying ≥{MIN_CREDIT_RATIO:.0%} of width "
            "(max premium + breach headroom at an acceptable credit-per-risk floor)."
        )
    else:
        st.caption(
            f"No spread clears the {MIN_CREDIT_RATIO:.0%} credit-to-width floor — "
            "credit is too thin at these strikes/widths to be worth the risk."
        )
