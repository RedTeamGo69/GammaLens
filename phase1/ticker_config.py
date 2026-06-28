"""
Central ticker configuration — single source of truth for every symbol the
dashboard supports.

Splits the per-ticker knobs (strike increment, wing widths, vol-index proxy,
etc.) out of `range_finder/spread_levels.py` so the GEX side can read them
without a circular import. `range_finder/spread_levels.py` re-exports the
spread-finder-specific keys for backwards compatibility.

Adding a new ticker only requires appending an entry here and confirming
Tradier returns a chain for it.
"""
from __future__ import annotations


# -----------------------------------------------------------------------------
# Per-ticker configuration
# -----------------------------------------------------------------------------
#
# Keys
#   display_name              — what to show in the sidebar
#   tradier_symbol            — symbol for Tradier API calls
#   yf_symbol                 — yfinance symbol for daily/weekly OHLC backfill
#   vol_proxy_yf              — yfinance symbol for the vol-index feature
#                               (^VIX for everything except QQQ which gets ^VXN)
#   strike_increment          — finest strike grid the chain typically lists
#   wing_widths               — credit-spread wing widths offered to the user
#   min_spread_width          — minimum-width floor by event regime
#   multiplier                — option contract multiplier (always 100 for the
#                               instruments we list)
#   dividend_yield            — annualised dividend yield for BS pricing.
#                               Currently informational; q=0 in BS today.
#   category                  — "index" | "etf" | "stock", used for sidebar
#                               grouping and feature-builder branching.
#   xsp_scale_to_spx          — legacy boolean (XSP only). Superseded by
#                               scale_divisor; kept for back-compat. Readers
#                               should call price_scale_divisor().
#   shares_har_with           — parent ticker whose scale-invariant HAR
#                               features + fit this "mini" reuses. XSP→"SPX",
#                               XND→"NDX". None for standalone/own-HAR tickers.
#                               Read via feature_source_ticker().
#   scale_divisor             — price ratio of the parent to this mini
#                               (XSP=SPX/10 → 10, XND=NDX/10 → 10). 1.0 default.
#                               Read via price_scale_divisor().
#   has_single_name_earnings  — single-stock earnings warning gate.
#                               True for AMZN/AMD; False for indexes/ETFs.
#   spread_finder_mode        — "spx_shared" : reuses a parent's HAR features
#                                              (SPX itself, XSP→SPX, XND→NDX —
#                                              the parent comes from
#                                              shares_har_with)
#                               "own_har"    : has its own historical OHLC
#                                              backfill + HAR fit
#                                              (QQQ, SPY, NDX, AMZN, AMD)
TICKER_CONFIG: dict[str, dict] = {
    "SPX": {
        "display_name": "SPX",
        "tradier_symbol": "SPX",
        "yf_symbol": "^GSPC",
        "vol_proxy_yf": "^VIX",
        "strike_increment": 5,
        "wing_widths": [50, 100, 150, 200],
        "min_spread_width": {"normal": 50, "event_1": 50, "event_2": 100, "fomc_week": 100},
        "multiplier": 100,
        "dividend_yield": 0.013,
        "category": "index",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": False,
        "spread_finder_mode": "spx_shared",
    },
    "XSP": {
        "display_name": "XSP",
        "tradier_symbol": "XSP",
        "yf_symbol": "^GSPC",   # XSP shares SPX's underlying history (price ≈ SPX/10)
        "vol_proxy_yf": "^VIX",
        "strike_increment": 1,
        "wing_widths": [5, 10, 15, 20, 25],
        "min_spread_width": {"normal": 5, "event_1": 5, "event_2": 10, "fomc_week": 10},
        "multiplier": 100,
        "dividend_yield": 0.013,
        "category": "index",
        "xsp_scale_to_spx": True,
        "shares_har_with": "SPX",
        "scale_divisor": 10,
        "has_single_name_earnings": False,
        "spread_finder_mode": "spx_shared",
    },
    "QQQ": {
        "display_name": "QQQ",
        "tradier_symbol": "QQQ",
        "yf_symbol": "QQQ",
        "vol_proxy_yf": "^VXN",
        "strike_increment": 1,
        "wing_widths": [15, 20, 25, 30, 35],
        "min_spread_width": {"normal": 15, "event_1": 15, "event_2": 20, "fomc_week": 25},
        "multiplier": 100,
        "dividend_yield": 0.0055,
        "category": "etf",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": False,
        "spread_finder_mode": "own_har",
    },
    "AMZN": {
        "display_name": "AMZN",
        "tradier_symbol": "AMZN",
        "yf_symbol": "AMZN",
        "vol_proxy_yf": "^VIX",
        "strike_increment": 2.5,
        "wing_widths": [10, 20, 30, 40, 50],
        "min_spread_width": {"normal": 10, "event_1": 10, "event_2": 20, "fomc_week": 20},
        "multiplier": 100,
        "dividend_yield": 0.0,
        "category": "stock",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": True,
        "spread_finder_mode": "own_har",
    },
    "AMD": {
        "display_name": "AMD",
        "tradier_symbol": "AMD",
        "yf_symbol": "AMD",
        "vol_proxy_yf": "^VIX",
        "strike_increment": 1,
        "wing_widths": [10, 20, 30, 40, 50],
        "min_spread_width": {"normal": 10, "event_1": 10, "event_2": 20, "fomc_week": 20},
        "multiplier": 100,
        "dividend_yield": 0.0,
        "category": "stock",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": True,
        "spread_finder_mode": "own_har",
    },
    "SPY": {
        "display_name": "SPY",
        "tradier_symbol": "SPY",
        "yf_symbol": "SPY",
        "vol_proxy_yf": "^VIX",          # S&P 500 ETF — VIX is the right vol gauge
        "strike_increment": 1,
        "wing_widths": [5, 10, 15, 20, 25],
        "min_spread_width": {"normal": 5, "event_1": 5, "event_2": 10, "fomc_week": 10},
        "multiplier": 100,
        "dividend_yield": 0.012,
        "category": "etf",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": False,
        "spread_finder_mode": "own_har",  # own HAR fit on SPY's own history
    },
    "NDX": {
        "display_name": "NDX",
        "tradier_symbol": "NDX",
        "yf_symbol": "^NDX",
        "vol_proxy_yf": "^VXN",          # Nasdaq-100 vol index
        "strike_increment": 10,
        "wing_widths": [200, 400, 600, 800],
        "min_spread_width": {"normal": 200, "event_1": 200, "event_2": 400, "fomc_week": 400},
        "multiplier": 100,
        "dividend_yield": 0.008,
        "category": "index",
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": False,
        "spread_finder_mode": "own_har",  # own HAR fit on ^NDX history
    },
    "XND": {
        "display_name": "XND",
        "tradier_symbol": "XND",
        "yf_symbol": "^NDX",             # rides on NDX's underlying history (price ≈ NDX/10)
        "vol_proxy_yf": "^VXN",
        "strike_increment": 1,
        "wing_widths": [20, 40, 60, 80],
        "min_spread_width": {"normal": 20, "event_1": 20, "event_2": 40, "fomc_week": 40},
        "multiplier": 100,
        "dividend_yield": 0.008,
        "category": "index",
        "xsp_scale_to_spx": False,        # uses scale_divisor instead (generalized)
        "shares_har_with": "NDX",         # reuses NDX's scale-invariant HAR features + fit
        "scale_divisor": 10,              # XND price ≈ NDX / 10 (micro index, like XSP→SPX)
        "has_single_name_earnings": False,
        "spread_finder_mode": "spx_shared",  # "shared" mode; parent = shares_har_with (NDX)
    },
}


# -----------------------------------------------------------------------------
# Dynamic config for arbitrary (non-curated) tickers
# -----------------------------------------------------------------------------
#
# The five entries above are hand-tuned. Any other optionable symbol the user
# types gets a config DERIVED from its live Tradier chain instead — most
# importantly a real ``strike_increment`` and spot-scaled ``wing_widths``, so a
# $50 stock no longer inherits SPX's 5-point grid / 50–200-point wings (the old
# silent ``get_config`` SPX fallback). ``resolve_config`` is the entry point;
# ``get_config`` keeps a safe generic default for the rare call sites that have
# no chain context.

# Wing widths target these fractions of spot (mirrors SPX's 1/2/3/4 %), each
# snapped to a whole number of strike increments.
_WING_TARGET_PCTS = (0.01, 0.02, 0.03, 0.04)


def derive_strike_increment(strikes) -> float:
    """Infer the listed strike spacing from a chain's strikes.

    Uses the modal (most common) positive gap between adjacent sorted strikes,
    which is robust to the wider gaps that appear in the far wings of a chain.
    Falls back to 1.0 when there isn't enough to measure.
    """
    try:
        uniq = sorted({round(float(s), 4) for s in strikes if s is not None})
    except (TypeError, ValueError):
        return 1.0
    if len(uniq) < 2:
        return 1.0

    from collections import Counter
    gaps = Counter()
    for a, b in zip(uniq, uniq[1:]):
        gap = round(b - a, 4)
        if gap > 0:
            gaps[gap] += 1
    if not gaps:
        return 1.0
    # Most common gap; tie-break toward the smaller increment.
    top = max(gaps.values())
    return min(g for g, c in gaps.items() if c == top)


def _derive_wing_widths(increment: float, spot: float) -> list[float]:
    """Spot-scaled wing widths snapped to whole strike increments."""
    if not spot or spot <= 0:
        # No spot context — offer a few plain increment multiples.
        base = [increment * m for m in (10, 20, 30, 40)]
        return sorted({round(w, 4) for w in base})
    widths = set()
    for pct in _WING_TARGET_PCTS:
        steps = max(1, round((spot * pct) / increment))
        widths.add(round(steps * increment, 4))
    return sorted(widths)


def _category_from_type(instrument_type: str | None) -> str:
    t = (instrument_type or "").lower()
    if t in ("index", "indexopt"):
        return "index"
    if t in ("etf", "etn"):
        return "etf"
    return "stock"


def build_dynamic_config(symbol: str, strikes=None, spot: float = 0.0,
                         instrument_type: str | None = None) -> dict:
    """Build a TICKER_CONFIG-shaped dict for an arbitrary optionable symbol.

    ``strike_increment`` and ``wing_widths`` come from the live chain (when
    ``strikes``/``spot`` are supplied); everything else uses sensible
    single-name defaults. Tradier symbol == yfinance symbol for ordinary
    stocks/ETFs, so ``yf_symbol`` mirrors the symbol (indices are the only
    exception, and those are all curated above).
    """
    sym = (symbol or "").upper()
    increment = derive_strike_increment(strikes) if strikes else 1.0
    wings = _derive_wing_widths(increment, spot)
    cat = _category_from_type(instrument_type)
    # min_spread_width floors keyed off the smallest offered wing, doubling
    # for the two-event / FOMC regimes (mirrors the curated entries' shape).
    w0 = wings[0] if wings else increment
    w1 = wings[1] if len(wings) > 1 else w0
    return {
        "display_name": sym,
        "tradier_symbol": sym,
        "yf_symbol": sym,
        "vol_proxy_yf": "^VIX",
        "strike_increment": increment,
        "wing_widths": wings,
        "min_spread_width": {"normal": w0, "event_1": w0, "event_2": w1, "fomc_week": w1},
        "multiplier": 100,
        "dividend_yield": 0.0,
        "category": cat,
        "xsp_scale_to_spx": False,
        "has_single_name_earnings": cat == "stock",
        "spread_finder_mode": "own_har",
        "is_dynamic": True,
    }


def resolve_config(symbol: str, *, strikes=None, spot: float = 0.0,
                   instrument_type: str | None = None) -> dict:
    """Return the curated config for a known ticker, otherwise derive one.

    This is the primary entry point for the GEX + Spread Finder flow. Curated
    tickers (SPX/XSP/QQQ/AMZN/AMD) are returned verbatim so their hand-tuned
    behavior is unchanged; any other symbol gets a chain-derived config.
    """
    sym = (symbol or "").upper()
    if sym in TICKER_CONFIG:
        return TICKER_CONFIG[sym]
    return build_dynamic_config(sym, strikes=strikes, spot=spot,
                                instrument_type=instrument_type)


def register_dynamic_config(symbol: str, *, strikes=None, spot: float = 0.0,
                            instrument_type: str | None = None) -> dict:
    """Resolve a symbol's config and register it into ``TICKER_CONFIG``.

    Curated tickers are returned untouched (never overwritten). For any other
    optionable symbol the chain-derived config is stored under its key so the
    entire existing codebase — ``get_config`` / ``get_ticker_config`` in
    spread_levels, the feature-builder's ``uses_own_har`` / ``yf_symbol`` /
    ``vol_proxy_yf`` routing, and the Spread Finder's ``RF_TICKER_CONFIG``
    lookups — sees the right increment / wing widths without any per-call-site
    plumbing. The derived config is deterministic from the chain, so storing it
    in the process-global dict is safe across Streamlit reruns and sessions.
    """
    sym = (symbol or "").upper()
    if not sym:
        return TICKER_CONFIG["SPX"]
    if sym in TICKER_CONFIG and not TICKER_CONFIG[sym].get("is_dynamic"):
        # Curated entry — never clobber the hand-tuned config.
        return TICKER_CONFIG[sym]
    cfg = build_dynamic_config(sym, strikes=strikes, spot=spot,
                               instrument_type=instrument_type)
    TICKER_CONFIG[sym] = cfg
    return cfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_config(ticker: str) -> dict:
    """Return the config dict for a ticker.

    Curated tickers return their hand-tuned entry. Unknown tickers get a
    generic single-name default (placeholder 1.0 strike increment) rather than
    silently inheriting SPX's 5-point grid — call ``resolve_config`` with chain
    context to get the real increment / wing widths.
    """
    sym = (ticker or "SPX").upper()
    if sym in TICKER_CONFIG:
        return TICKER_CONFIG[sym]
    return build_dynamic_config(sym)


def all_tickers() -> list[str]:
    """All supported tickers in insertion order."""
    return list(TICKER_CONFIG.keys())


def tickers_by_category() -> dict[str, list[str]]:
    """Tickers grouped by category for grouped UI rendering."""
    out: dict[str, list[str]] = {}
    for sym, cfg in TICKER_CONFIG.items():
        out.setdefault(cfg["category"], []).append(sym)
    return out


def is_spread_finder_eligible(ticker: str) -> bool:
    """True if the Spread Finder UI should render for this ticker."""
    return get_config(ticker)["spread_finder_mode"] in ("spx_shared", "own_har")


def uses_own_har(ticker: str) -> bool:
    """True if the ticker has its own HAR pipeline (separate from SPX's)."""
    return get_config(ticker)["spread_finder_mode"] == "own_har"


def feature_source_ticker(ticker: str) -> str:
    """The ticker whose HAR features + saved fit this one should read.

    A scaled "mini" (XSP, XND) reuses its parent's scale-invariant features
    (returns / log-ranges are identical across the /10 scale) — the parent is
    declared in ``shares_har_with``. Everything else sources from itself; the
    legacy ``spx_shared`` mode with no explicit parent means SPX (the base).
    Generalises the old hardcoded ``ticker if uses_own_har(ticker) else "SPX"``.
    """
    cfg = get_config(ticker)
    parent = cfg.get("shares_har_with")
    if parent:
        return parent.upper()
    if cfg.get("spread_finder_mode") == "spx_shared":
        return "SPX"
    return (ticker or "SPX").upper()


def price_scale_divisor(ticker: str) -> float:
    """Divisor mapping a parent's price level to this mini's (XSP = SPX / 10).

    1.0 for everything that isn't a scaled mini. Prefers the generalized
    ``scale_divisor`` field, falling back to the legacy ``xsp_scale_to_spx``
    boolean so any un-migrated reader still scales XSP correctly.
    """
    cfg = get_config(ticker)
    div = cfg.get("scale_divisor")
    if div:
        return float(div)
    if cfg.get("xsp_scale_to_spx"):
        return 10.0
    return 1.0


def has_single_name_earnings(ticker: str) -> bool:
    """True for tickers that report quarterly earnings (single-stock gate)."""
    return bool(get_config(ticker).get("has_single_name_earnings", False))
