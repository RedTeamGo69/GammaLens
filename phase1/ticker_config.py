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
#   xsp_scale_to_spx          — only XSP=True; XSP price ≈ SPX/10 and reuses
#                               the SPX HAR features at 1/10 scale.
#   has_single_name_earnings  — single-stock earnings warning gate.
#                               True for AMZN/AMD; False for indexes/ETFs.
#   spread_finder_mode        — "spx_shared" : reuse SPX's HAR features
#                                              (SPX, XSP)
#                               "own_har"    : has its own historical OHLC
#                                              backfill + HAR fit
#                                              (QQQ, AMZN, AMD)
TICKER_CONFIG: dict[str, dict] = {
    "SPX": {
        "display_name": "SPX",
        "tradier_symbol": "SPX",
        "yf_symbol": "^GSPC",
        "vol_proxy_yf": "^VIX",
        "strike_increment": 5,
        "wing_widths": [100, 200, 300, 400, 500],
        "min_spread_width": {"normal": 100, "event_1": 100, "event_2": 200, "fomc_week": 200},
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
        "wing_widths": [5, 10, 15, 20],
        "min_spread_width": {"normal": 5, "event_1": 5, "event_2": 10, "fomc_week": 10},
        "multiplier": 100,
        "dividend_yield": 0.013,
        "category": "index",
        "xsp_scale_to_spx": True,
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
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_config(ticker: str) -> dict:
    """Return the config dict for a ticker, falling back to SPX on miss."""
    return TICKER_CONFIG.get((ticker or "SPX").upper(), TICKER_CONFIG["SPX"])


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


def has_single_name_earnings(ticker: str) -> bool:
    """True for tickers that report quarterly earnings (single-stock gate)."""
    return bool(get_config(ticker).get("has_single_name_earnings", False))
