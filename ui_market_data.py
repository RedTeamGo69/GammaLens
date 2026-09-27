"""Cross-session cached Tradier lookups shared by the app and its tabs.

Lives outside streamlit_app.py on purpose: under ``streamlit run`` the app
module is ``__main__``, so importing ``streamlit_app`` from a tab loads a
second copy with separate caches (Streamlit keys caches by module name).
"""
from __future__ import annotations

import streamlit as st

from phase1.data_client import TradierDataClient


@st.cache_data(ttl=600, show_spinner=False)
def get_expirations_cached(tradier_token: str, ticker: str) -> list[str]:
    """Tradier expirations change at most once per day; cache for 10 minutes
    so the sidebar render doesn't hit the API on every widget rerun."""
    return TradierDataClient(token=tradier_token).get_expirations(ticker)


@st.cache_data(ttl=600, show_spinner=False)
def validate_ticker_cached(tradier_token: str, symbol: str):
    """Validate a typed symbol against Tradier (must be optionable).

    Returns the instrument dict ({symbol, type, name, has_options}) or None.
    Cached 10 minutes so re-typing a symbol doesn't re-hit the API."""
    return TradierDataClient(token=tradier_token).validate_ticker(symbol)
