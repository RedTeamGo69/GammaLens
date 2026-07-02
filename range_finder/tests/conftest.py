"""Shared fixtures for range_finder tests.

Hermeticity guard: `.streamlit/secrets.toml` exists in the repo root on the
dev machine, so `data_collector._tradier_token()` resolves a REAL token even
under pytest — a Tradier-first anchor lookup would silently hit the live API
from unit tests. The autouse fixture below pins the anchor-open source list
to the yfinance path (which every test stubs) for ALL tests; tests that
exercise the Tradier/Cboe sources re-patch `_ANCHOR_OPEN_SOURCES` (or call
`_open_from_tradier` / `_open_from_cboe` directly with their internals
mocked) explicitly.
"""
import pytest

import range_finder.data_collector as dc


@pytest.fixture(autouse=True)
def _hermetic_anchor_sources(monkeypatch):
    monkeypatch.setattr(dc, "_ANCHOR_OPEN_SOURCES", [dc._open_from_yf])
