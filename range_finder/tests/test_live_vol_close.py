"""Live vol-proxy close (the VIX regime-shift breaker's input).

The Spread Finder's breaker compares a live VIX against its trailing level.
Its old fetch referenced a module-level ``yf`` that was never imported, so the
NameError was swallowed and the breaker silently never fired. These pin the
shared source chain that replaced it: Tradier quote first (VIX/VIX9D/…), then
yfinance for proxies Tradier can't quote (^VXN), and ``None`` — never an
invented number — when every source is empty.
"""
import pandas as pd
import pytest

import range_finder.data_collector as dc


class _FakeTicker:
    def __init__(self, frame):
        self._frame = frame

    def history(self, period="5d"):
        return self._frame


def test_live_vol_close_uses_first_source_with_a_value(monkeypatch):
    calls = []

    def empty(sym):
        calls.append(("empty", sym))
        return None

    def hit(sym):
        calls.append(("hit", sym))
        return 21.37

    monkeypatch.setattr(dc, "_LIVE_VOL_SOURCES", [empty, hit])
    assert dc.live_vol_close("^VIX") == pytest.approx(21.37)
    assert calls == [("empty", "^VIX"), ("hit", "^VIX")]


def test_live_vol_close_returns_none_when_all_sources_empty(monkeypatch):
    monkeypatch.setattr(dc, "_LIVE_VOL_SOURCES", [lambda s: None, lambda s: None])
    assert dc.live_vol_close("^VIX") is None


def test_live_vol_close_skips_a_raising_source(monkeypatch):
    def boom(sym):
        raise RuntimeError("network down")

    monkeypatch.setattr(dc, "_LIVE_VOL_SOURCES", [boom, lambda s: 19.5])
    assert dc.live_vol_close("^VIX") == pytest.approx(19.5)


def test_live_from_tradier_maps_symbol_and_resolves_quote(monkeypatch):
    seen = {}

    class _Client:
        def __init__(self, token):
            seen["token"] = token

        def get_full_quote(self, ticker):
            seen["ticker"] = ticker
            # after hours: null last must fall through to close
            return {"last": None, "close": 22.4, "prevclose": 20.0}

    import phase1.data_client as data_client
    monkeypatch.setattr(dc, "_tradier_token", lambda: "tok")
    monkeypatch.setattr(data_client, "TradierDataClient", _Client)
    assert dc._live_from_tradier("^VIX") == pytest.approx(22.4)
    assert seen == {"token": "tok", "ticker": "VIX"}


def test_live_from_tradier_declines_unquotable_proxy(monkeypatch):
    monkeypatch.setattr(dc, "_tradier_token", lambda: "tok")
    assert dc._live_from_tradier("^VXN") is None


def test_live_from_yf_reads_last_close(monkeypatch):
    frame = pd.DataFrame({"Close": [18.0, float("nan"), 24.25]})
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeTicker(frame))
    assert dc._live_from_yf("^VXN") == pytest.approx(24.25)


def test_live_from_yf_empty_history_is_none(monkeypatch):
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeTicker(pd.DataFrame()))
    assert dc._live_from_yf("^VXN") is None


def test_spread_finder_live_vol_close_delegates(monkeypatch):
    """The UI wrapper must reach the shared chain (regression: NameError)."""
    import ui_spread_finder as usf
    monkeypatch.setattr(dc, "_LIVE_VOL_SOURCES", [lambda s: 30.1])
    usf._live_vol_close.clear()
    assert usf._live_vol_close("^VIX") == pytest.approx(30.1)
