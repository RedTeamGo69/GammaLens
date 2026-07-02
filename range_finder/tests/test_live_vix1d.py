"""fetch_live_vix1d source selection (range_finder.data_collector).

Tradier /markets/quotes is primary (authenticated broker API); yfinance is
the fallback when the token is missing, the quote comes back empty, or the
request fails. The signature stays zero-arg — the 0DTE UI calls it blind.
"""
import pandas as pd
import pytest

import range_finder.data_collector as dc


class _FakeTradier:
    def __init__(self, spot):
        self._spot = spot

    def get_spot_price(self, ticker="SPX"):
        if isinstance(self._spot, Exception):
            raise self._spot
        return self._spot


class _FakeYfTicker:
    def __init__(self, close):
        self._close = close

    def history(self, period="5d"):
        if self._close is None:
            return pd.DataFrame()
        return pd.DataFrame({"Close": [self._close]},
                            index=pd.DatetimeIndex(["2026-07-01"]))


def _patch(monkeypatch, *, token, tradier_spot=None, yf_close=None):
    monkeypatch.setattr(dc, "_tradier_token", lambda: token)
    import phase1.data_client as pdc
    monkeypatch.setattr(pdc, "TradierDataClient",
                        lambda tok: _FakeTradier(tradier_spot))
    monkeypatch.setattr(dc.yf, "Ticker", lambda sym: _FakeYfTicker(yf_close))


def test_tradier_quote_is_primary(monkeypatch):
    _patch(monkeypatch, token="tok", tradier_spot=13.02, yf_close=99.0)
    assert dc.fetch_live_vix1d() == pytest.approx(13.02)


def test_no_token_falls_back_to_yfinance(monkeypatch):
    _patch(monkeypatch, token="", tradier_spot=13.02, yf_close=14.5)
    assert dc.fetch_live_vix1d() == pytest.approx(14.5)


def test_zero_quote_falls_back_to_yfinance(monkeypatch):
    # get_spot_price degrades to 0.0 for unmatched symbols — that must not
    # be surfaced as a real VIX1D level.
    _patch(monkeypatch, token="tok", tradier_spot=0.0, yf_close=14.5)
    assert dc.fetch_live_vix1d() == pytest.approx(14.5)


def test_tradier_error_falls_back_to_yfinance(monkeypatch):
    _patch(monkeypatch, token="tok",
           tradier_spot=ConnectionError("api down"), yf_close=14.5)
    assert dc.fetch_live_vix1d() == pytest.approx(14.5)


def test_both_sources_down_returns_none(monkeypatch):
    _patch(monkeypatch, token="tok",
           tradier_spot=ConnectionError("api down"), yf_close=None)
    assert dc.fetch_live_vix1d() is None
