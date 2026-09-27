"""Vol-proxy weekly bars go through the bar_sources seam (Tradier first,
validated, yfinance fallback) instead of raw yfinance downloads, and the
Tradier symbol map lives in one place."""
import pandas as pd
import pytest

import range_finder.bar_sources as bs
import range_finder.data_collector as dc

MONDAYS = pd.DatetimeIndex(["2026-06-01", "2026-06-08", "2026-06-15"])


def _bars(base: float) -> pd.DataFrame:
    return pd.DataFrame({"open": base, "high": base + 2, "low": base - 2,
                         "close": base + 1, "volume": 1.0}, index=MONDAYS)


@pytest.fixture
def bar_calls(monkeypatch):
    calls = []

    def fake(yf_symbol, tradier_symbol, years, label):
        calls.append((yf_symbol, tradier_symbol))
        if yf_symbol == "^BROKEN":
            raise RuntimeError("no source")
        return _bars(20.0 if yf_symbol.startswith("^") else 500.0)

    monkeypatch.setattr(bs, "fetch_weekly_bars", fake)
    import range_finder.cboe_data as cboe
    monkeypatch.setattr(cboe, "merge_cboe_weekly_ohlc", lambda df, idx, cols: df)
    return calls


@pytest.mark.parametrize("yf_symbol,expected", [
    ("^GSPC", "SPX"), ("^VIX", "VIX"), ("^VIX9D", "VIX9D"),
    ("^VXN", None),                    # Tradier can't quote it
    ("QQQ", "QQQ"),                    # plain symbols are their own
])
def test_tradier_symbol_for(yf_symbol, expected):
    assert bs.tradier_symbol_for(yf_symbol) == expected


def test_underlying_vol_proxy_uses_the_seam(bar_calls):
    df = dc.fetch_underlying_weekly("QQQ", "QQQ", "^VXN", years=1)
    assert bar_calls == [("QQQ", "QQQ"), ("^VXN", None)]
    assert df["vol_proxy_close"].iloc[0] == pytest.approx(21.0)


def test_underlying_vol_proxy_failure_degrades_to_nan(bar_calls):
    df = dc.fetch_underlying_weekly("QQQ", "QQQ", "^BROKEN", years=1)
    assert df["vol_proxy_close"].isna().all()
    assert len(df) == 3                 # the underlying rows survive


def test_spx_vix_seed_uses_the_seam(bar_calls):
    df = dc.fetch_spx_vix(years=1)
    assert bar_calls == [("^GSPC", "SPX"), ("^VIX", "VIX")]
    assert df["vix_close"].iloc[0] == pytest.approx(21.0)


def test_term_structure_fallback_uses_the_seam(monkeypatch):
    """Cboe is primary for VIX9D/VIX3M; the fallback goes through bar_sources
    (Tradier, then yfinance) instead of a raw yf.download."""
    import range_finder.cboe_data as cboe
    import range_finder.feature_builder as fb

    def cboe_down(index, name):
        raise ConnectionError("cdn down")

    calls = []

    def bars(yf_symbol, tradier_symbol, years, label):
        calls.append((yf_symbol, tradier_symbol))
        return _bars(18.0)

    monkeypatch.setattr(cboe, "fetch_cboe_weekly_closes", cboe_down)
    monkeypatch.setattr(bs, "fetch_weekly_bars", bars)
    out = fb.fetch_vix_term_structure(years=1)
    assert calls == [("^VIX9D", "VIX9D"), ("^VIX3M", "VIX3M")]
    assert out["vix9d_close"].notna().any()
