"""Bar source selection + validation (range_finder.bar_sources).

Pins the cutover safety contracts:

  * Tradier is primary; a failed/invalid Tradier series falls back to
    yfinance instead of erroring,
  * validation rejects empty, gapped, non-positive, and mislabeled series
    BEFORE they can displace yfinance data,
  * both sources produce the same output schema (lowercase ohlcv, normalized
    ascending index) so data_collector's renaming works either way,
  * SOURCE_MAP pins a symbol back to yfinance without touching fetch code.

No live API calls — TradierDataClient.get_history and yf.download are
mocked, mirroring phase1/tests/test_data_client.py conventions.
"""
import pandas as pd
import pytest

import range_finder.bar_sources as bs


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _tradier_days(dates_closes: dict) -> list[dict]:
    """Tradier /markets/history 'day' dicts."""
    return [
        {"date": d, "open": c - 1.0, "high": c + 1.0, "low": c - 2.0,
         "close": c, "volume": 1000}
        for d, c in dates_closes.items()
    ]


_WEEK_MONDAYS = {"2026-06-15": 100.0, "2026-06-22": 102.0, "2026-06-29": 104.0}
_DAILY_DATES = {"2026-06-29": 100.0, "2026-06-30": 101.0, "2026-07-01": 102.0}


class _FakeClient:
    def __init__(self, days):
        self._days = days

    def get_history(self, symbol, interval="daily", start=None, end=None):
        if isinstance(self._days, Exception):
            raise self._days
        return self._days


def _patch_tradier(monkeypatch, days) -> None:
    monkeypatch.setattr(bs, "_tradier_token_or_empty", lambda: "tok",
                        raising=False)
    import range_finder.data_collector as dc
    monkeypatch.setattr(dc, "_tradier_token", lambda: "tok")
    import phase1.data_client as pdc
    monkeypatch.setattr(pdc, "TradierDataClient", lambda tok: _FakeClient(days))


def _patch_yfinance(monkeypatch, dates_closes: dict) -> None:
    import yfinance as yf

    def _download(symbol, start=None, end=None, interval="1d",
                  progress=False, timeout=60):
        if not dates_closes:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_closes])
        closes = list(dates_closes.values())
        return pd.DataFrame({
            "Open": [c - 1.0 for c in closes], "High": [c + 1.0 for c in closes],
            "Low": [c - 2.0 for c in closes], "Close": closes,
            "Volume": [1000] * len(closes),
        }, index=idx)

    monkeypatch.setattr(yf, "download", _download)


# ── source selection ───────────────────────────────────────────────────────────

def test_tradier_is_primary(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_WEEK_MONDAYS))
    _patch_yfinance(monkeypatch, {"2026-06-15": 999.0})

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(100.0)  # Tradier, not yfinance
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2026-06-15")


def test_tradier_error_falls_back(monkeypatch):
    _patch_tradier(monkeypatch, ConnectionError("api down"))
    _patch_yfinance(monkeypatch, _WEEK_MONDAYS)

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3  # yfinance served the bars


def test_empty_tradier_falls_back(monkeypatch):
    _patch_tradier(monkeypatch, [])
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3


def test_no_token_goes_straight_to_yfinance(monkeypatch):
    import range_finder.data_collector as dc
    monkeypatch.setattr(dc, "_tradier_token", lambda: "")
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3


def test_source_map_pins_symbol_to_yfinance(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_WEEK_MONDAYS))
    _patch_yfinance(monkeypatch, {"2026-06-15": 999.0})
    monkeypatch.setitem(bs.SOURCE_MAP, "SPX", "yfinance")

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(999.0)  # pinned to yfinance


def test_no_tradier_symbol_uses_yfinance(monkeypatch):
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^WEIRD", None, 1.0, "WEIRD")

    assert len(df) == 3


# ── validation rejects bad primary series ──────────────────────────────────────

def test_gapped_series_falls_back(monkeypatch):
    gapped = _tradier_days({"2026-01-05": 100.0, "2026-03-02": 105.0})  # 8-week hole
    _patch_tradier(monkeypatch, gapped)
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(100.0)
    assert len(df) == 3  # yfinance won


def test_nonpositive_prices_fall_back(monkeypatch):
    bad = _tradier_days(_DAILY_DATES)
    bad[1]["close"] = 0.0
    _patch_tradier(monkeypatch, bad)
    _patch_yfinance(monkeypatch, {"2026-06-29": 55.0})

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(55.0)


def test_non_monday_weekly_labels_fall_back(monkeypatch):
    fridays = _tradier_days({"2026-06-19": 100.0, "2026-06-26": 102.0})
    _patch_tradier(monkeypatch, fridays)
    _patch_yfinance(monkeypatch, _WEEK_MONDAYS)

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df.index[0] == pd.Timestamp("2026-06-15")  # yfinance's Mondays


def test_holiday_monday_weeks_still_pass():
    # A weekly series labeled entirely on Mondays validates even when a
    # session gap (holiday week) exists inside the window.
    idx = pd.DatetimeIndex(["2026-06-15", "2026-06-22", "2026-06-29"])
    df = pd.DataFrame({
        "open": [99.0, 101.0, 103.0], "high": [101.0, 103.0, 105.0],
        "low": [98.0, 100.0, 102.0], "close": [100.0, 102.0, 104.0],
        "volume": [0.0, 0.0, 0.0],   # index volume can legitimately be 0
    }, index=idx)

    ok, reason = bs._validate_bars(df, "1wk")

    assert ok, reason


# ── output schema parity between sources ───────────────────────────────────────

def test_both_sources_same_schema(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_DAILY_DATES))
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    t = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")
    monkeypatch.setitem(bs.SOURCE_MAP, "SPX", "yfinance")
    y = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert list(t.columns) == list(y.columns)
    assert t.index.equals(y.index)
    assert (t.dtypes == y.dtypes).all()
