"""Range-based vol estimators (range_finder.vol_estimators).

Known-value checks against hand-computed Parkinson / Garman-Klass values,
warmup-NaN lengths, and degenerate-bar handling.
"""
import math

import numpy as np
import pandas as pd
import pytest

from range_finder.vol_estimators import (
    compute_garman_klass_vol,
    compute_parkinson_vol,
)


def _bars(n: int, high: float, low: float, open_: float = None,
          close: float = None) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=n, freq="B")
    return pd.DataFrame({
        "open": [open_ if open_ else low] * n,
        "high": [high] * n,
        "low": [low] * n,
        "close": [close if close else high] * n,
    }, index=idx)


# ── Parkinson ──────────────────────────────────────────────────────────────────

def test_parkinson_constant_range_hand_value():
    # Constant H/L=1.02 daily bars: sigma_bar^2 = ln(1.02)^2 / (4 ln 2)
    df = _bars(10, high=102.0, low=100.0)
    expected = math.sqrt(math.log(1.02) ** 2 / (4 * math.log(2))) * math.sqrt(252)

    out = compute_parkinson_vol(df, window=5)

    assert out.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_parkinson_warmup_length():
    df = _bars(10, high=102.0, low=100.0)
    out = compute_parkinson_vol(df, window=5)
    assert out.isna().sum() == 4            # strict full-window default
    out_loose = compute_parkinson_vol(df, window=5, min_periods=4)
    assert out_loose.isna().sum() == 3


def test_parkinson_weekly_cadence_scaling():
    df = _bars(10, high=102.0, low=100.0)
    daily = compute_parkinson_vol(df, window=5, trading_periods=252)
    weekly = compute_parkinson_vol(df, window=5, trading_periods=52)
    ratio = daily.iloc[-1] / weekly.iloc[-1]
    assert ratio == pytest.approx(math.sqrt(252 / 52), rel=1e-9)


def test_parkinson_ignores_bad_bars():
    df = _bars(10, high=102.0, low=100.0)
    df.loc[df.index[5], "low"] = 0.0        # degenerate bar
    out = compute_parkinson_vol(df, window=5)
    # windows containing the bad bar can't reach full min_periods -> NaN
    assert out.iloc[5:10].isna().sum() == 5


# ── Garman-Klass ───────────────────────────────────────────────────────────────

def test_gk_hand_value():
    # O=100, H=103, L=99, C=101 every bar:
    # var = 0.5*ln(103/99)^2 - (2ln2-1)*ln(101/100)^2
    df = _bars(10, high=103.0, low=99.0, open_=100.0, close=101.0)
    var_bar = (0.5 * math.log(103 / 99) ** 2
               - (2 * math.log(2) - 1) * math.log(101 / 100) ** 2)
    expected = math.sqrt(var_bar) * math.sqrt(252)

    out = compute_garman_klass_vol(df, window=5)

    assert out.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_gk_more_efficient_than_close_only_information():
    # Sanity: on identical range bars, GK and Parkinson land in the same
    # ballpark (both estimate the same sigma), not orders of magnitude apart.
    df = _bars(30, high=102.0, low=99.0, open_=100.0, close=101.0)
    gk = compute_garman_klass_vol(df, window=20).iloc[-1]
    park = compute_parkinson_vol(df, window=20).iloc[-1]
    assert 0.5 < gk / park < 2.0


def test_gk_clips_negative_bar_variance():
    # Huge close-open move with a tiny range makes the GK bar variance
    # negative — it must clip to 0, not go NaN via sqrt(negative).
    df = _bars(10, high=110.0, low=109.9, open_=100.0, close=110.0)
    out = compute_garman_klass_vol(df, window=5)
    assert (out.dropna() == 0.0).all()


def test_gk_warmup_length():
    df = _bars(10, high=103.0, low=99.0, open_=100.0, close=101.0)
    out = compute_garman_klass_vol(df, window=5)
    assert out.isna().sum() == 4


# ── custom column names (daily_spx uses spx_* prefixes) ───────────────────────

def test_custom_column_names():
    df = _bars(10, high=102.0, low=100.0).rename(columns={
        "open": "spx_open", "high": "spx_high",
        "low": "spx_low", "close": "spx_close",
    })
    out = compute_parkinson_vol(df, window=5,
                                high_col="spx_high", low_col="spx_low")
    assert out.notna().sum() == 6
