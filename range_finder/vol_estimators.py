# =============================================================================
# vol_estimators.py
# Range-based volatility estimators from OHLC bars.
#
# The app's HV features (hv5/hv10/hv20, hv_ratio) use close-to-close log
# returns — the least efficient estimator OHLC data allows. Parkinson (1980)
# uses the high-low range (~5x more efficient under GBM); Garman-Klass
# (1980) adds open/close (~7.4x). Same data, sharper vol estimate — natural
# CANDIDATE features for the walk-forward experiments
# (range_finder/feature_experiments.py). Nothing in production consumes
# these until a candidate clears the adoption gate.
#
# Both work at any cadence: pass daily bars with trading_periods=252 or
# weekly bars with trading_periods=52.
# =============================================================================

import math

import numpy as np
import pandas as pd

_PARKINSON_FACTOR = 1.0 / (4.0 * math.log(2.0))
_GK_CO_FACTOR = 2.0 * math.log(2.0) - 1.0


def compute_parkinson_vol(
    df: pd.DataFrame,
    window: int,
    *,
    high_col: str = "high",
    low_col: str = "low",
    trading_periods: int = 252,
    min_periods: int = None,
) -> pd.Series:
    """Annualized Parkinson volatility over a rolling window.

        sigma^2_bar = (1 / (4 ln 2)) * ln(H/L)^2
        sigma_ann   = sqrt(rolling_mean(sigma^2_bar)) * sqrt(trading_periods)

    Bars with missing/non-positive H or L contribute NaN (rolling mean
    skips them only via min_periods — default requires a full window).
    """
    h = pd.to_numeric(df[high_col], errors="coerce")
    l = pd.to_numeric(df[low_col], errors="coerce")
    valid = (h > 0) & (l > 0)
    log_hl = pd.Series(np.where(valid, np.log(h / l), np.nan), index=df.index)

    var_bar = _PARKINSON_FACTOR * log_hl ** 2
    mp = min_periods if min_periods is not None else window
    out = np.sqrt(var_bar.rolling(window, min_periods=mp).mean()) \
        * math.sqrt(trading_periods)
    out.name = f"parkinson_{window}"
    return out


def compute_garman_klass_vol(
    df: pd.DataFrame,
    window: int,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    trading_periods: int = 252,
    min_periods: int = None,
) -> pd.Series:
    """Annualized Garman-Klass volatility over a rolling window.

        sigma^2_bar = 0.5 * ln(H/L)^2  -  (2 ln 2 - 1) * ln(C/O)^2
        sigma_ann   = sqrt(rolling_mean(sigma^2_bar)) * sqrt(trading_periods)

    The estimator can produce a (slightly) negative bar variance on
    degenerate bars; those are clipped to 0 before the rolling mean.
    """
    o = pd.to_numeric(df[open_col], errors="coerce")
    h = pd.to_numeric(df[high_col], errors="coerce")
    l = pd.to_numeric(df[low_col], errors="coerce")
    c = pd.to_numeric(df[close_col], errors="coerce")
    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)

    log_hl = pd.Series(np.where(valid, np.log(h / l), np.nan), index=df.index)
    log_co = pd.Series(np.where(valid, np.log(c / o), np.nan), index=df.index)

    var_bar = (0.5 * log_hl ** 2 - _GK_CO_FACTOR * log_co ** 2).clip(lower=0.0)
    mp = min_periods if min_periods is not None else window
    out = np.sqrt(var_bar.rolling(window, min_periods=mp).mean()) \
        * math.sqrt(trading_periods)
    out.name = f"garman_klass_{window}"
    return out
