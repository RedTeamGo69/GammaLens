"""Synthetic Spread Finder inputs shared by the view-model tests and the
tab-level AppTest harness (tests/fixtures/spread_finder_app.py)."""
import numpy as np
import pandas as pd

WEEK = "2026-06-15"


def spread_finder_features(n: int = 120, week: str = WEEK, *, stale: bool = False,
                           vix: float = 16.0) -> pd.DataFrame:
    """Synthetic model_features whose last row is ``week``'s forecast row."""
    idx = pd.date_range(end=week, periods=n, freq="W-MON")
    rng = np.random.default_rng(3)
    pos = np.arange(n, dtype=float)
    d1 = 0.02 + 0.004 * np.sin(pos / 5) + rng.normal(0, 0.002, n)
    wk = 0.022 + 0.003 * np.cos(pos / 6)
    mo = 0.024 + 0.002 * np.sin(pos / 11)
    df = pd.DataFrame({"har_d1": d1, "har_w": wk, "har_m": mo,
                       "vix_close": vix, "event_count": 0}, index=idx)
    df["log_range"] = np.log(0.005 + 0.5 * d1 + 0.3 * wk + 0.2 * mo)
    df.loc[idx[-1], "log_range"] = np.nan                 # forecast scaffold
    prior = idx - pd.Timedelta(days=7)
    df["path_source_week"] = prior - (pd.Timedelta(days=7) if stale else pd.Timedelta(0))
    df.index.name = "week_start"
    return df
