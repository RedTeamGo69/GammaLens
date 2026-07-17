# =============================================================================
# side_placement_experiment.py
# Walk-forward comparison: symmetric /2 split vs empirical side-share
# quantile placement (option 2 of the calibration-audit finding).
#
# The question this answers with data instead of argument:
#   "The range-coverage audit reads ~nominal — but how often does a week
#    TOUCH a band edge under each placement, and what does the extra
#    protection cost in band width?"
#
# For each test week t (walk-forward, oldest→newest):
#   * fit the production spec on feature rows strictly BEFORE t (pinned to
#     TRAIN_WINDOW_YEARS, mirroring the Monday cron),
#   * forecast the PI-upper range u_t off week t's feature row,
#   * estimate q_t = side-share quantile from weekly bars strictly BEFORE t,
#   * place bands at O_t*(1 ± u_t*s) for s ∈ {0.5, q_t(β=0.10), q_t(β=0.05)},
#   * score touches against week t's realized (H-O)/O and (O-L)/O.
#
# No buffer anywhere — this isolates the split change; the heuristic buffer
# rides on top identically under either placement.
#
# READ-ONLY: only SELECTs through the existing loaders. Nothing is written.
#
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.side_placement_experiment
#   ... [--ticker SPX] [--weeks 150] [--spec M3_extended]
# =============================================================================

import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

from range_finder.calibration import _binomial_ci
from range_finder.har_model import (
    MODEL_SPECS,
    PI_ALPHA,
    TRAIN_WINDOW_YEARS,
    estimate_side_share_quantile,
    fit_model,
    forecast_next_week,
)

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 60      # refuse to score a week fit on less than this
MIN_SCORED = 30          # refuse conclusions below this many scored weeks
BETAS = (0.30, 0.20, 0.10, 0.05)   # side-share tail masses to evaluate


def _fit_and_forecast(train: pd.DataFrame, row: pd.Series, cols: list[str],
                      ref: float) -> "dict | None":
    """Production-shaped fit (OLS on log_range) + next-week forecast.

    The constant is added HERE, mirroring time_series_split (har_model.py) —
    fit_model receives X with const in production, and omitting the
    intercept inflates residual variance ~4x and with it every PI."""
    import statsmodels.api as sm
    sub = train[cols + ["log_range"]].dropna()
    if len(sub) < MIN_TRAIN_ROWS:
        return None
    try:
        X = sm.add_constant(sub[cols])
        result = fit_model(X, sub["log_range"], model_name="exp")
        return forecast_next_week(result, row, cols, ref, alpha=PI_ALPHA)
    except Exception as e:
        log.debug(f"fit/forecast failed: {e}")
        return None


def run_experiment(conn, ticker: str = "SPX", n_weeks: int = 150,
                   spec: str = "M3_extended") -> dict:
    from range_finder.feature_builder import get_features, _load_weekly_for_ticker

    feats = get_features(conn, ticker=ticker)
    weekly = _load_weekly_for_ticker(conn, ticker).sort_index()
    if feats.empty or weekly.empty:
        raise RuntimeError("features or weekly bars empty — bootstrap first")
    feats = feats.sort_index()

    spec_cols = [c for c in MODEL_SPECS[spec] if c in feats.columns]

    # Scoreable weeks: realized target + OHLC present (completed weeks only).
    ok = weekly[["spx_open", "spx_high", "spx_low"]].notna().all(axis=1)
    ok &= (weekly["spx_high"] - weekly["spx_low"]) > 0
    scoreable = [t for t in feats.index
                 if t in weekly.index[ok] and not pd.isna(feats.loc[t].get("log_range"))]
    test_weeks = scoreable[-n_weeks:]

    shares = ["half"] + [f"q{int(b * 100):02d}" for b in BETAS]
    rows = []
    for t in test_weeks:
        train = feats[feats.index < t]
        train = train[train.index >= t - pd.Timedelta(days=int(365.25 * TRAIN_WINDOW_YEARS))]
        row = feats.loc[t]
        o = float(weekly.loc[t, "spx_open"])
        fc = _fit_and_forecast(train, row, spec_cols, o)
        if fc is None or o <= 0:
            continue
        u = float(fc["upper_pct"])

        hist = weekly[weekly.index < t]
        qs = {}
        for b in BETAS:
            est = estimate_side_share_quantile(hist, beta=b)
            qs[f"q{int(b * 100):02d}"] = est["q"]  # None → skip that arm below

        up = (float(weekly.loc[t, "spx_high"]) - o) / o
        dn = (o - float(weekly.loc[t, "spx_low"])) / o

        rec = {"week": t, "u": u, "up": up, "dn": dn,
               "range_covered": (up + dn) <= u}
        for name in shares:
            s = 0.5 if name == "half" else qs.get(name)
            if s is None:
                rec[f"{name}_touch"] = None
                rec[f"{name}_width"] = None
                continue
            band = u * s
            rec[f"{name}_share"] = s
            rec[f"{name}_touch"] = (up > band) or (dn > band)
            rec[f"{name}_call_touch"] = up > band
            rec[f"{name}_put_touch"] = dn > band
            rec[f"{name}_width"] = band
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no scoreable weeks — is the features table populated?")

    out = {"ticker": ticker, "spec": spec, "n": len(df),
           "range_coverage": float(df["range_covered"].mean()),
           "arms": {}}
    for name in shares:
        scored = df[df[f"{name}_touch"].notna()]
        n = len(scored)
        if n == 0:
            continue
        touched = scored[f"{name}_touch"].astype(bool)
        out["arms"][name] = {
            "n": n,
            "touch_rate": float(touched.mean()),
            "touch_ci": _binomial_ci(int(touched.sum()), n),
            "call_touch": float(scored[f"{name}_call_touch"].astype(bool).mean()),
            "put_touch": float(scored[f"{name}_put_touch"].astype(bool).mean()),
            "avg_band_pct": float(scored[f"{name}_width"].mean()),
            "avg_share": float(scored.get(f"{name}_share", pd.Series([0.5] * n)).mean()),
        }

    # Anchor-position evidence: where does the open sit in the realized range?
    s_up = df["up"] / (df["up"] + df["dn"])
    out["s_up_quantiles"] = {q: float(s_up.quantile(q))
                             for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
    out["_frame"] = df
    return out


def _fmt_pct(x):
    return "-" if x is None or x != x else f"{x:.1%}"


def print_report(res: dict) -> None:
    print(f"\nSide-placement walk-forward — {res['ticker']} / {res['spec']}, "
          f"{res['n']} scored weeks")
    print(f"  range coverage (sanity, nominal ~90%): "
          f"{_fmt_pct(res['range_coverage'])}")
    s = res["s_up_quantiles"]
    print(f"  open's position in realized range (s_up quantiles): "
          f"p10={s[0.10]:.2f} p25={s[0.25]:.2f} p50={s[0.50]:.2f} "
          f"p75={s[0.75]:.2f} p90={s[0.90]:.2f}")
    print(f"  {'placement':<12}{'any-touch':<22}{'call':<8}{'put':<8}"
          f"{'avg band/side':<15}{'avg share'}")
    for name, a in res["arms"].items():
        lo, hi = a["touch_ci"]
        label = {"half": "legacy /2"}.get(name, f"side-share {name}")
        print(f"  {label:<12}{_fmt_pct(a['touch_rate']):>8} "
              f"(CI {_fmt_pct(lo)}-{_fmt_pct(hi)})   "
              f"{_fmt_pct(a['call_touch']):<8}{_fmt_pct(a['put_touch']):<8}"
              f"{a['avg_band_pct']:.2%}{'':<8}{a['avg_share']:.3f}")
    if res["n"] < MIN_SCORED:
        print(f"  [!] n < {MIN_SCORED} - numbers shown, conclusions refused.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--weeks", type=int, default=150)
    parser.add_argument("--spec", default="M3_extended")
    args = parser.parse_args()

    # spread_levels' import-time basicConfig(INFO) wins the root config race —
    # override it so 150 walk-forward fits don't print 150 summary tables.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    from range_finder.db import get_connection
    conn = get_connection()
    res = run_experiment(conn, ticker=args.ticker, n_weeks=args.weeks,
                         spec=args.spec)
    print_report(res)


if __name__ == "__main__":
    sys.exit(main())
