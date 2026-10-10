# =============================================================================
# target_form_experiment.py
# Walk-forward comparison: no-touch (path) bands vs settlement (terminal)
# bands, and direction-neutral vs per-side placement.
#
# The question this answers with data instead of argument:
#   "XSP weeklys are European and cash-settled, so a spread's P&L at expiry
#    depends only on Friday's settlement price — but every strike is placed
#    off a band the week must never TOUCH. How much width does that costing
#    convention buy that the product never uses?"
#
# calibration.py already documents that strike-touch is strictly worse than
# range coverage, and estimate_side_share_quantile exists to close that gap.
# Neither addresses the prior question of whether touch is the right event to
# protect against at all. For a European cash-settled product it is not the
# assignment event; it is the mark-to-market / stop-out / hedge-adjustment
# event. Those are different risks deserving different widths, and collapsing
# them into one band silently prices the wider of the two.
#
# For each test week t (walk-forward, oldest -> newest):
#   * fit the production spec on feature rows strictly BEFORE t (pinned to
#     TRAIN_WINDOW_YEARS, mirroring the Monday cron),
#   * forecast the PI-upper range u_t off week t's feature row,
#   * estimate side shares from weekly bars strictly BEFORE t — both the
#     pooled direction-neutral q and separate q_up / q_dn,
#   * place bands at O_t*(1 +- u_t*s),
#   * score each band under BOTH conventions:
#       path     — survives if (H-O)/O and (O-L)/O both stay inside
#       terminal — survives if |close_Fri - O|/O stays inside
#
# No buffer anywhere — this isolates the convention change; the heuristic
# buffer rides on top identically either way.
#
# READ-ONLY: only SELECTs through the existing loaders. Nothing is written.
#
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.target_form_experiment
#   ... [--ticker SPX] [--weeks 310] [--spec M3_extended]
# =============================================================================

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from range_finder.calibration import _binomial_ci
from range_finder.har_model import (
    MODEL_SPECS,
    PI_ALPHA,
    SIDE_SHARE_BETA,
    SIDE_SHARE_MIN_OBS,
    TRAIN_WINDOW_YEARS,
    estimate_side_share_quantile,
    fit_model,
    forecast_next_week,
)

log = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 60      # refuse to score a week fit on less than this
MIN_SCORED = 30          # refuse conclusions below this many scored weeks

# Band multipliers applied to the PI-upper range forecast. The grid brackets
# the production side-share (~0.84 at beta=0.20) generously in both
# directions so the "what width does each convention need" question can be
# answered by reading across the survival curve.
BAND_GRID = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30)

# Survival levels to solve the grid for. 0.70 is the stated success metric;
# 0.85/0.90 bracket where a credit-spread book usually has to live.
TARGET_SURVIVAL = (0.70, 0.85, 0.90)


# =============================================================================
# PER-SIDE SIDE-SHARE — the asymmetry the pooled estimator cannot express
# =============================================================================

def estimate_side_share_per_side(
    weekly_df: pd.DataFrame,
    beta: float = SIDE_SHARE_BETA,
    min_obs: int = SIDE_SHARE_MIN_OBS,
) -> dict:
    """Separate (1-beta) quantiles of the up-side and down-side range shares.

    har_model.estimate_side_share_quantile pools s_up with 1 - s_up, which
    forces a direction-neutral q BY CONSTRUCTION. That is a deliberate choice
    — it doubles the sample and refuses to bet on direction — but it also
    makes the single most robust asymmetry in index returns (the downside
    tail is fatter and faster) structurally unrepresentable in the bands.

    This estimator keeps the two sides apart so the experiment can measure
    whether that asymmetry is worth carrying. It is NOT a directional bet:
    q_up and q_dn are both estimated from history, and neither depends on a
    view about the coming week.

    Returns {"q_up", "q_dn", "n", "beta"}; the q's are None below min_obs.
    """
    cols = ("spx_open", "spx_high", "spx_low")
    if weekly_df is None or weekly_df.empty or any(c not in weekly_df.columns for c in cols):
        return {"q_up": None, "q_dn": None, "n": 0, "beta": beta}

    o = weekly_df["spx_open"].astype(float)
    h = weekly_df["spx_high"].astype(float)
    l = weekly_df["spx_low"].astype(float)
    rng = h - l
    valid = (rng > 0) & o.notna() & h.notna() & l.notna() & (h >= o) & (o >= l)
    if int(valid.sum()) < min_obs:
        return {"q_up": None, "q_dn": None, "n": int(valid.sum()), "beta": beta}

    s_up = ((h - o) / rng)[valid].clip(0.0, 1.0)
    s_dn = 1.0 - s_up

    return {
        "q_up": round(max(float(s_up.quantile(1.0 - beta)), 0.5), 4),
        "q_dn": round(max(float(s_dn.quantile(1.0 - beta)), 0.5), 4),
        "n": int(valid.sum()),
        "beta": beta,
    }


# =============================================================================
# WALK-FORWARD SCORING
# =============================================================================

def _fit_and_forecast(train: pd.DataFrame, row: pd.Series, cols: list[str],
                      ref: float) -> "dict | None":
    """Production-shaped fit (OLS on log_range) + next-week forecast.

    The constant is added HERE, mirroring time_series_split (har_model.py) —
    omitting the intercept inflates residual variance and with it every PI.
    Same helper shape as side_placement_experiment._fit_and_forecast.
    """
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


def run_experiment(conn, ticker: str = "SPX", n_weeks: int = 310,
                   spec: str = "M3_extended") -> dict:
    from range_finder.feature_builder import _load_weekly_for_ticker, get_features

    feats = get_features(conn, ticker=ticker).sort_index()
    weekly = _load_weekly_for_ticker(conn, ticker).sort_index()
    if feats.empty or weekly.empty:
        raise RuntimeError("features or weekly bars empty — bootstrap first")

    spec_cols = [c for c in MODEL_SPECS[spec] if c in feats.columns]

    need = ["spx_open", "spx_high", "spx_low", "spx_close"]
    ok = weekly[need].notna().all(axis=1)
    ok &= (weekly["spx_high"] - weekly["spx_low"]) > 0
    scoreable = [t for t in feats.index
                 if t in weekly.index[ok]
                 and not pd.isna(feats.loc[t].get("log_range"))]
    test_weeks = scoreable[-n_weeks:]

    rows = []
    for t in test_weeks:
        train = feats[feats.index < t]
        train = train[train.index >= t - pd.Timedelta(days=int(365.25 * TRAIN_WINDOW_YEARS))]
        row = feats.loc[t]
        o = float(weekly.loc[t, "spx_open"])
        if o <= 0:
            continue
        fc = _fit_and_forecast(train, row, spec_cols, o)
        if fc is None:
            continue
        u = float(fc["upper_pct"])

        hist = weekly[weekly.index < t]
        pooled = estimate_side_share_quantile(hist)["q"]
        per_side = estimate_side_share_per_side(hist)

        up = (float(weekly.loc[t, "spx_high"]) - o) / o      # realized up excursion
        dn = (o - float(weekly.loc[t, "spx_low"])) / o       # realized down excursion
        term = (float(weekly.loc[t, "spx_close"]) - o) / o   # terminal (settlement) move

        rows.append({
            "week": t, "u": u,
            "up": up, "dn": dn, "term": term,
            "q_pooled": pooled,
            "q_up": per_side["q_up"], "q_dn": per_side["q_dn"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no scoreable weeks — is the features table populated?")

    return {
        "ticker": ticker, "spec": spec, "n": len(df),
        "grid": _score_grid(df),
        "placements": _score_placements(df),
        "asymmetry": _score_asymmetry(df),
        "_frame": df,
    }


def _survival(df: pd.DataFrame, band_up: pd.Series, band_dn: pd.Series) -> dict:
    """Survival rates under both conventions for a given per-side band.

    path     — neither excursion reached its band edge
    terminal — Friday's settlement landed inside the band
    """
    path = (df["up"] <= band_up) & (df["dn"] <= band_dn)
    terminal = (df["term"] <= band_up) & (-df["term"] <= band_dn)
    n = len(df)
    return {
        "n": n,
        "path": float(path.mean()),
        "path_ci": _binomial_ci(int(path.sum()), n),
        "terminal": float(terminal.mean()),
        "terminal_ci": _binomial_ci(int(terminal.sum()), n),
        "avg_band_pct": float(((band_up + band_dn) / 2).mean()),
    }


def _score_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Survival under both conventions across the symmetric band grid."""
    out = []
    for m in BAND_GRID:
        band = df["u"] * m
        s = _survival(df, band, band)
        out.append({"multiplier": m, "avg_band_pct": s["avg_band_pct"],
                    "path": s["path"], "terminal": s["terminal"],
                    "path_lo": s["path_ci"][0], "path_hi": s["path_ci"][1],
                    "terminal_lo": s["terminal_ci"][0],
                    "terminal_hi": s["terminal_ci"][1]})
    return pd.DataFrame(out)


def _score_placements(df: pd.DataFrame) -> dict:
    """The production placement (pooled q) vs per-side placement, both
    conventions. Weeks without an estimable q are dropped per-arm."""
    out = {}

    pooled = df[df["q_pooled"].notna()]
    if len(pooled):
        band = pooled["u"] * pooled["q_pooled"]
        out["pooled (production)"] = _survival(pooled, band, band)

    sided = df[df["q_up"].notna() & df["q_dn"].notna()]
    if len(sided):
        out["per-side"] = _survival(
            sided, sided["u"] * sided["q_up"], sided["u"] * sided["q_dn"]
        )
    return out


def _score_asymmetry(df: pd.DataFrame) -> dict:
    """Realized tail asymmetry, and what the estimators made of it."""
    excursion_q = {q: {"up": float(df["up"].quantile(q)),
                       "dn": float(df["dn"].quantile(q))}
                   for q in (0.50, 0.80, 0.90, 0.95)}
    est = {}
    sided = df[df["q_up"].notna() & df["q_dn"].notna()]
    if len(sided):
        est = {"q_up": float(sided["q_up"].mean()),
               "q_dn": float(sided["q_dn"].mean()),
               "q_pooled": float(sided["q_pooled"].mean())
               if sided["q_pooled"].notna().any() else float("nan")}
    return {"excursion_quantiles": excursion_q, "avg_estimates": est}


def solve_required_multiplier(grid: pd.DataFrame, convention: str,
                              target: float) -> "float | None":
    """Smallest grid multiplier whose survival rate reaches `target`.

    None when even the widest grid point falls short — reported as such
    rather than extrapolated, because the grid's top end is already well
    beyond any band the book would sell.
    """
    hits = grid[grid[convention] >= target]
    return float(hits["multiplier"].min()) if len(hits) else None


def breakeven_credit_ratio(loss_rate: float) -> float:
    """Credit/width ratio at which a vertical's expectancy is exactly zero.

    With credit c, width w and loss probability p, treating a loss as the
    full width (the conservative convention update_expiration_outcome
    already stores):

        E = (1-p)*c - p*(w - c) = c - p*w

    so E = 0 at c/w = p. The required credit ratio IS the loss rate — a
    convenient identity for reading a survival number as a P&L constraint.
    """
    return float(loss_rate)


# =============================================================================
# REPORT
# =============================================================================

def _fmt_pct(x) -> str:
    return "-" if x is None or x != x else f"{x:.1%}"


def print_report(res: dict) -> None:
    df = res["_frame"]
    print(f"\nTarget-form walk-forward — {res['ticker']} / {res['spec']}, "
          f"{res['n']} scored weeks "
          f"({df['week'].min().date()} -> {df['week'].max().date()})")

    print("\n  SURVIVAL BY BAND WIDTH (symmetric, band = PI-upper x multiplier)")
    print(f"  {'mult':>6}{'avg band':>11}{'path (no-touch)':>26}"
          f"{'terminal (settlement)':>28}")
    grid = res["grid"]
    for _, r in grid.iterrows():
        print(f"  {r['multiplier']:>6.2f}{r['avg_band_pct']:>10.2%}   "
              f"{_fmt_pct(r['path']):>6} "
              f"(CI {_fmt_pct(r['path_lo'])}-{_fmt_pct(r['path_hi'])})   "
              f"{_fmt_pct(r['terminal']):>6} "
              f"(CI {_fmt_pct(r['terminal_lo'])}-{_fmt_pct(r['terminal_hi'])})")

    print("\n  WIDTH REQUIRED TO REACH A SURVIVAL LEVEL")
    for target in TARGET_SURVIVAL:
        m_path = solve_required_multiplier(grid, "path", target)
        m_term = solve_required_multiplier(grid, "terminal", target)
        fmt = lambda m: f"{m:.2f}x" if m is not None else "off-grid"
        print(f"    {target:.0%} survival -> path {fmt(m_path):>9} | "
              f"terminal {fmt(m_term):>9}"
              + ("   <- terminal needs less width" if
                 (m_path is not None and m_term is not None and m_term < m_path)
                 else ""))

    print("\n  PLACEMENT (production side-share vs per-side)")
    for label, s in res["placements"].items():
        print(f"    {label:<22} band {s['avg_band_pct']:.2%}/side   "
              f"path {_fmt_pct(s['path'])} "
              f"(CI {_fmt_pct(s['path_ci'][0])}-{_fmt_pct(s['path_ci'][1])})   "
              f"terminal {_fmt_pct(s['terminal'])} "
              f"(CI {_fmt_pct(s['terminal_ci'][0])}-{_fmt_pct(s['terminal_ci'][1])})")
        print(f"    {'':<22} breakeven credit ratio: "
              f"path {breakeven_credit_ratio(1 - s['path']):.1%}  "
              f"terminal {breakeven_credit_ratio(1 - s['terminal']):.1%}")

    asym = res["asymmetry"]
    print("\n  REALIZED EXCURSION ASYMMETRY (fraction of anchor)")
    for q, v in asym["excursion_quantiles"].items():
        skew = v["dn"] - v["up"]
        print(f"    p{int(q * 100):<3} up {v['up']:.2%}   dn {v['dn']:.2%}   "
              f"(dn-up {skew:+.2%})")
    est = asym.get("avg_estimates") or {}
    if est:
        print(f"    estimator means: q_up {est['q_up']:.3f}  "
              f"q_dn {est['q_dn']:.3f}  pooled {est['q_pooled']:.3f}")

    if res["n"] < MIN_SCORED:
        print(f"\n  [!] n < {MIN_SCORED} - numbers shown, conclusions refused.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--weeks", type=int, default=310)
    parser.add_argument("--spec", default="M3_extended")
    args = parser.parse_args()

    # spread_levels' import-time basicConfig(INFO) wins the root config race —
    # override it so hundreds of walk-forward fits don't print their summaries.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    # No DATABASE_URL pre-check here: db.get_connection resolves it from
    # Streamlit secrets OR the environment, and raises its own clear message
    # when neither has it. An env-only guard would refuse runs that work.
    from range_finder.db import get_connection

    conn = get_connection()
    res = run_experiment(conn, ticker=args.ticker, n_weeks=args.weeks,
                         spec=args.spec)
    print_report(res)


if __name__ == "__main__":
    sys.exit(main())
