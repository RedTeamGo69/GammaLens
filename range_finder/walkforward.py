# =============================================================================
# walkforward.py
# Expanding-window (walk-forward) out-of-sample evaluation.
#
# Everything the app previously called "OOS" came from ONE chronological
# 80/20 split — a single test window whose result can be luck. Walk-forward
# refits the model on an expanding window every `step` observations and
# predicts each subsequent observation with a fit that has never seen it,
# yielding an OOS series long enough to answer three questions the single
# split can't:
#   1. Which spec actually predicts better out-of-sample?
#   2. Does the "80%" prediction interval empirically cover ~80%?
#   3. Do candidate changes (WLS recency weighting, longer history, new
#      features) clear the adoption gate?
#
# OFFLINE ONLY: a comparison run is dozens-to-hundreds of OLS refits.
# Cheap on a laptop, pointless in the 9:30 ET cron. Nothing in the
# production path imports this module.
# =============================================================================

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from range_finder.har_model import (
    PI_ALPHA,
    _MIN_MAPE_IMPROVEMENT,
    _MIN_R2_IMPROVEMENT,
    _oos_r2,
    fit_model,
    fit_model_wls,
)

log = logging.getLogger(__name__)

# Coverage gate: a candidate may not move empirical two-sided coverage
# further from nominal than the baseline by more than this (2pp).
_MAX_COVERAGE_WORSENING = 0.02

# 2 years of weekly rows; daily callers should pass ~126 (half a trading
# year) explicitly — same rationale as the cadence-aware feature gate.
_DEFAULT_MIN_TRAIN = 104
_DEFAULT_STEP = 4

_FIT_FNS = {"ols": fit_model, "wls": fit_model_wls}


@dataclass(frozen=True)
class WalkForwardConfig:
    """One candidate to evaluate. Immutable so configs can't drift mid-run."""
    label: str
    feature_cols: tuple[str, ...]
    fit_fn_name: str = "ols"                     # "ols" | "wls"
    fit_kwargs: dict = field(default_factory=dict)
    min_date: str | None = None                  # filter df.index >= min_date
    max_train: int | None = None                 # rolling-window cap (rows); None = expanding


# =============================================================================
# CORE — expanding-window evaluation
# =============================================================================

def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "log_range",
    min_train: int = _DEFAULT_MIN_TRAIN,
    step: int = _DEFAULT_STEP,
    fit_fn=fit_model,
    fit_kwargs: dict = None,
    alpha: float = PI_ALPHA,
    max_train: int = None,
) -> pd.DataFrame:
    """Walk the series: refit on an expanding window every `step` obs,
    predict each subsequent observation out-of-sample.

    ``max_train`` caps the training window's row count (a ROLLING window —
    how the production "always train on the last N years" policy behaves);
    None keeps the window expanding. Either way the window always ends
    strictly before the predicted observation.

    Returns a DataFrame indexed like `df` (OOS observations only) with
    columns: y_true, y_pred, obs_ci_lower, obs_ci_upper, train_n.
    ``train_n`` is the size of the training window behind each prediction —
    by construction train_n <= the observation's position, i.e. no
    prediction ever comes from a fit that saw it.
    """
    fit_kwargs = fit_kwargs or {}
    cols_needed = [target_col] + list(feature_cols)
    clean = df[cols_needed].dropna()
    n = len(clean)
    if n <= min_train:
        raise ValueError(
            f"walk_forward_evaluate: {n} clean rows <= min_train={min_train}"
        )

    # Silence the per-fit summary tables fit_model/fit_model_wls log —
    # hundreds of refits would swamp the console. The functions themselves
    # stay untouched (they're the production fitters).
    har_logger = logging.getLogger("range_finder.har_model")
    prior_level = har_logger.level
    har_logger.setLevel(logging.WARNING)

    records = []
    result = None
    train_n = 0
    try:
        for i in range(min_train, n):
            if result is None or (i - min_train) % step == 0:
                lo = max(0, i - max_train) if max_train else 0
                train = clean.iloc[lo:i]
                X_tr = sm.add_constant(train[list(feature_cols)])
                y_tr = train[target_col]
                result = fit_fn(X_tr, y_tr, model_name=f"wf@{i}", **fit_kwargs)
                train_n = len(train)

            row = clean.iloc[[i]]
            X_new = sm.add_constant(row[list(feature_cols)], has_constant="add")
            X_new = X_new.reindex(columns=result.model.exog_names, fill_value=0.0)
            frame = result.get_prediction(X_new).summary_frame(alpha=alpha)

            records.append({
                "index": clean.index[i],
                "y_true": float(row[target_col].iloc[0]),
                "y_pred": float(frame["mean"].iloc[0]),
                "obs_ci_lower": float(frame["obs_ci_lower"].iloc[0]),
                "obs_ci_upper": float(frame["obs_ci_upper"].iloc[0]),
                "train_n": train_n,
            })
    finally:
        har_logger.setLevel(prior_level)

    out = pd.DataFrame.from_records(records).set_index("index")
    out.index.name = clean.index.name
    return out


# =============================================================================
# SUMMARY — metrics incl. empirical PI coverage
# =============================================================================

def summarize(frame: pd.DataFrame, alpha: float = PI_ALPHA) -> dict:
    """Aggregate a walk-forward output frame into comparison metrics.

    Coverage is computed on the log scale, which is equivalent to the pct
    scale (exp is monotone). Nominal two-sided = 1 - alpha; one-sided
    (what strike placement rides) = 1 - alpha/2.
    """
    y_true, y_pred = frame["y_true"], frame["y_pred"]

    mae_log = float((y_true - y_pred).abs().mean())
    mae_pct = float((np.exp(y_true) - np.exp(y_pred)).abs().mean())

    # Directional accuracy vs the previous realized value (same definition
    # as har_model.evaluate_oos).
    y_lag = y_true.shift(1).dropna()
    if len(y_lag):
        pred_al = y_pred.loc[y_lag.index]
        true_al = y_true.loc[y_lag.index]
        direction_acc = float(
            (np.sign(pred_al - y_lag.values) == np.sign(true_al - y_lag.values)).mean()
        )
    else:
        direction_acc = float("nan")

    two_sided = float(((y_true >= frame["obs_ci_lower"])
                       & (y_true <= frame["obs_ci_upper"])).mean())
    one_sided = float((y_true <= frame["obs_ci_upper"]).mean())

    return {
        "n_oos": int(len(frame)),
        "oos_r2": _oos_r2(y_true, y_pred),
        "mae_log": mae_log,
        "mae_pct": mae_pct,
        "direction_acc": direction_acc,
        "coverage_two_sided": two_sided,
        "coverage_one_sided": one_sided,
        "nominal_two_sided": 1 - alpha,
        "nominal_one_sided": 1 - alpha / 2,
        "coverage_gap": two_sided - (1 - alpha),
    }


# =============================================================================
# COMPARISON — candidates vs baseline with the adoption gate
# =============================================================================

def compare_configs(
    df: pd.DataFrame,
    configs: list[WalkForwardConfig],
    baseline_label: str,
    *,
    target_col: str = "log_range",
    min_train: int = _DEFAULT_MIN_TRAIN,
    step: int = _DEFAULT_STEP,
    alpha: float = PI_ALPHA,
) -> pd.DataFrame:
    """Walk-forward every config, compare against the named baseline, and
    apply the adoption gate (same thresholds as compare_enhancements):

        keep if (delta oos_r2 > _MIN_R2_IMPROVEMENT
                 OR delta mae_pct > _MIN_MAPE_IMPROVEMENT)
            AND two-sided coverage doesn't move further from nominal than
                the baseline's by more than 2pp.
    """
    results: dict[str, dict] = {}
    for cfg in configs:
        try:
            work = df
            if cfg.min_date:
                work = df[df.index >= pd.Timestamp(cfg.min_date)]
            frame = walk_forward_evaluate(
                work, list(cfg.feature_cols),
                target_col=target_col, min_train=min_train, step=step,
                fit_fn=_FIT_FNS[cfg.fit_fn_name], fit_kwargs=cfg.fit_kwargs,
                alpha=alpha, max_train=cfg.max_train,
            )
            results[cfg.label] = {"label": cfg.label, **summarize(frame, alpha)}
        except Exception as e:
            log.error(f"walk-forward failed for '{cfg.label}': {e}")

    if baseline_label not in results:
        raise RuntimeError(f"baseline '{baseline_label}' did not produce results")

    base = results[baseline_label]
    comp = pd.DataFrame(results.values())
    comp["delta_r2"] = comp["oos_r2"] - base["oos_r2"]
    comp["delta_mae_pct"] = base["mae_pct"] - comp["mae_pct"]

    base_cov_dist = abs(base["coverage_gap"])
    comp["cov_dist"] = comp["coverage_gap"].abs()
    comp["keep"] = (
        ((comp["delta_r2"] > _MIN_R2_IMPROVEMENT)
         | (comp["delta_mae_pct"] > _MIN_MAPE_IMPROVEMENT))
        & (comp["cov_dist"] <= base_cov_dist + _MAX_COVERAGE_WORSENING)
    )
    comp.loc[comp["label"] == baseline_label, "keep"] = False   # baseline isn't a candidate

    comp = comp.sort_values("oos_r2", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("  WALK-FORWARD COMPARISON — EXPANDING-WINDOW OUT-OF-SAMPLE")
    print("=" * 100)
    display = ["label", "n_oos", "oos_r2", "mae_pct", "direction_acc",
               "coverage_two_sided", "coverage_one_sided",
               "delta_r2", "delta_mae_pct", "keep"]
    print(comp[[c for c in display if c in comp.columns]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    ))
    print("=" * 100)
    print(f"  Gate: keep if delta_r2 > {_MIN_R2_IMPROVEMENT} OR delta_mae_pct > "
          f"{_MIN_MAPE_IMPROVEMENT * 100:.1f}pp, AND |coverage - nominal| doesn't "
          f"worsen by > {_MAX_COVERAGE_WORSENING * 100:.0f}pp vs baseline "
          f"(nominal two-sided {1 - alpha:.0%})")
    print("=" * 100 + "\n")

    return comp


# =============================================================================
# CLI — OLS vs WLS on the weekly SPX features (the compare_enhancements
# question, answered walk-forward this time)
# =============================================================================
# Usage:
#   DATABASE_URL=postgres://... python -m range_finder.walkforward [--spec M3_extended]

def main() -> None:
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        description="Walk-forward OLS-vs-WLS comparison on weekly SPX features")
    parser.add_argument("--spec", default="M3_extended",
                        help="Weekly spec to evaluate (default M3_extended)")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--step", type=int, default=_DEFAULT_STEP)
    parser.add_argument("--min-train", type=int, default=_DEFAULT_MIN_TRAIN)
    parser.add_argument("--conformal", action="store_true",
                        help="Evaluate the conformal-λ adoption gate on the "
                             "OLS walk-forward output")
    parser.add_argument("--persist", action="store_true",
                        help="With --conformal: persist λ to "
                             "interval_calibration if the gate passes")
    args = parser.parse_args()

    if not os.environ.get("DATABASE_URL", "").strip():
        log.error("DATABASE_URL not set — cannot read model_features.")
        sys.exit(1)

    from range_finder.db import get_connection
    from range_finder.feature_builder import get_features
    from range_finder.har_model import MODEL_SPECS, feature_has_enough_data

    conn = get_connection()
    df = get_features(conn, exclude_covid=True, ticker=args.ticker)
    if df.empty:
        log.error(f"model_features empty for {args.ticker}")
        sys.exit(1)

    feat_cols = tuple(c for c in MODEL_SPECS[args.spec]
                      if feature_has_enough_data(df, c))
    log.info(f"{args.spec} usable features: {list(feat_cols)}  "
             f"({len(df)} weekly rows)")

    configs = [
        WalkForwardConfig(label=f"{args.spec} OLS", feature_cols=feat_cols),
        WalkForwardConfig(label=f"{args.spec} WLS(hl=52)", feature_cols=feat_cols,
                          fit_fn_name="wls", fit_kwargs={"half_life": 52}),
    ]
    compare_configs(df, configs, baseline_label=f"{args.spec} OLS",
                    min_train=args.min_train, step=args.step)

    if args.conformal:
        from range_finder.conformal import evaluate_conformal_gate, upsert_lambda

        frame = walk_forward_evaluate(
            df, list(feat_cols),
            min_train=args.min_train, step=args.step,
        )
        gate = evaluate_conformal_gate(frame)
        print("\nCONFORMAL ADOPTION GATE")
        print(f"  lambda            : {gate['lambda']:.4f}")
        print(f"  holdout coverage  : {gate['coverage_before']:.1%} -> "
              f"{gate['coverage_after']:.1%} (nominal {gate['nominal']:.0%}, "
              f"n={gate['n_holdout']})")
        print(f"  improvement       : {gate['improvement']*100:+.1f}pp toward nominal")
        print(f"  verdict           : {'KEEP' if gate['keep'] else 'REJECT'} "
              f"(needs >= +3.0pp)")
        if gate["keep"] and args.persist:
            upsert_lambda(conn, args.ticker, args.spec, gate["lambda"],
                          n_obs=gate["n_calibrate"])
            print(f"  persisted to interval_calibration ({args.ticker}/{args.spec}).")
            print("  NOTE: production stays unchanged until "
                  "conformal.CONFORMAL_ENABLED is flipped to True.")
        elif args.persist:
            print("  --persist ignored: gate did not pass.")


if __name__ == "__main__":
    main()
