"""
har_model_daily.py
Daily-cadence HAR model for the 0DTE spread finder.

Re-uses the cadence-agnostic OLS/forecast helpers from ``har_model.py``
(``fit_model``, ``fit_model_wls``, ``time_series_split``, ``evaluate_oos``,
``forecast_next_week``) and provides:

* ``MODEL_SPECS_DAILY`` — three specs that mirror M1/M2/M3 from the weekly
  model but use the daily HAR lags + VIX1D where appropriate.
* ``forecast_next_session`` — thin wrapper that calls ``forecast_next_week``
  and overrides ``vix_implied_pct`` with the VIX1D-implied range so the
  resulting dict makes sense at 0DTE call sites.
* ``run_daily_pipeline`` — end-to-end fit-all-specs / save-best entry
  point used by ``bootstrap_range_finder.py`` step 10.

Only SPX is fit. XSP reuses the SPX-trained model at inference (XSP = SPX/10
and they share ^VIX1D, so the range_pct forecast applies unchanged).
"""
from __future__ import annotations

import logging

from range_finder.har_model import (
    fit_model,
    fit_model_wls,
    evaluate_oos,
    time_series_split,
    forecast_next_week,
    feature_has_enough_data,
    compare_models,
    PI_ALPHA,
)
from range_finder.model_persistence import save_model, load_model
from range_finder.feature_builder_daily import get_daily_features

log = logging.getLogger(__name__)


# =============================================================================
# DAILY MODEL SPECS
# =============================================================================

HAR_CORE_DAILY = ["har_d1_daily", "har_w_daily", "har_m_daily"]

MODEL_SPECS_DAILY = {
    # M1 trains on the full 4-yr daily history; doesn't need VIX1D so it's a
    # safe fallback when VIX1D rows are absent (pre-2022) or sparse.
    "M1_daily_baseline": HAR_CORE_DAILY,

    # M2 adds VIX (full history) + VIX1D + VIX1D-implied range. Requires
    # enough VIX1D history (~12 months) to clear feature_has_enough_data.
    "M2_daily_vix": HAR_CORE_DAILY + [
        "vix_close",
        "vix1d_close",
        "vix1d_implied_range",
    ],

    # M3 adds short-term HV, return lags, and day-of event flags. Needs the
    # most history; auto-skips if event-flag columns are too sparse.
    "M3_daily_extended": HAR_CORE_DAILY + [
        "vix_close",
        "vix1d_close",
        "vix1d_implied_range",
        "hv5",
        "spx_return_lag1",
        "abs_return_lag1",
        "has_fomc_today",
        "has_cpi_today",
        "has_nfp_today",
    ],
}


# =============================================================================
# FORECAST WRAPPER
# =============================================================================

def forecast_next_session(result, feature_row, feature_cols, spx_ref,
                          alpha: float = PI_ALPHA) -> dict:
    """Forecast next session's range using a daily HAR fit.

    Math is identical to ``forecast_next_week`` (the function only cares
    about the feature vector and the trained OLS result, not the cadence).
    We override ``vix_implied_pct`` and ``model_vs_vix`` so the dict
    compares the model against VIX1D rather than weekly VIX.
    """
    forecast = forecast_next_week(
        result, feature_row, feature_cols, spx_ref, alpha=alpha,
    )

    # If the feature row carries a VIX1D-implied range, surface it as the
    # IV-side comparison. Falls back to whatever forecast_next_week wrote
    # (which would be weekly vix_implied_range if it was in the row).
    vix1d_implied = feature_row.get("vix1d_implied_range") \
        if hasattr(feature_row, "get") else None
    try:
        vix1d_implied = float(vix1d_implied) if vix1d_implied is not None else 0.0
    except (TypeError, ValueError):
        vix1d_implied = 0.0

    if vix1d_implied:
        forecast["vix_implied_pct"] = round(vix1d_implied, 4)
        forecast["model_vs_vix"]    = round(forecast["point_pct"] - vix1d_implied, 4)

    return forecast


# =============================================================================
# PIPELINE
# =============================================================================

def run_daily_pipeline(conn, preferred_model: str = "M2_daily_vix",
                        exclude_covid: bool = True) -> dict:
    """End-to-end daily HAR pipeline.

    Loads daily features (SPX only), fits every spec that has enough data,
    compares OOS metrics, saves the preferred (or best available) fit to
    ``saved_models`` under ticker ``"SPX"``. XSP loads the same fit at
    inference time.

    Returns ``{"preferred": spec_name, "metrics": ..., "results": ...}``.
    """
    df = get_daily_features(conn, ticker="SPX", exclude_covid=exclude_covid)
    if df.empty:
        raise RuntimeError("daily_model_features is empty for SPX — "
                           "run bootstrap step 9 first.")

    log.info(f"Loaded {len(df)} daily feature rows for modeling (SPX)")

    all_metrics: dict[str, dict] = {}
    all_results: dict[str, tuple] = {}

    for spec_name, feat_cols in MODEL_SPECS_DAILY.items():
        available = [c for c in feat_cols if feature_has_enough_data(df, c)]
        if len(available) < len(feat_cols):
            missing = set(feat_cols) - set(available)
            log.warning(f"{spec_name}: dropping missing features {sorted(missing)}")
        if len(available) < 2:
            log.warning(f"{spec_name}: <2 usable features — skipping")
            continue
        try:
            X_train, X_test, y_train, y_test = time_series_split(
                df, feature_cols=available,
            )
            result = fit_model(X_train, y_train, model_name=spec_name)
            metrics = evaluate_oos(result, X_test, y_test, model_name=spec_name)
            all_metrics[spec_name] = metrics
            all_results[spec_name] = (result, available)
        except Exception as e:
            log.error(f"Failed to fit {spec_name}: {e}")

    if not all_results:
        raise RuntimeError("No daily specs fit successfully — "
                           "check feature coverage in daily_model_features.")

    compare_models(all_metrics)

    if preferred_model not in all_results:
        fallback = list(all_results.keys())[0]
        log.warning(f"Preferred daily model '{preferred_model}' unavailable — "
                    f"falling back to {fallback}.")
        preferred_model = fallback

    # Save EVERY spec that fit, not just the preferred one. The 0DTE
    # finder's model dropdown offers all of MODEL_SPECS_DAILY, but only
    # the preferred spec used to be persisted — selecting M1/M3 in the
    # UI then failed with "No saved model found for SPX/<spec>" until a
    # manual bootstrap. Mirrors the weekly cron, which saves all specs.
    for spec_name, (result, features) in all_results.items():
        save_model(
            result,
            features,
            spec_name,
            all_metrics.get(spec_name, {}),
            conn=conn,
            ticker="SPX",
        )
    log.info(f"Saved {len(all_results)} daily specs: {sorted(all_results.keys())}")

    return {
        "preferred": preferred_model,
        "metrics":   all_metrics,
        "results":   all_results,
    }
