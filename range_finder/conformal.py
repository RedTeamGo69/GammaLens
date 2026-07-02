# =============================================================================
# conformal.py
# Split-conformal prediction-interval correction — OFF BY DEFAULT.
#
# The HAR forecast's 80% PI comes from Gaussian OLS theory on the log scale.
# If empirical coverage drifts from nominal (walkforward.summarize measures
# it), a single scale factor λ on the log-scale interval half-width restores
# calibration without touching the model:
#
#     score_i = |y_true_i - y_pred_i| / halfwidth_i      (walk-forward OOS)
#     λ       = empirical (1-α) quantile of scores over a trailing window
#
# λ > 1 widens (intervals were overconfident), λ < 1 narrows. Under perfect
# Gaussian calibration λ ≈ 1. This is distribution-free: it makes no
# assumption about WHY the residuals misbehave.
#
# Adoption is evidence-gated and ships disabled:
#   * `python -m range_finder.walkforward --conformal` computes λ, checks it
#     against the gate (coverage must move ≥ 3pp closer to nominal on
#     held-out data without hurting accuracy), and with `--persist` writes
#     the λ row to interval_calibration.
#   * Production reads it ONLY when CONFORMAL_ENABLED is True AND a λ row
#     exists for (ticker, model_name) — `maybe_apply_conformal` is a no-op
#     otherwise, so wiring it at forecast call sites is free.
#
# 2026-07 status: walk-forward coverage of the weekly OLS specs is ~82.6%
# two-sided vs 80% nominal — already calibrated, λ would be ~1. The module
# exists as the corrective for when that stops being true (and for the 0DTE
# model, whose empirical coverage is unknown until forecast_log_daily
# accumulates).
# =============================================================================

import logging
import math
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)

# Master switch — flip only after the adoption gate passes and a λ row is
# persisted. With this False, maybe_apply_conformal never alters a forecast.
CONFORMAL_ENABLED = False

# Adoption gate: applying λ on held-out data must move two-sided coverage at
# least this much closer to nominal...
_MIN_COVERAGE_IMPROVEMENT = 0.03
# ...without degrading point accuracy (λ doesn't touch y_pred, so this is a
# tripwire, not an expected failure mode).

_DEFAULT_WINDOW = 104


# =============================================================================
# λ ESTIMATION — pure function on a walk-forward output frame
# =============================================================================

def conformal_lambda(frame: pd.DataFrame, alpha: float = 0.20,
                     window: int = _DEFAULT_WINDOW) -> float:
    """Split-conformal scale factor from walk-forward OOS predictions.

    ``frame`` is walkforward.walk_forward_evaluate output (log-scale
    y_true/y_pred/obs_ci bounds). Uses the trailing ``window`` observations.
    """
    work = frame.tail(window)
    halfwidth = (work["obs_ci_upper"] - work["obs_ci_lower"]) / 2.0
    valid = halfwidth > 0
    if valid.sum() < 10:
        raise ValueError(f"conformal_lambda: only {int(valid.sum())} usable "
                         "observations (need >= 10)")
    scores = ((work["y_true"] - work["y_pred"]).abs() / halfwidth)[valid]
    # Finite-sample conformal quantile: ceil((n+1)(1-alpha))/n -th order stat.
    n = len(scores)
    q = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(scores.quantile(q, interpolation="higher"))


def apply_conformal(forecast: dict, lam: float) -> dict:
    """Return a NEW forecast dict with the PI half-width scaled by λ.

    Operates on the log scale (where the OLS interval is symmetric):
        center    = log(point_pct)
        halfwidth = (log(upper_pct) - log(lower_pct)) / 2
        new bounds = exp(center ± λ·halfwidth)

    Price levels are recomputed preserving the symmetric-band convention
    (pi_lower_px derives from the UPPER range bound — see the note at
    har_model.forecast_next_week). The point forecast never changes.
    """
    out = dict(forecast)
    point = float(forecast["point_pct"])
    lower = float(forecast["lower_pct"])
    upper = float(forecast["upper_pct"])
    if point <= 0 or lower <= 0 or upper <= 0:
        log.warning("apply_conformal: non-positive bounds — returning forecast unchanged")
        out["conformal_lambda"] = None
        return out

    center = math.log(point)
    halfwidth = (math.log(upper) - math.log(lower)) / 2.0

    new_upper = math.exp(center + lam * halfwidth)
    new_lower = math.exp(center - lam * halfwidth)

    spx_ref = float(forecast.get("spx_ref_close") or 0.0)
    half_upper = new_upper / 2.0

    out["lower_pct"] = round(new_lower, 4)
    out["upper_pct"] = round(new_upper, 4)
    if spx_ref:
        out["pi_upper_px"] = round(spx_ref * (1 + half_upper), 2)
        out["pi_lower_px"] = round(spx_ref * (1 - half_upper), 2)
    out["conformal_lambda"] = round(float(lam), 4)
    return out


# =============================================================================
# PERSISTENCE — interval_calibration table
# =============================================================================

def upsert_lambda(conn, ticker: str, model_name: str, lam: float,
                  n_obs: int, method: str = "split_conformal") -> None:
    """Persist a gate-approved λ for (ticker, model_name)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO interval_calibration
            (ticker, model_name, lambda, n_obs, method, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, model_name) DO UPDATE SET
            lambda     = excluded.lambda,
            n_obs      = excluded.n_obs,
            method     = excluded.method,
            updated_at = excluded.updated_at
    """, (ticker, model_name, float(lam), int(n_obs), method, now))
    conn.commit()
    log.info(f"interval_calibration: {ticker}/{model_name} lambda={lam:.4f} "
             f"(n={n_obs}, {method})")


def load_lambda(conn, ticker: str, model_name: str) -> float | None:
    """Read the persisted λ for (ticker, model_name), or None."""
    try:
        row = conn.execute(
            "SELECT lambda FROM interval_calibration "
            "WHERE ticker = ? AND model_name = ?",
            (ticker, model_name),
        ).fetchone()
    except Exception as e:
        log.warning(f"interval_calibration read failed: {e}")
        return None
    return float(row[0]) if row and row[0] is not None else None


def maybe_apply_conformal(forecast: dict, conn, ticker: str,
                          model_name: str) -> dict:
    """Production hook: apply λ only if enabled AND persisted. No-op (returns
    the input dict unchanged, same object) otherwise — safe to wire at every
    forecast call site."""
    if not CONFORMAL_ENABLED:
        return forecast
    lam = load_lambda(conn, ticker, model_name)
    if lam is None:
        return forecast
    log.info(f"Applying conformal lambda={lam:.4f} to {ticker}/{model_name} forecast")
    return apply_conformal(forecast, lam)


# =============================================================================
# ADOPTION GATE — evaluated on walk-forward output, split calibrate/holdout
# =============================================================================

def evaluate_conformal_gate(frame: pd.DataFrame, alpha: float = 0.20,
                            window: int = _DEFAULT_WINDOW) -> dict:
    """Fit λ on the first part of a walk-forward frame, verify on the rest.

    Returns a dict with λ, before/after held-out coverage, and `keep`:
    coverage must move ≥ _MIN_COVERAGE_IMPROVEMENT closer to nominal.
    """
    if len(frame) < 40:
        raise ValueError(f"evaluate_conformal_gate: {len(frame)} OOS obs "
                         "(need >= 40 for a calibrate/holdout split)")

    split = int(len(frame) * 0.6)
    calib, hold = frame.iloc[:split], frame.iloc[split:]

    lam = conformal_lambda(calib, alpha=alpha, window=window)
    nominal = 1 - alpha

    inside_before = ((hold["y_true"] >= hold["obs_ci_lower"])
                     & (hold["y_true"] <= hold["obs_ci_upper"]))
    cov_before = float(inside_before.mean())

    center = (hold["obs_ci_upper"] + hold["obs_ci_lower"]) / 2.0
    halfwidth = (hold["obs_ci_upper"] - hold["obs_ci_lower"]) / 2.0
    inside_after = ((hold["y_true"] >= center - lam * halfwidth)
                    & (hold["y_true"] <= center + lam * halfwidth))
    cov_after = float(inside_after.mean())

    improvement = abs(cov_before - nominal) - abs(cov_after - nominal)
    return {
        "lambda": lam,
        "n_calibrate": split,
        "n_holdout": len(hold),
        "coverage_before": cov_before,
        "coverage_after": cov_after,
        "nominal": nominal,
        "improvement": improvement,
        "keep": improvement >= _MIN_COVERAGE_IMPROVEMENT,
    }
