from __future__ import annotations

import numpy as np
from scipy.stats import norm

from phase1.config import (
    SYNTH_IV_MIN,
    SYNTH_IV_MAX,
    SYNTH_FIT_MAX_REL_ERROR,
    SYNTH_IV_REFERENCE,
    HYBRID_IV_MODE,
)

# Economic lower bound on accepted synthetic IV. BS gamma(sigma) is unimodal
# so the inverter has two roots for any sub-peak target gamma: a small-sigma
# root (economically meaningful for typical options) and a large-sigma root
# (nonsensical). The inverter prefers the smaller root, which is correct
# most of the time — but for a deep-OTM strike the "small" root can itself
# be nonsensically tiny (e.g. 2% annualized IV on a 5% OTM weekly put).
# Reject anything below this floor and fall through to the no-solution
# path rather than accepting an IV that would fabricate a spurious gamma
# contribution. The 0.05 floor is comfortably below any plausible SPX IV
# (the historic VIX minimum is ~9%) while still generous enough that a
# legitimate low-vol weekly doesn't get filtered.
SYNTH_IV_ECONOMIC_FLOOR = 0.05


def bs_gamma(S, K, T, r, sigma):
    """
    Scalar Black-Scholes gamma.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _bs_gamma_sigma_grid(S, K, T, r, sigmas):
    """
    Vectorized BS gamma across an array of sigmas (fixed S, K, T, r).

    Used by infer_iv_from_gamma's grid scan — the scalar bs_gamma() loop
    made 200 scipy norm.pdf calls per synthetic-IV option, which dominated
    the engine runtime on chains with many missing-IV contracts. The
    standard-normal pdf is inlined (exp(-d1²/2)/√(2π)) so the whole grid
    is one numpy pass.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigmas**2) * T) / (sigmas * sqrt_T)
    pdf = np.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI
    return pdf / (S * sigmas * sqrt_T)


def infer_iv_from_gamma(target_gamma, S, K, T, r, low=SYNTH_IV_MIN, high=SYNTH_IV_MAX, steps=70):
    """
    Infer a synthetic IV that matches vendor gamma at current spot.

    BS gamma(sigma) is unimodal (rises to a peak, then falls), so naive bisection
    can fail. We locate the peak, then use Brent's method on the sub-interval(s)
    that bracket the target.

    Root selection: both roots reproduce the vendor gamma AT SPOT, but they
    imply very different gamma profiles AWAY from spot — which is what the
    zero-gamma sweep and scenario math actually consume. The old rule
    ("prefer the lower-sigma root") is right for deep-OTM strikes (whose
    gamma peak sits at high sigma) but lands on the wrong branch for
    near-ATM strikes, where the peak sigma is tiny and the economically
    real IV is the HIGHER root. We instead pick the root closest to
    SYNTH_IV_REFERENCE in log space: near-ATM ambiguity resolves to the
    plausible ~15-30% root, deep-OTM noise roots (2-4% or 150%+) lose to
    whichever is nearer the reference, and the economic floor in
    fit_synthetic_iv still rejects the truly absurd survivors.

    Returns:
        inferred sigma (float) or 0.0 if no reasonable solution.
    """
    from scipy.optimize import brentq

    if target_gamma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0

    def gamma_residual(sigma):
        return bs_gamma(S, K, T, r, sigma) - target_gamma

    # Dense grid to find the peak of gamma(sigma) and bracket solutions
    n_grid = 200
    grid = np.linspace(low, high, n_grid)
    gammas = _bs_gamma_sigma_grid(S, K, T, r, grid)

    peak_idx = int(np.argmax(gammas))
    peak_gamma = float(gammas[peak_idx])

    # Target is above the peak — no solution exists
    if target_gamma > peak_gamma * 1.02:
        return 0.0

    # Try left side of peak first (lower sigma — more financially meaningful)
    solutions = []
    for seg_start, seg_end in [(0, peak_idx), (peak_idx, n_grid - 1)]:
        if seg_end <= seg_start:
            continue
        seg_gammas = gammas[seg_start:seg_end + 1]
        seg_sigmas = grid[seg_start:seg_end + 1]

        # Find sub-intervals where residual changes sign
        residuals = seg_gammas - target_gamma
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                try:
                    sol = brentq(gamma_residual, float(seg_sigmas[i]), float(seg_sigmas[i + 1]),
                                 xtol=1e-8, rtol=1e-8, maxiter=100)
                    solutions.append(sol)
                except ValueError:
                    continue
            elif abs(residuals[i]) < 1e-12:
                solutions.append(float(seg_sigmas[i]))

    if not solutions:
        # Grid fallback for near-peak targets
        idx = int(np.argmin(np.abs(gammas - target_gamma)))
        best_sigma = float(grid[idx])
        best_gamma = float(gammas[idx])
        if best_gamma > 0 and abs(best_gamma - target_gamma) / target_gamma < 0.08:
            return best_sigma
        return 0.0

    # Pick the root closest to the reference IV in log space (see docstring)
    return float(min(solutions, key=lambda s: abs(np.log(s / SYNTH_IV_REFERENCE))))


def fit_synthetic_iv(target_gamma, S, K, T, r):
    """
    Fit synthetic IV from vendor gamma and score the fit.

    Returns a dict:
    {
        "accepted": bool,
        "iv": float,
        "target_gamma": float,
        "fitted_gamma": float,
        "rel_error": float | None,
        "reason": str,
    }
    """
    if target_gamma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return {
            "accepted": False,
            "iv": 0.0,
            "target_gamma": target_gamma,
            "fitted_gamma": 0.0,
            "rel_error": None,
            "reason": "invalid_inputs",
        }

    iv = infer_iv_from_gamma(target_gamma, S, K, T, r)
    if iv <= 0:
        return {
            "accepted": False,
            "iv": 0.0,
            "target_gamma": target_gamma,
            "fitted_gamma": 0.0,
            "rel_error": None,
            "reason": "no_solution",
        }

    # Economic-sanity floor: reject implausibly low IVs even when the
    # numeric fit is tight. BS gamma(σ) is small both at σ→0 AND at σ
    # well past the peak, so the optimizer can land on a "tight-fit"
    # sigma of 0.02-0.04 for deep OTM strikes when the vendor gamma
    # itself is near zero. That's not a vol estimate — it's numerical
    # noise. Kicking those back to the no-solution path avoids feeding
    # nonsense IVs into stats.call_iv / stats.put_iv and contaminating
    # the ATM IV average.
    if iv < SYNTH_IV_ECONOMIC_FLOOR:
        return {
            "accepted": False,
            "iv": float(iv),
            "target_gamma": float(target_gamma),
            "fitted_gamma": float(bs_gamma(S, K, T, r, iv)),
            "rel_error": None,
            "reason": "below_economic_floor",
        }

    fitted_gamma = float(bs_gamma(S, K, T, r, iv))
    rel_error = float(abs(fitted_gamma - target_gamma) / max(target_gamma, 1e-12))
    accepted = bool(rel_error <= SYNTH_FIT_MAX_REL_ERROR)

    return {
        "accepted": accepted,
        "iv": float(iv),
        "target_gamma": float(target_gamma),
        "fitted_gamma": fitted_gamma,
        "rel_error": rel_error,
        "reason": "accepted" if accepted else "fit_error_too_high",
    }


def prepare_option_for_model(opt, sign, T, spot, r):
    """
    Rich evaluation path for model input selection.

    Returns a dict:
    {
      "accepted": bool,
      "reason": str,
      "normalized": dict | None,
      "synthetic_fit_rel_error": float | None,
      "synthetic_target_gamma": float | None,
      "synthetic_fitted_gamma": float | None,
    }
    """
    K = opt["strike"]
    oi = opt["openInterest"]
    iv = opt.get("impliedVolatility", 0.0) or 0.0
    vendor_gamma = opt.get("vendorGamma", 0.0) or 0.0

    if iv > 0:
        gamma_now = bs_gamma(spot, K, T, r, iv)
        return {
            "accepted": True,
            "reason": "direct_iv",
            "synthetic_fit_rel_error": None,
            "synthetic_target_gamma": None,
            "synthetic_fitted_gamma": None,
            "normalized": {
                "strike": K,
                "oi": oi,
                "iv": iv,
                "sign": sign,
                "T": T,
                "gamma_now": gamma_now,
                "iv_source": "direct_iv",
                "synthetic_fit_rel_error": None,
            },
        }

    if HYBRID_IV_MODE and vendor_gamma > 0:
        fit = fit_synthetic_iv(vendor_gamma, spot, K, T, r)
        if fit["accepted"]:
            return {
                "accepted": True,
                "reason": "synthetic_iv",
                "synthetic_fit_rel_error": fit["rel_error"],
                "synthetic_target_gamma": fit["target_gamma"],
                "synthetic_fitted_gamma": fit["fitted_gamma"],
                "normalized": {
                    "strike": K,
                    "oi": oi,
                    "iv": fit["iv"],
                    "sign": sign,
                    "T": T,
                    "gamma_now": fit["fitted_gamma"],
                    "iv_source": "synthetic_iv",
                    "synthetic_fit_rel_error": fit["rel_error"],
                },
            }

        return {
            "accepted": False,
            "reason": f"synthetic_{fit['reason']}",
            "synthetic_fit_rel_error": fit["rel_error"],
            "synthetic_target_gamma": fit["target_gamma"],
            "synthetic_fitted_gamma": fit["fitted_gamma"],
            "normalized": None,
        }

    return {
        "accepted": False,
        "reason": "no_model_input",
        "synthetic_fit_rel_error": None,
        "synthetic_target_gamma": None,
        "synthetic_fitted_gamma": None,
        "normalized": None,
    }


def normalize_option_for_model(opt, sign, T, spot, r):
    """
    Backward-compatible wrapper.
    """
    result = prepare_option_for_model(opt, sign, T, spot, r)
    return result["normalized"]
