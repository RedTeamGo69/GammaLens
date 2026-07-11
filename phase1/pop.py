"""
Probability-of-profit helpers built on short-strike deltas.

POP ≈ 1 − |Δ_short_call| − |Δ_short_put| — the standard delta approximation
for a credit condor (each short delta ≈ P(expires ITM)). Drop the missing
side's term for a single vertical.

Display-only by design: vendor greeks can be stale/zero and the BS fallback
inherits mid-IV noise, so POP never gates a verdict anywhere — callers show
the value with its source ("vendor" / "bs_iv") or "—" when unavailable.
"""
from __future__ import annotations

import math

from phase1.config import DEFAULT_RISK_FREE_RATE


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = 0.0,
    option_type: str = "call",
) -> float | None:
    """Black-Scholes delta; None when any input can't support the formula."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return None
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years) / (
        iv * math.sqrt(t_years)
    )
    if option_type == "call":
        return math.exp(-q * t_years) * _norm_cdf(d1)
    return -math.exp(-q * t_years) * _norm_cdf(-d1)


def resolve_short_delta(
    row: dict,
    spot: float,
    t_years: float,
    option_type: str,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = 0.0,
) -> tuple[float | None, str]:
    """Best-available delta for one chain row: vendor first, BS-from-mid-IV next.

    Returns (delta, source) where source is "vendor" | "bs_iv" | "unavailable".
    A vendor delta of exactly 0.0 is treated as missing — Tradier zero-fills
    greeks it doesn't have, and a genuinely 0-delta option is untradeable
    anyway.
    """
    vendor = float(row.get("vendorDelta", 0) or 0)
    if vendor != 0.0:
        return vendor, "vendor"

    iv = float(row.get("impliedVolatility", 0) or 0)
    strike = float(row.get("strike", 0) or 0)
    bs = bs_delta(spot, strike, t_years, iv, r=r, q=q, option_type=option_type)
    if bs is not None:
        return bs, "bs_iv"
    return None, "unavailable"


def condor_pop(
    delta_short_call: float | None,
    delta_short_put: float | None,
) -> float | None:
    """POP ≈ 1 − |Δsc| − |Δsp|, clamped to [0, 1]; None when either is missing."""
    if delta_short_call is None or delta_short_put is None:
        return None
    pop = 1.0 - abs(delta_short_call) - abs(delta_short_put)
    return max(0.0, min(1.0, pop))
