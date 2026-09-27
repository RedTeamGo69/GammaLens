"""Spread Finder judgement rules, kept pure so tests can reach them.

They used to live inline in ``ui_spread_finder._render_spread_finder_tab``
— the VIX regime-shift breaker sat dead for weeks there because its input
fetch raised a swallowed NameError and no test touched the rule.
"""
import math
from dataclasses import dataclass

# Live/trailing VIX ratio that trips the breaker: roughly a 2σ move on the
# weekly VIX change distribution. At EXTREME the forecast is badly stale.
REGIME_ELEVATED_RATIO = 1.5
REGIME_EXTREME_RATIO = 2.0

# Reference-price deviation from live spot. Beyond RESET no weekly market
# move explains it (data corruption: an SPX ref of 100 against a ~7,400
# spot walked the whole ladder to the chain edge); beyond WARN keep it but
# make the user look.
REF_RESET_DEV = 0.25
REF_WARN_DEV = 0.12


@dataclass(frozen=True)
class RegimeShift:
    severity: str          # "elevated" | "extreme"
    live_vix: float
    trailing_vix: float
    ratio: float


def _positive(value) -> "float | None":
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def detect_regime_shift(live_vix, trailing_vix) -> "RegimeShift | None":
    """A VIX regime shift since the (one-week-lagged) HAR features, or None.

    HAR features lag a week, so after an overnight IV spike the forecast is
    anchored to the pre-spike regime and short strikes come out too narrow.
    ``live_vix`` must be the genuinely fetched value, never a display
    default: comparing an invented number would miss a real spike or invent
    a phantom one. Without both values there is no judgement.
    """
    live, trailing = _positive(live_vix), _positive(trailing_vix)
    if live is None or trailing is None:
        return None
    ratio = live / trailing
    if ratio < REGIME_ELEVATED_RATIO:
        return None
    severity = "extreme" if ratio >= REGIME_EXTREME_RATIO else "elevated"
    return RegimeShift(severity, live, trailing, ratio)


@dataclass(frozen=True)
class ReferenceCheck:
    status: str                    # "ok" | "warn" | "reset"
    deviation: "float | None"      # |ref/spot - 1|, None when ref unusable


def check_reference(ref, spot) -> ReferenceCheck:
    """Judge a (sticky, user-editable) reference price against live spot."""
    spot_v = _positive(spot)
    if spot_v is None:
        return ReferenceCheck("ok", None)      # nothing to judge against
    ref_v = _positive(ref)
    if ref_v is None:
        return ReferenceCheck("reset", None)
    dev = abs(ref_v / spot_v - 1.0)
    if dev > REF_RESET_DEV:
        return ReferenceCheck("reset", dev)
    if dev > REF_WARN_DEV:
        return ReferenceCheck("warn", dev)
    return ReferenceCheck("ok", dev)
