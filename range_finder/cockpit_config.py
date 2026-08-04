"""
Monday Cockpit configuration — every tunable threshold in one place.

House convention: constants live in Python modules (like phase1/config.py and
phase1/ticker_config.py), not TOML. The UI folds its widget values into a
CockpitConfig instance per render and the pure engine consumes only that
dataclass, so tests construct variants freely and no global is read at
decision time.
"""
from __future__ import annotations

from dataclasses import dataclass

# The Cockpit is a single-instrument decision surface by design.
COCKPIT_TICKER = "XSP"
# Weekly HAR spec the forecast is read from (XSP shares SPX's fit).
COCKPIT_MODEL_SPEC = "M3_extended"

# Defaults, mirrored into CockpitConfig below.
#
# VRP_MIN_RATIO was a HARD gate by decision (2026-07-11), logged to
# cockpit_verdicts so it could be audited later the way the 0DTE VRP banner
# was. That audit ran 2026-08-03 (148 walk-forward weeks, 2023-10 → 2026-07,
# M3_extended refit on the trailing TRAIN_WINDOW_YEARS before each week, plus
# the 16 weeks with a real Monday-frozen XSP straddle in em_snapshots). It
# failed on both counts:
#
#   1. WRONG MEASURING STICK. Every historical vol-premium number in this repo
#      (feature_builder.vix_implied_range, har_model's model_vs_vix) is a
#      VIX-derived range. The Cockpit gates on the ATM WEEKLY STRADDLE, which
#      measured 0.703× the VIX-derived range (VIX is a 30d variance-swap rate
#      carrying skew and term-structure contango; the straddle is 4-day ATM).
#      On the VIX scale a 1.10 gate passes 99.3% of weeks — a sanity check. On
#      the straddle scale the ratio's median is 1.115, so 1.10 landed exactly
#      at the median and behaved as a coin-flip filter. The straddle scale is
#      the CORRECT one to gate on (it is what actually gets sold) — do not
#      "fix" compute_vrp toward the VIX number; only the threshold was wrong.
#   2. NO EDGE. corr(ratio, condor no-touch) = −0.061. At the 1.10 gate,
#      passing weeks won 10.7% vs 17.2% for failing weeks (lift −6.5%); at
#      1.00 the lift was −9.8%. Only 1.20 turned positive (+6.0%, n=33). The
#      ratio DOES predict vol richness (corr with implied−realized = +0.376,
#      P(implied>realized) rising 50%→83% across quintiles) — it just does not
#      translate into condor survival, because a touch is a path/drift event
#      rather than a range-magnitude one. Same shape as the 0DTE VRP banner.
#
# So the ratio is still computed, still shown, still logged — but it warns
# instead of forcing SKIP. Set vrp_hard_gate=True to restore the old behavior.
VRP_MIN_RATIO = 1.10           # implied weekly range vs forecast; warn below
VRP_HARD_GATE = False          # True → thin VRP forces SKIP (pre-audit behavior)
K_EXPECTED_MOVE = 1.0          # short strikes at anchor ± k × EM (UI slider)
SNAP_TOLERANCE_PCT = 0.5       # GEX wall snap tolerance, % of anchor
COCKPIT_MIN_CREDIT_RATIO = 0.20  # condor credit/width floor (spread_levels' informational floor is 0.05)
EVENT_POLICY = "skip"          # tier-1 event inside Mon–Fri: "skip" | "widen"
EVENT_WIDEN_FACTOR = 1.25
TRIPWIRE_LOW = 0.5             # implied/forecast reconcile band (units tripwire)
TRIPWIRE_HIGH = 3.0
MIN_SAMPLE = 10                # behavioral gate: trades needed for a hard signal
BID_ASK_FLAG_PCT = 10.0        # per-leg bid-ask % of mid worth flagging
OI_FLOOR = 100                 # per-leg open-interest floor

# Leg-in entry guidance (user's actual workflow, 2026-07-11): the whole
# condor is sold at once only when spot sits at/near the SWEET SPOT — the
# midpoint between the two short strikes; otherwise one vertical is sold
# first and the other legged in later, but never past Wednesday.
SWEET_SPOT_TOL_EM = 0.25       # "near" = within ±this × EM of the midpoint
ENTRY_CUTOFF_WEEKDAY = 2       # 0=Mon … 2=Wed: last day to enter/complete


@dataclass(frozen=True)
class CockpitConfig:
    """Immutable threshold bundle consumed by cockpit_engine / preflight_engine."""
    vrp_min_ratio: float = VRP_MIN_RATIO
    vrp_hard_gate: bool = VRP_HARD_GATE
    k_expected_move: float = K_EXPECTED_MOVE
    snap_tolerance_pct: float = SNAP_TOLERANCE_PCT
    min_credit_ratio: float = COCKPIT_MIN_CREDIT_RATIO
    event_policy: str = EVENT_POLICY
    event_widen_factor: float = EVENT_WIDEN_FACTOR
    tripwire_low: float = TRIPWIRE_LOW
    tripwire_high: float = TRIPWIRE_HIGH
    min_sample: int = MIN_SAMPLE
    bid_ask_flag_pct: float = BID_ASK_FLAG_PCT
    oi_floor: int = OI_FLOOR
    wing_width: float | None = None  # None → first entry of the ticker's wing_widths
    sweet_spot_tol_em: float = SWEET_SPOT_TOL_EM
    entry_cutoff_weekday: int = ENTRY_CUTOFF_WEEKDAY
    # Per-contract, per-leg all-in fee (commission + regulatory). Public.com
    # is commission-free with pennies of regulatory fees, so the default 0.0
    # changes nothing — but when set, the cockpit surfaces the round-trip
    # fee drag against the condor's credit so thin-credit weeks are judged
    # net of costs instead of gross. Worst case is entry-only fees (expire
    # worthless: 4 legs); a traded exit doubles it.
    fees_per_contract: float = 0.0


# Legacy symmetric split — each side gets half the forecast range. Kept as the
# fallback when a forecast carries no side_share_q (a DB hiccup in the
# estimator degrades to pre-side-share behavior rather than erroring).
LEGACY_SIDE_SHARE = 0.5


def forecast_side_share(forecast: dict | None) -> float:
    """Per-side share of the forecast range, for band and strike placement.

    Lives here rather than in har_model so cockpit_engine and preflight_engine
    (both of which already import this module, neither of which imports
    statsmodels) can share one convention.

    The weekly HAR model forecasts how WIDE the week trades, not WHERE the
    open sits inside that width. Splitting the forecast half-up/half-down
    protects a strike only when the open lands mid-range, and it does not:
    the open concentrates near one extreme (arcsine law), so a trending week
    breaches a half-split strike while the realized range stays comfortably
    inside the forecast. har_model.estimate_side_share_quantile measures that
    directly — s_up = (H−O)/(H−L) per completed week — and its (1−beta)
    pooled quantile q is the per-side share that historically contained all
    but a beta tail. Callers place at ref × (1 ± u·q) instead of ref × (1 ± u/2).

    The Spread Finder, the TradingView export and scheduled_snapshot have
    threaded side_share_q through since it was introduced; the Cockpit and
    Pre-Flight were left on the hardcoded /2 (fixed 2026-08-03), which placed
    their strikes ~1.67× tighter than every other surface for the same week.
    Clamped to [0.5, 1.0]: below 0.5 would be tighter than the legacy split,
    above 1.0 would give one side more than the whole forecast range.
    """
    q = (forecast or {}).get("side_share_q")
    try:
        q = float(q)
    except (TypeError, ValueError):
        return LEGACY_SIDE_SHARE
    return min(max(q, LEGACY_SIDE_SHARE), 1.0)
