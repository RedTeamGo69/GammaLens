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

# Defaults, mirrored into CockpitConfig below. VRP_MIN_RATIO is a hard gate
# by decision (2026-07-11): thin VRP → SKIP, with every verdict + inputs
# logged to cockpit_verdicts so the gate itself can be audited later the way
# the 0DTE VRP banner was.
VRP_MIN_RATIO = 1.10           # implied weekly range must be ≥ 1.10× forecast
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
