from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# Market calendars
CASH_CALENDAR = "NYSE"
OPTIONS_CALENDAR = "CBOE_Index_Options"

# Expiring cash-settled index roots do not share one final-trading clock.
# Offsets are relative to the holiday/early-close-aware 4:15 index-options
# calendar session. AM roots trade on the prior session; PM roots trade on
# the expiration session. SPX's expiring AM series currently includes the
# Cboe curb through 5:00 ET (45 minutes beyond the calendar's regular close).
INDEX_OPTION_EXPIRATION_RULES = {
    "SPX":  {"settlement": "AM", "session": "previous",   "close_offset_minutes": 45},
    "SPXW": {"settlement": "PM", "session": "expiration", "close_offset_minutes": -15},
    "XSP":  {"settlement": "PM", "session": "expiration", "close_offset_minutes": -15},
    "NDX":  {"settlement": "AM", "session": "previous",   "close_offset_minutes": 0},
    "NDXP": {"settlement": "PM", "session": "expiration", "close_offset_minutes": -15},
    "RUT":  {"settlement": "AM", "session": "previous",   "close_offset_minutes": 0},
    "RUTW": {"settlement": "PM", "session": "expiration", "close_offset_minutes": -15},
}

# Cash indices the GEX quick-picks show and the cron captures beyond the
# forward-study roster. They stay out of TICKER_CONFIG because its bytes are
# forward-study hashed; the cron registers them as indices instead (daily
# cadence, no single-name earnings) and the app derives them from the chain.
EXTRA_CAPTURE_INDICES = ("RUT",)

# Wider range for computation (all_options used in sweep/profile/scenarios)
# Captures deep OTM put gamma relevant during tail events
COMPUTATION_RANGE_PCT = 0.08

# Zero-gamma sweep config
ZG_SWEEP_RANGE_PCT = 0.05
ZG_SWEEP_STEP = 5.0
ZG_FINE_STEP = 0.5

# Dynamic sweep: scale range by ATM IV when available
ZG_SWEEP_DYNAMIC = True
ZG_SWEEP_MIN_RANGE_PCT = 0.03
ZG_SWEEP_MAX_RANGE_PCT = 0.12
ZG_SWEEP_IV_SCALE = 1.5

# Time floor: 1 minute in years
T_FLOOR = 1 / (365.25 * 24 * 60)

# Default risk-free rate used when no live rate is passed
DEFAULT_RISK_FREE_RATE = 0.045

# Trading-time T: count only market hours for time-to-expiry
# This significantly improves gamma accuracy for 0DTE/1DTE options
USE_TRADING_TIME = True
TRADING_HOURS_PER_YEAR = 252 * 6.5  # 252 trading days × 6.5 hours per session

# Fetching
MAX_WORKERS = 4
CHAIN_RETRIES = 2
CHAIN_RETRY_SLEEP = 0.6

# Parity / quote quality settings
MAX_PARITY_SPREAD = 2.0
PARITY_NEAR_SPOT_CANDIDATES = 15
PARITY_FINAL_STRIKES = 5
PARITY_RELATIVE_BAND = 0.015
PARITY_HARD_LOW_MULTIPLIER = 0.90
PARITY_HARD_HIGH_MULTIPLIER = 1.10
MIN_PARITY_STRIKES = 2

# Parity estimator method
PARITY_METHOD = "weighted_median"   # options: "median", "weighted_median"

# Weighting controls for parity
PARITY_WEIGHT_EPS = 1e-6
PARITY_SPREAD_WEIGHT_POWER = 1.0
PARITY_DISTANCE_SIGMA_PCT = 0.01

# Synthetic IV bounds and fit acceptance
SYNTH_IV_MIN = 0.01
SYNTH_IV_MAX = 3.00
SYNTH_FIT_MAX_REL_ERROR = 0.08
# Root-selection anchor for the gamma→IV inverter. BS gamma(σ) is unimodal,
# so any sub-peak target gamma has TWO σ roots that both reproduce the
# vendor gamma at current spot. The inverter picks the root closest to this
# reference in log space — a tenor-typical SPX IV — instead of blindly
# taking the smaller root, which lands on the wrong branch for near-ATM
# strikes (where the gamma peak sits at very low σ).
SYNTH_IV_REFERENCE = 0.20

# Hybrid IV mode
HYBRID_IV_MODE = True

# Liquidity / support scoring
SUPPORT_MAX_SPREAD_FOR_SCORE = 10.0
SUPPORT_HIGH_THRESHOLD = 75.0
SUPPORT_MODERATE_THRESHOLD = 50.0

# Strike support scoring weights (must sum to 1.0)
STRIKE_SUPPORT_W_OI = 0.30
STRIKE_SUPPORT_W_BREADTH = 0.20
STRIKE_SUPPORT_W_CONTRACTS = 0.20
STRIKE_SUPPORT_W_DIRECT_IV = 0.15
STRIKE_SUPPORT_W_SPREAD = 0.10
STRIKE_SUPPORT_W_TWO_SIDED = 0.05

# Expiration support scoring weights (must sum to 1.0)
EXP_SUPPORT_W_OI = 0.35
EXP_SUPPORT_W_CONTRACTS = 0.25
EXP_SUPPORT_W_STRIKE_BREADTH = 0.15
EXP_SUPPORT_W_DIRECT_IV = 0.15
EXP_SUPPORT_W_SPREAD = 0.10

# Stale-data / quote-quality defenses
STALE_FRESHNESS_HIGH_THRESHOLD = 85.0
STALE_FRESHNESS_MODERATE_THRESHOLD = 65.0
STALE_NO_TWO_SIDED_RATIO_WARN = 0.25
STALE_WIDE_SPREAD_RATIO_WARN = 0.25
STALE_CROSSED_RATIO_WARN = 0.05
STALE_MIN_HARD_FILTER_PASS = 3
