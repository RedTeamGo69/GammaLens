from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from .history_policy import TRADIER_POLICY_VERSION

LEGACY_UNIVERSE = ("SPX", "SPY", "AAPL", "AMD")
UNIVERSE = (*LEGACY_UNIVERSE, "META", "TSLA", "HOOD")
MODELS = ("M1_baseline", "M2_vix", "M3_extended", "M4_full")
COHORT = "prospective_opening_week"
DEFAULT_STUDY_ID = "spread-finder-weekly-v3-open"
SCORER_VERSION = "weekly-close-path-v3-session-open"
DATA_READY_MINUTES = 20
RECONCILE_DAYS = 14
MAX_CAPTURE_ATTEMPTS = 3


def study_universe(config):
    """A new registration must not silently expand an older study's roster."""
    universe = tuple(config.get('universe', LEGACY_UNIVERSE))
    if not universe or len(set(universe)) != len(universe) or not set(universe) <= set(UNIVERSE):
        raise ValueError('Invalid registered study universe')
    return universe


def utcnow():
    return datetime.now(timezone.utc)


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(json_text(value).encode()).hexdigest()


def model_methodology_id(methodology_id, model, feature_columns):
    """Cumulative comparison identity, deliberately independent of weekly fits.

    Training rows, dates, coefficients and covariance belong to the immutable
    input/fit snapshots. Including them here would split every week into its
    own scoreboard. Actual feature subsets and rule changes must still split.
    """
    return digest({"methodology": methodology_id, "model": model,
                   "feature_columns": list(feature_columns)})


def methodology():
    """Content identity excludes unrelated UI/research work and fit dates."""
    root = Path(__file__).resolve().parents[2]
    paths = ["har_model.py", "feature_builder.py", "spread_levels.py",
             "gex_policy.py", "recommendations.py", "trading_week.py",
             "conformal.py", "event_calendars.py", "forward_test/capture.py",
             "forward_test/provider.py", "forward_test/config.py",
             "forward_test/history_policy.py"]
    hashes = {p: sha256((root / "range_finder" / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in paths}
    for p in ("phase1/ticker_config.py", "phase1/quote_filters.py"):
        hashes[p] = sha256((root / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    config = {"protocol": 1, "universe": UNIVERSE, "models": MODELS,
              "capture_minutes_after_open": [0, 45], "tracking_policy": "first_session_open",
              "gex_enabled": False,
              "conformal_enabled": False, "source_hashes": hashes,
              "history_policy": TRADIER_POLICY_VERSION,
              "versions": {p: version(p) for p in ("numpy", "pandas", "statsmodels", "pandas_market_calendars")}}
    return digest(config), config
