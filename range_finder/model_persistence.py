# =============================================================================
# model_persistence.py
# HAR model serialization — save/load fitted models to Postgres. There is
# no on-disk fallback: Streamlit Cloud's disk is thrown away on restart, so a
# local pickle only hid failed saves behind a "rows saved" success message.
# =============================================================================

import io
import logging
import pickle
import sys
from datetime import datetime, timezone

import statsmodels.api as sm

log = logging.getLogger(__name__)


# ── Restricted unpickling ────────────────────────────────────────────────────
#
# pickle.loads on a blob from Postgres executes whatever
# callables the stream names — a poisoned saved_models row is an RCE
# primitive for anyone who obtains DATABASE_URL. Model payloads only ever
# contain statsmodels results + numpy/pandas containers + stdlib scalars, so
# unpickling is restricted to those module roots. Anything else (os.system,
# subprocess.Popen, builtins.eval, ...) raises UnpicklingError instead of
# executing.

_ALLOWED_MODULE_ROOTS = (
    "statsmodels", "numpy", "pandas", "scipy", "pyarrow",
    "datetime", "dateutil", "pytz", "zoneinfo",
    "collections", "builtins", "copyreg", "_codecs",
)

_BUILTIN_DENYLIST = {"eval", "exec", "compile", "open", "__import__",
                     "getattr", "setattr", "delattr", "input", "breakpoint"}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".")[0]
        if root not in _ALLOWED_MODULE_ROOTS:
            raise pickle.UnpicklingError(
                f"Blocked unpickle of {module}.{name} — model payloads may "
                f"only reference {_ALLOWED_MODULE_ROOTS}")
        if root == "builtins" and name in _BUILTIN_DENYLIST:
            raise pickle.UnpicklingError(f"Blocked unpickle of builtins.{name}")
        return super().find_class(module, name)


def _restricted_loads(blob: bytes):
    return _RestrictedUnpickler(io.BytesIO(blob)).load()

# Pickle fallback path — used only if the DB save raises

# Schema version for saved model payloads. Bump this whenever the payload
# dict shape changes, or whenever the HAR feature definitions change in a
# way that invalidates previously-fitted models (so load_model refuses to
# return a stale-but-loadable model instead of silently producing wrong
# forecasts). Load will raise a clear error on mismatch and the UI will
# fall through to "refit the model" guidance.
#
# v2 (2026-04): vix_implied_range rescaled by Brownian range factor
# (2*sqrt(2/pi) ≈ 1.5958). Old fits have a coefficient calibrated to the
# unscaled 1-SD vol; loading them against the rescaled feature would shift
# every forecast. Refit required.
#
# v3 (2026-06): feature-set change across the weekly specs. vix_implied_range
# was dropped from M2_vix/M3_extended/M4_full (it was perfectly collinear with
# vix_close — see har_model.MODEL_SPECS), and M6_regime was removed entirely.
# A v2 fit carries the old feature_cols (incl. vix_implied_range / regime
# interactions), so loading it would predict off a different design than the
# current spec. Refit required — the Monday cron and the UI "Forecast" button
# both regenerate fits automatically on the next run.
#
# NOT a version bump (2026-07): the Cboe/Tradier data-source migration
# changed WHERE training data comes from and HOW MUCH of it exists, but not
# a single feature definition — saved fits stay loadable and correct.
SCHEMA_VERSION = 3


def save_model(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_cols: list[str],
    model_name: str,
    metrics: dict,
    conn,
    ticker: str = "SPX",
):
    """Save the fitted model + metadata to Postgres; a DB error raises.

    `ticker` scopes the fit so SPX and XSP can coexist under the same
    `model_name`; the row is keyed by (model_name, ticker).
    """
    payload = {
        "schema_version":     SCHEMA_VERSION,
        "statsmodels_version": sm.__version__,
        "python_version":     f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "result":             result,
        "feature_cols":       feature_cols,
        "model_name":         model_name,
        "ticker":             ticker,
        "metrics":            metrics,
        "fitted_at":          datetime.now(timezone.utc).isoformat(),
    }

    blob = pickle.dumps(payload)
    now = datetime.now(timezone.utc).isoformat()

    import psycopg2
    conn.execute("""
        INSERT INTO saved_models (model_name, ticker, model_data, fitted_at, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT(model_name, ticker) DO UPDATE SET
            model_data = EXCLUDED.model_data,
            fitted_at  = EXCLUDED.fitted_at,
            updated_at = EXCLUDED.updated_at
    """, (model_name, ticker, psycopg2.Binary(blob), payload["fitted_at"], now))
    conn.commit()
    log.info(f"Model saved to database: {ticker}/{model_name}")


class IncompatibleModelError(Exception):
    """Raised when a loaded model payload doesn't match the current schema.

    The UI catches this as "saved model incompatible" and prompts the user
    to click Generate Forecast to refit. Raising early is strictly better
    than silently returning a model that would produce wrong predictions.
    """


def _validate_payload(payload: dict, source: str) -> None:
    """Raise IncompatibleModelError if the payload schema is a version we
    can't safely load. Missing schema_version is treated as pre-versioning
    legacy (version 0) and accepted with a warning — existing models keep
    working across this commit. Going forward, bumping SCHEMA_VERSION and
    catching the mismatch is how we force a refit when the HAR feature
    definitions change in a way that invalidates old fits."""
    # Sanity-check required fields first — if these are missing the
    # payload is corrupted, not just old.
    for required in ("result", "feature_cols", "model_name"):
        if required not in payload:
            raise IncompatibleModelError(
                f"{source}: saved model payload missing required field "
                f"'{required}' — payload is corrupted or incomplete."
            )

    payload_version = payload.get("schema_version", 0)
    if payload_version == 0:
        log.warning(
            f"{source}: loading legacy pre-versioned model (no schema_version "
            f"field). Current SCHEMA_VERSION={SCHEMA_VERSION}. Consider "
            f"refitting to pick up any feature definition changes."
        )
        return
    if payload_version != SCHEMA_VERSION:
        raise IncompatibleModelError(
            f"{source}: saved model schema_version={payload_version} but "
            f"current SCHEMA_VERSION={SCHEMA_VERSION}. Refit to regenerate."
        )
    # Warn if statsmodels version differs so unpickle quirks are easier
    # to diagnose. Don't raise — minor version bumps are usually
    # backward-compatible and forcing a refit on every upgrade is too
    # aggressive.
    saved_sm = payload.get("statsmodels_version")
    if saved_sm and saved_sm != sm.__version__:
        log.warning(
            f"{source}: saved model fitted with statsmodels {saved_sm}, "
            f"loading with {sm.__version__} — should still work but flag "
            f"any prediction anomalies."
        )


def load_model(model_name: str, conn, ticker: str = "SPX") -> dict:
    """Load a previously saved model from Postgres.

    `ticker` scopes the lookup so SPX and XSP fits don't collide.

    Raises FileNotFoundError when no fit is saved (callers then try the
    parent ticker or prompt a fit), IncompatibleModelError if the payload
    schema_version doesn't match SCHEMA_VERSION ("model needs refitting"),
    and lets a DB error propagate rather than serving a stale fit.
    """
    cur = conn.execute(
        "SELECT model_data, fitted_at FROM saved_models "
        "WHERE model_name = %s AND ticker = %s",
        (model_name, ticker)
    )
    row = cur.fetchone()
    if not row:
        raise FileNotFoundError(f"No saved model found for {ticker}/{model_name}")
    blob = row[0]
    if isinstance(blob, memoryview):
        blob = bytes(blob)
    payload = _restricted_loads(blob)
    _validate_payload(payload, f"DB:{ticker}/{model_name}")
    log.info(f"Model loaded from database: {ticker}/{model_name}  (fitted {payload['fitted_at']})")
    return payload
