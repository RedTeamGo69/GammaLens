"""The weekly recommendation module: one fit path and one forecast/plan/tier
path for the Spread Finder tab, xlsx export, Monday cron, bootstrap and
unattended forward capture.

Fitting: ``fit_specs`` / ``fit_and_save_specs`` own the production column
rule (``production_feature_columns``), the minimum-column skip, the
feature-source routing (XSP rides SPX's rows) and the TRAIN_WINDOW_YEARS /
COVID-excluded read window. ``load_saved_fit`` owns the saved-fit lookup
with its parent-ticker fallback.

Recommending: the displayed (pre-EM-floor) strike is the production
recommendation. The EM-floored ladder remains context; it must not silently
widen this study.
"""
from dataclasses import dataclass, field as _field

from range_finder.har_model import (
    MODEL_SPECS, PI_ALPHA, fit_validation_and_production, forecast_next_week,
    production_feature_columns, train_window_min_date,
)
from range_finder.model_persistence import save_model
from range_finder.spread_levels import build_spread_plan, build_spread_tiers
from range_finder.gex_policy import uses_disabled_gex_feature

# A spec with fewer usable columns than this is skipped, not fit.
MIN_FIT_FEATURES = 2

TIER_KEYS = ("lower_pi", "point", "pi_upper", "effective")
TIER_LABELS = ("Lower PI", "Point Estimate", "80% PI Upper", "Effective (+buffer)")


def displayed_tier(tier):
    return {
        "put_short": tier.model_put_short if tier.model_put_short is not None else tier.put_short,
        "call_short": tier.model_call_short if tier.model_call_short is not None else tier.call_short,
        "put_spreads": tier.model_put_spreads or tier.put_spreads,
        "call_spreads": tier.model_call_spreads or tier.call_spreads,
    }


def tier_bands(tiers):
    return {key: (round(float(displayed_tier(t)["put_short"]), 2),
                  round(float(displayed_tier(t)["call_short"]), 2))
            for key, t in zip(TIER_KEYS, tiers)}


def chain_entry_to_quotes(entry):
    from phase1.quote_filters import filter_to_preferred_root
    calls, puts, _ = filter_to_preferred_root(entry.get("calls", []), entry.get("puts", []))
    quotes = {}
    for side, options in (("call", calls), ("put", puts)):
        for opt in options:
            quote = quotes.setdefault(opt["strike"], {})
            for field in ("bid", "ask"):
                quote[f"{side}_{field}"] = opt.get(field, 0.0) or 0.0
    return quotes


def build_recommendations(*, result, feature_row, feature_cols, reference,
                          vix, week_start, ticker, side_share_q=None,
                          chain_quotes=None, weekly_em=None, conn=None,
                          model_name=None):
    if uses_disabled_gex_feature(feature_cols):
        raise ValueError("Saved fit uses disabled GEX feature; refit required")
    forecast = forecast_next_week(result, feature_row, feature_cols, reference,
                                  alpha=PI_ALPHA, side_share_q=side_share_q)
    from range_finder.conformal import maybe_apply_conformal, CONFORMAL_ENABLED
    if CONFORMAL_ENABLED:
        forecast = maybe_apply_conformal(forecast, conn, ticker, model_name)
    plan = build_spread_plan(forecast=forecast, feature_row=feature_row,
                            week_start=week_start, vix_level=vix, ticker=ticker,
                            chain_quotes=chain_quotes, side_share_q=side_share_q)
    tiers = build_spread_tiers(forecast=forecast, plan=plan, spx_ref=reference,
                              vix_level=vix, chain_quotes=chain_quotes,
                              ticker=ticker, weekly_em=weekly_em)
    return forecast, plan, tiers


# ── Fitting ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpecFit:
    spec: str
    result: object              # statsmodels production fit
    feature_cols: list
    metrics: dict


@dataclass(frozen=True)
class FitReport:
    fits: dict = _field(default_factory=dict)     # spec -> SpecFit
    skipped: dict = _field(default_factory=dict)  # spec -> reason


def load_production_features(conn, ticker):
    """The feature rows production fits and forecasts ``ticker`` from: its
    feature-source ticker's rows, pinned to TRAIN_WINDOW_YEARS (deeper
    backfills must not silently retrain production), COVID excluded."""
    from phase1.ticker_config import feature_source_ticker
    from range_finder.feature_builder import get_features
    return get_features(conn, min_date=train_window_min_date(),
                        exclude_covid=True, ticker=feature_source_ticker(ticker))


def fit_specs(features, specs=None) -> FitReport:
    """Validation + production fit of every spec on ``features``.

    Never raises for one spec: a spec with too few usable columns or a failed
    fit lands in ``skipped`` with its reason, so callers decide whether a
    missing spec is fatal (the cron requires its calibration spec).
    """
    fits, skipped = {}, {}
    for spec in (specs or MODEL_SPECS):
        cols = production_feature_columns(features, spec)
        if len(cols) < MIN_FIT_FEATURES:
            skipped[spec] = f"only {len(cols)} usable features"
            continue
        try:
            _validation, result, metrics = fit_validation_and_production(
                features, feature_cols=cols, model_name=spec)
        except Exception as e:
            skipped[spec] = f"fit failed: {e}"
            continue
        fits[spec] = SpecFit(spec, result, cols, metrics)
    return FitReport(fits, skipped)


def fit_and_save_specs(conn, ticker, *, specs=None, save_tickers=None,
                       features=None) -> FitReport:
    """Fit every spec on ``ticker``'s production features and save each fit
    to saved_models under every ticker in ``save_tickers`` (default: just
    ``ticker``). Pass ``features`` to reuse an already-loaded frame."""
    if features is None:
        features = load_production_features(conn, ticker)
    report = fit_specs(features, specs)
    for fit in report.fits.values():
        for save_ticker in (save_tickers or [ticker]):
            save_model(fit.result, fit.feature_cols, fit.spec, fit.metrics,
                       conn=conn, ticker=save_ticker)
    return report


def load_saved_fit(load, model, ticker):
    """``(payload, source_ticker)`` for ``ticker``'s saved ``model`` fit.

    ``load(model, ticker)`` is the loader (the UI passes its cached one).
    A scaled mini (XSP) rides its feature-source parent's fit — identical by
    construction — only when it has none of its own. An incompatible own fit
    is NOT masked by the parent's: it raises so the caller asks for a refit.
    """
    from phase1.ticker_config import feature_source_ticker
    try:
        return load(model, ticker), ticker
    except FileNotFoundError:
        parent = feature_source_ticker(ticker)
        if parent == ticker:
            raise
        return load(model, parent), parent
