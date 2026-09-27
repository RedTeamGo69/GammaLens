"""The weekly recommendation module's fit and saved-fit interface.

Before this module owned fitting, five orchestrations (UI Forecast, UI
warm-up, Monday cron, bootstrap, har_model.run_full_pipeline) each ran their
own "fit every spec" loop with different column filters and skip rules.
"""
import numpy as np
import pandas as pd
import pytest

import range_finder.recommendations as rec
from range_finder.model_persistence import IncompatibleModelError


def _features(n_complete: int = 80) -> pd.DataFrame:
    """HAR-core-only frame: M1 is fittable, every richer spec loses columns."""
    idx = pd.date_range("2024-01-08", periods=n_complete + 1, freq="W-MON")
    pos = np.arange(len(idx), dtype=float)
    rng = np.random.default_rng(7)
    d1 = 0.02 + pos * 0.0001 + rng.normal(0, 0.002, len(idx))
    wk = 0.025 + np.sin(pos / 4.0) * 0.003
    mo = 0.03 + np.cos(pos / 7.0) * 0.002
    df = pd.DataFrame({"har_d1": d1, "har_w": wk, "har_m": mo}, index=idx)
    df["log_range"] = np.log(0.01 + 0.5 * d1 + 0.3 * wk + 0.2 * mo)
    df.loc[idx[-1], "log_range"] = np.nan          # forecast scaffold row
    return df


def test_fit_specs_fits_each_spec_on_its_usable_columns():
    report = rec.fit_specs(_features(), specs=["M1_baseline", "M2_vix"])
    m1 = report.fits["M1_baseline"]
    assert m1.feature_cols == ["har_d1", "har_w", "har_m"]
    assert int(m1.result.nobs) == 80              # scaffold row excluded
    assert "oos_r2" in m1.metrics
    # Production behavior: a spec missing a column (vix_close) still fits on
    # what it has, as long as >= MIN_FIT_FEATURES remain.
    assert report.fits["M2_vix"].feature_cols == ["har_d1", "har_w", "har_m"]
    assert report.skipped == {}


def test_fit_specs_skips_spec_with_too_few_usable_columns():
    df = _features()[["har_d1", "log_range"]]
    report = rec.fit_specs(df, specs=["M1_baseline"])
    assert report.fits == {}
    assert "1 usable" in report.skipped["M1_baseline"]


def test_fit_specs_records_a_failed_fit_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise ValueError("singular")
    monkeypatch.setattr(rec, "fit_validation_and_production", boom)
    report = rec.fit_specs(_features(), specs=["M1_baseline"])
    assert report.fits == {}
    assert "singular" in report.skipped["M1_baseline"]


def test_fit_and_save_specs_loads_production_window_and_saves_each_ticker(monkeypatch):
    loaded, saved = [], []
    monkeypatch.setattr(rec, "load_production_features",
                        lambda conn, ticker: loaded.append(ticker) or _features())
    monkeypatch.setattr(rec, "save_model",
                        lambda result, cols, spec, metrics, conn=None, ticker=None:
                        saved.append((spec, ticker)))
    report = rec.fit_and_save_specs("conn", "XSP", specs=["M1_baseline"],
                                    save_tickers=["SPX", "XSP"])
    assert loaded == ["XSP"]
    assert saved == [("M1_baseline", "SPX"), ("M1_baseline", "XSP")]
    assert set(report.fits) == {"M1_baseline"}


def test_fit_and_save_specs_defaults_to_saving_under_the_ticker(monkeypatch):
    saved = []
    monkeypatch.setattr(rec, "load_production_features", lambda conn, ticker: _features())
    monkeypatch.setattr(rec, "save_model",
                        lambda result, cols, spec, metrics, conn=None, ticker=None:
                        saved.append(ticker))
    rec.fit_and_save_specs("conn", "SPY", specs=["M1_baseline"])
    assert saved == ["SPY"]


def test_load_production_features_routes_to_feature_source(monkeypatch):
    calls = []
    import range_finder.feature_builder as fb
    monkeypatch.setattr(fb, "get_features",
                        lambda conn, **kw: calls.append(kw) or pd.DataFrame())
    rec.load_production_features("conn", "XSP")
    assert calls[0]["ticker"] == "SPX"               # XSP rides SPX's rows
    assert calls[0]["exclude_covid"] is True
    assert calls[0]["min_date"] is not None


# ── saved-fit lookup ──────────────────────────────────────────────────────────

def test_load_saved_fit_prefers_the_tickers_own_fit():
    payload, source = rec.load_saved_fit(
        lambda m, t: {"ticker": t}, "M3_extended", "XSP")
    assert (payload["ticker"], source) == ("XSP", "XSP")


def test_load_saved_fit_falls_back_to_parent_when_ticker_has_none():
    def load(model, ticker):
        if ticker == "XSP":
            raise FileNotFoundError("no XSP fit")
        return {"ticker": ticker}
    payload, source = rec.load_saved_fit(load, "M3_extended", "XSP")
    assert (payload["ticker"], source) == ("SPX", "SPX")


def test_load_saved_fit_does_not_mask_an_incompatible_own_fit():
    def load(model, ticker):
        raise IncompatibleModelError("schema v1")
    with pytest.raises(IncompatibleModelError):
        rec.load_saved_fit(load, "M3_extended", "XSP")


def test_load_saved_fit_without_parent_reraises_not_found():
    def load(model, ticker):
        raise FileNotFoundError("none")
    with pytest.raises(FileNotFoundError):
        rec.load_saved_fit(load, "M3_extended", "SPY")
