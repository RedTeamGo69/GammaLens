"""Expanding-window walk-forward evaluation (range_finder.walkforward).

Pins the properties that make walk-forward results trustworthy:

  * NO LEAKAGE — every prediction comes from a fit whose training window
    ends strictly before the predicted observation,
  * refit cadence — the model refits every `step` observations, not per-obs
    (runtime) and not once (that would just be the old single split),
  * summarize() math on hand-checkable inputs, incl. empirical PI coverage,
  * the WLS fit-function switch runs end-to-end,
  * compare_configs applies the adoption gate incl. the coverage guard.
"""
import numpy as np
import pandas as pd
import pytest

import range_finder.walkforward as wf
from range_finder.walkforward import WalkForwardConfig


# ── fixtures ───────────────────────────────────────────────────────────────────

def _synthetic_weekly(n: int = 160, seed: int = 42) -> pd.DataFrame:
    """Deterministic AR-flavored series with one predictive feature."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.3, n)
    y = -5.0 + 0.5 * x + noise          # log_range-scaled magnitudes
    idx = pd.date_range("2023-01-02", periods=n, freq="7D")
    return pd.DataFrame({"log_range": y, "feat": x}, index=idx)


# ── no leakage / refit cadence ─────────────────────────────────────────────────

def test_no_leakage_train_window_precedes_prediction():
    df = _synthetic_weekly()
    out = wf.walk_forward_evaluate(df, ["feat"], min_train=100, step=5)

    clean = df[["log_range", "feat"]].dropna()
    positions = clean.index.get_indexer(out.index)
    # train_n rows were used to fit the model that predicted position i:
    # the training window is iloc[:train_n], so train_n <= i means the
    # predicted observation was never inside its own training set.
    assert (out["train_n"].values <= positions).all()
    assert (out["train_n"].values >= 100).all()


def test_refit_count_matches_step():
    df = _synthetic_weekly(n=160)
    calls = []

    def counting_fit(X, y, model_name="x", **kw):
        calls.append(len(y))
        return wf.fit_model(X, y, model_name=model_name)

    out = wf.walk_forward_evaluate(df, ["feat"], min_train=100, step=5,
                                   fit_fn=counting_fit)

    n_oos = len(out)                      # 160 - 100 = 60 predictions
    assert n_oos == 60
    import math
    assert len(calls) == math.ceil(n_oos / 5)   # refit every 5 obs
    assert calls == sorted(calls)               # expanding, never shrinking


def test_too_few_rows_raises():
    df = _synthetic_weekly(n=50)
    with pytest.raises(ValueError):
        wf.walk_forward_evaluate(df, ["feat"], min_train=100)


# ── summarize ──────────────────────────────────────────────────────────────────

def test_summarize_hand_checked_values():
    frame = pd.DataFrame({
        "y_true":       [1.0, 2.0, 3.0, 4.0],
        "y_pred":       [1.0, 2.0, 3.0, 5.0],
        "obs_ci_lower": [0.0, 1.0, 2.0, 4.5],   # last: y_true below lower
        "obs_ci_upper": [2.0, 3.0, 4.0, 5.5],
    }, index=pd.date_range("2026-01-05", periods=4, freq="7D"))

    s = wf.summarize(frame, alpha=0.20)

    assert s["n_oos"] == 4
    assert s["mae_log"] == pytest.approx(0.25)          # |4-5|/4
    assert s["coverage_two_sided"] == pytest.approx(0.75)
    assert s["coverage_one_sided"] == pytest.approx(1.0)  # all <= upper
    assert s["nominal_two_sided"] == pytest.approx(0.80)
    assert s["coverage_gap"] == pytest.approx(-0.05)


def test_summarize_perfect_predictions():
    frame = pd.DataFrame({
        "y_true":       [1.0, 2.0, 3.0],
        "y_pred":       [1.0, 2.0, 3.0],
        "obs_ci_lower": [0.5, 1.5, 2.5],
        "obs_ci_upper": [1.5, 2.5, 3.5],
    }, index=pd.date_range("2026-01-05", periods=3, freq="7D"))

    s = wf.summarize(frame)

    assert s["oos_r2"] == pytest.approx(1.0)
    assert s["mae_log"] == pytest.approx(0.0)
    assert s["coverage_two_sided"] == pytest.approx(1.0)


# ── WLS switch / compare_configs ───────────────────────────────────────────────

def test_wls_config_runs():
    df = _synthetic_weekly()
    out = wf.walk_forward_evaluate(
        df, ["feat"], min_train=100, step=10,
        fit_fn=wf.fit_model_wls, fit_kwargs={"half_life": 52},
    )
    assert len(out) == 60
    assert out["y_pred"].notna().all()


def test_compare_configs_gate_and_baseline():
    df = _synthetic_weekly()
    configs = [
        WalkForwardConfig(label="base", feature_cols=("feat",)),
        # identical spec — cannot clear the improvement gate
        WalkForwardConfig(label="same-again", feature_cols=("feat",)),
    ]

    comp = wf.compare_configs(df, configs, baseline_label="base",
                              min_train=100, step=10)

    by_label = {r["label"]: r for _, r in comp.iterrows()}
    assert by_label["base"]["keep"] == False          # baseline never a candidate
    assert by_label["same-again"]["keep"] == False    # zero improvement
    assert by_label["same-again"]["delta_r2"] == pytest.approx(0.0)


def test_compare_configs_min_date_filters():
    df = _synthetic_weekly(n=220)
    cutoff = str(df.index[60].date())
    configs = [
        WalkForwardConfig(label="full", feature_cols=("feat",)),
        WalkForwardConfig(label="short", feature_cols=("feat",),
                          min_date=cutoff),
    ]

    comp = wf.compare_configs(df, configs, baseline_label="full",
                              min_train=100, step=10)

    by_label = {r["label"]: r for _, r in comp.iterrows()}
    # full: 220-100=120 OOS preds; short: (220-60)-100=60
    assert by_label["full"]["n_oos"] == 120
    assert by_label["short"]["n_oos"] == 60


def test_compare_configs_missing_baseline_raises():
    df = _synthetic_weekly()
    with pytest.raises(RuntimeError):
        wf.compare_configs(
            df, [WalkForwardConfig(label="only", feature_cols=("feat",))],
            baseline_label="nope", min_train=100, step=10,
        )
