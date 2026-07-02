"""Conformal interval correction (range_finder.conformal).

Pins the safety contracts that let this ship wired-but-disabled:

  * maybe_apply_conformal is a strict no-op while CONFORMAL_ENABLED is
    False or no λ row exists,
  * apply_conformal returns a NEW dict, never mutates, λ=1 preserves
    bounds, λ>1 widens monotonically, price levels stay exp-consistent
    with the symmetric-band convention,
  * conformal_lambda ≈ 1 on perfectly-calibrated Gaussian input,
  * the adoption gate rejects when intervals are already calibrated and
    keeps when they're materially overconfident.
"""
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

import range_finder.conformal as cf


# ── fixtures ───────────────────────────────────────────────────────────────────

def _forecast() -> dict:
    return {
        "point_pct": 0.0200,
        "lower_pct": 0.0140,
        "upper_pct": 0.0286,   # ~symmetric around point on the log scale
        "point_upper_px": 6060.0,
        "point_lower_px": 5940.0,
        "pi_upper_px": 6085.8,
        "pi_lower_px": 5914.2,
        "spx_ref_close": 6000.0,
        "confidence_level": 80,
        "alpha": 0.20,
    }


def _wf_frame(n: int = 200, width_factor: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Walk-forward-shaped frame with N(0,1) errors and CI half-width set to
    width_factor * (true 80% Gaussian half-width = 1.2816σ)."""
    rng = np.random.default_rng(seed)
    y_pred = np.zeros(n)
    y_true = rng.normal(0.0, 1.0, n)
    half = 1.2816 * width_factor
    return pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred,
        "obs_ci_lower": y_pred - half, "obs_ci_upper": y_pred + half,
        "train_n": 100,
    }, index=pd.date_range("2023-01-02", periods=n, freq="7D"))


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE interval_calibration (
            ticker TEXT NOT NULL, model_name TEXT NOT NULL,
            lambda REAL NOT NULL, n_obs INTEGER, method TEXT, updated_at TEXT,
            PRIMARY KEY (ticker, model_name)
        )
    """)
    return conn


# ── apply_conformal ────────────────────────────────────────────────────────────

def test_lambda_one_preserves_bounds_but_returns_new_dict():
    fc = _forecast()
    out = cf.apply_conformal(fc, 1.0)

    assert out is not fc
    assert out["upper_pct"] == pytest.approx(fc["upper_pct"], abs=1e-4)
    assert out["lower_pct"] == pytest.approx(fc["lower_pct"], abs=1e-4)
    assert out["point_pct"] == fc["point_pct"]
    assert out["conformal_lambda"] == pytest.approx(1.0)


def test_lambda_above_one_widens_monotone():
    fc = _forecast()
    out12 = cf.apply_conformal(fc, 1.2)
    out15 = cf.apply_conformal(fc, 1.5)

    assert out12["upper_pct"] > fc["upper_pct"]
    assert out12["lower_pct"] < fc["lower_pct"]
    assert out15["upper_pct"] > out12["upper_pct"]
    assert out15["lower_pct"] < out12["lower_pct"]
    # point forecast untouched
    assert out15["point_pct"] == fc["point_pct"]


def test_price_levels_exp_consistent():
    fc = _forecast()
    out = cf.apply_conformal(fc, 1.3)

    # Symmetric-band convention: both PI price levels derive from upper_pct.
    # px is computed from the UNROUNDED bound (matching forecast_next_week),
    # while the dict's upper_pct is rounded to 4dp — allow that rounding slack.
    assert out["pi_upper_px"] == pytest.approx(
        fc["spx_ref_close"] * (1 + out["upper_pct"] / 2), abs=0.5)
    assert out["pi_lower_px"] == pytest.approx(
        fc["spx_ref_close"] * (1 - out["upper_pct"] / 2), abs=0.5)
    # log-scale scaling: log-width grew by exactly λ
    old_w = math.log(fc["upper_pct"]) - math.log(fc["lower_pct"])
    new_w = math.log(out["upper_pct"]) - math.log(out["lower_pct"])
    assert new_w == pytest.approx(1.3 * old_w, rel=0.02)


def test_apply_does_not_mutate_input():
    fc = _forecast()
    snapshot = dict(fc)
    cf.apply_conformal(fc, 1.5)
    assert fc == snapshot


# ── conformal_lambda ───────────────────────────────────────────────────────────

def test_lambda_near_one_when_calibrated():
    lam = cf.conformal_lambda(_wf_frame(width_factor=1.0), alpha=0.20)
    assert 0.85 <= lam <= 1.20


def test_lambda_above_one_when_overconfident():
    # Intervals half as wide as they should be -> λ ≈ 2
    lam = cf.conformal_lambda(_wf_frame(width_factor=0.5), alpha=0.20)
    assert 1.7 <= lam <= 2.4


def test_lambda_requires_enough_rows():
    with pytest.raises(ValueError):
        cf.conformal_lambda(_wf_frame(n=5), alpha=0.20)


# ── persistence + production hook ──────────────────────────────────────────────

def test_maybe_apply_noop_when_disabled(monkeypatch):
    conn = _make_conn()
    cf.upsert_lambda(conn, "SPX", "M3_extended", 1.5, 100)
    monkeypatch.setattr(cf, "CONFORMAL_ENABLED", False)
    fc = _forecast()

    out = cf.maybe_apply_conformal(fc, conn, "SPX", "M3_extended")

    assert out is fc                      # same object, untouched


def test_maybe_apply_noop_without_lambda_row(monkeypatch):
    conn = _make_conn()
    monkeypatch.setattr(cf, "CONFORMAL_ENABLED", True)
    fc = _forecast()

    out = cf.maybe_apply_conformal(fc, conn, "SPX", "M3_extended")

    assert out is fc


def test_maybe_apply_applies_when_enabled_and_persisted(monkeypatch):
    conn = _make_conn()
    cf.upsert_lambda(conn, "SPX", "M3_extended", 1.5, 100)
    monkeypatch.setattr(cf, "CONFORMAL_ENABLED", True)
    fc = _forecast()

    out = cf.maybe_apply_conformal(fc, conn, "SPX", "M3_extended")

    assert out is not fc
    assert out["conformal_lambda"] == pytest.approx(1.5)
    assert out["upper_pct"] > fc["upper_pct"]


def test_upsert_lambda_idempotent_overwrite():
    conn = _make_conn()
    cf.upsert_lambda(conn, "SPX", "M3_extended", 1.5, 100)
    cf.upsert_lambda(conn, "SPX", "M3_extended", 1.2, 120)

    assert cf.load_lambda(conn, "SPX", "M3_extended") == pytest.approx(1.2)
    n = conn.execute("SELECT COUNT(*) FROM interval_calibration").fetchone()[0]
    assert n == 1


# ── adoption gate ──────────────────────────────────────────────────────────────

def test_gate_rejects_calibrated_intervals():
    # n=500 keeps holdout sampling noise below the 3pp gate; at small n a
    # calibrated series can trip the gate by luck — that's expected behavior
    # for an empirical gate, not a bug.
    gate = cf.evaluate_conformal_gate(_wf_frame(n=500, width_factor=1.0))
    assert gate["keep"] is False          # nothing to fix


def test_gate_keeps_overconfident_intervals():
    gate = cf.evaluate_conformal_gate(_wf_frame(width_factor=0.5))
    assert gate["keep"] is True
    assert gate["lambda"] > 1.5
    # held-out coverage moved toward nominal
    assert abs(gate["coverage_after"] - gate["nominal"]) \
        < abs(gate["coverage_before"] - gate["nominal"])


def test_gate_requires_enough_observations():
    with pytest.raises(ValueError):
        cf.evaluate_conformal_gate(_wf_frame(n=30))
