"""PI-coverage calibration math (range_finder.calibration).

Pins the audit's core contracts:

  * one-sided coverage = share of scored periods with actual <= PI upper,
  * two-sided coverage computed only over rows that carry lower_pct
    (legacy rows without it can't fake coverage),
  * buffer breach rate audits actual > effective_range (PI upper + buffer),
  * per-spec breakdown pools NULL model_name rows as "(unknown)",
  * n < MIN_SAMPLE is flagged insufficient — numbers, not conclusions,
  * the DB loaders only return rows with BOTH a forecast and an outcome.
"""
import sqlite3

import pandas as pd
import pytest

import range_finder.calibration as cal


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _forecast_frame(rows) -> pd.DataFrame:
    """rows: (actual, lower, upper, effective, model_name) tuples."""
    return pd.DataFrame({
        "week_start": pd.date_range("2026-01-05", periods=len(rows), freq="7D"),
        "actual_range_pct": [r[0] for r in rows],
        "lower_pct": [r[1] for r in rows],
        "upper_pct": [r[2] for r in rows],
        "effective_range_pct": [r[3] for r in rows],
        "model_name": [r[4] for r in rows],
    })


def _make_conn() -> sqlite3.Connection:
    """In-memory DB mirroring the spread_log columns the loaders read."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE spread_log (
            week_start TEXT NOT NULL, ticker TEXT NOT NULL DEFAULT 'SPX',
            model_name TEXT, point_pct REAL, lower_pct REAL, upper_pct REAL,
            effective_range_pct REAL, buffer_pct REAL, event_count INTEGER,
            actual_high REAL, actual_low REAL, actual_range_pct REAL,
            outcome TEXT,
            PRIMARY KEY (week_start, ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE forecast_log_daily (
            session_date TEXT NOT NULL, ticker TEXT NOT NULL DEFAULT 'SPX',
            model_name TEXT NOT NULL, point_pct REAL, lower_pct REAL,
            upper_pct REAL, spx_ref REAL, vix1d_close REAL,
            generated_at TEXT, actual_range_pct REAL, scored_at TEXT,
            PRIMARY KEY (session_date, ticker, model_name)
        )
    """)
    return conn


# ── pi_coverage ────────────────────────────────────────────────────────────────

def test_one_sided_coverage_known_answer():
    # 8 of 10 actuals inside the upper bound -> 0.80 exactly
    rows = [(0.01, 0.005, 0.02, 0.023, "M3")] * 8 + \
           [(0.03, 0.005, 0.02, 0.023, "M3")] * 2
    cov = cal.pi_coverage(_forecast_frame(rows))

    assert cov["one_sided"] == pytest.approx(0.80)
    assert cov["n"] == 10


def test_two_sided_only_counts_rows_with_lower():
    rows = [
        (0.010, 0.005, 0.02, None, "M3"),   # inside both
        (0.001, 0.005, 0.02, None, "M3"),   # below lower -> outside two-sided
        (0.010, None,  0.02, None, None),   # legacy row: no lower_pct
    ]
    cov = cal.pi_coverage(_forecast_frame(rows))

    assert cov["n"] == 3
    assert cov["n_two_sided"] == 2
    assert cov["two_sided"] == pytest.approx(0.5)
    assert cov["one_sided"] == pytest.approx(1.0)   # all three <= upper


def test_boundary_touch_counts_as_inside():
    rows = [(0.02, 0.005, 0.02, None, "M3")]        # actual == upper
    cov = cal.pi_coverage(_forecast_frame(rows))
    assert cov["one_sided"] == pytest.approx(1.0)


def test_empty_frame_is_nan_not_crash():
    cov = cal.pi_coverage(_forecast_frame([]))
    assert cov["n"] == 0
    assert cov["one_sided"] != cov["one_sided"]      # NaN
    assert cov["sufficient"] is False


def test_sufficiency_threshold():
    rows = [(0.01, 0.005, 0.02, None, "M3")] * (cal.MIN_SAMPLE - 1)
    assert cal.pi_coverage(_forecast_frame(rows))["sufficient"] is False
    rows = [(0.01, 0.005, 0.02, None, "M3")] * cal.MIN_SAMPLE
    assert cal.pi_coverage(_forecast_frame(rows))["sufficient"] is True


def test_binomial_ci_brackets_point_estimate():
    cov = cal.pi_coverage(_forecast_frame(
        [(0.01, 0.005, 0.02, None, "M3")] * 16 +
        [(0.03, 0.005, 0.02, None, "M3")] * 4
    ))
    lo, hi = cov["one_sided_ci"]
    assert lo < cov["one_sided"] < hi
    assert 0.0 <= lo and hi <= 1.0


# ── buffer_breach_rate ─────────────────────────────────────────────────────────

def test_buffer_breach_rate_known_answer():
    rows = [(0.010, None, 0.02, 0.023, "M3")] * 9 + \
           [(0.030, None, 0.02, 0.023, "M3")] * 1   # one blows through buffer
    buf = cal.buffer_breach_rate(_forecast_frame(rows))

    assert buf["breach_rate"] == pytest.approx(0.10)
    assert buf["n"] == 10


def test_buffer_ignores_rows_without_effective_range():
    rows = [(0.010, None, 0.02, None, "M3")] * 5
    buf = cal.buffer_breach_rate(_forecast_frame(rows))
    assert buf["n"] == 0


# ── rolling_coverage / coverage_by_model ──────────────────────────────────────

def test_rolling_coverage_window():
    rows = [(0.01, None, 0.02, None, "M3")] * 30
    out = cal.rolling_coverage(_forecast_frame(rows), window=26)

    assert out["rolling_one_sided"].isna().sum() == 25    # warmup
    assert out["rolling_one_sided"].dropna().iloc[-1] == pytest.approx(1.0)


def test_coverage_by_model_pools_unknown():
    rows = [
        (0.01, None, 0.02, None, "M3_extended"),
        (0.03, None, 0.02, None, "M3_extended"),
        (0.01, None, 0.02, None, None),
        (0.01, None, 0.02, None, None),
    ]
    out = cal.coverage_by_model(_forecast_frame(rows))

    by_name = {r["model_name"]: r for _, r in out.iterrows()}
    assert by_name["M3_extended"]["one_sided"] == pytest.approx(0.5)
    assert by_name["(unknown)"]["one_sided"] == pytest.approx(1.0)


# ── DB loaders ─────────────────────────────────────────────────────────────────

def test_loader_returns_only_completed_rows():
    conn = _make_conn()
    conn.executemany(
        "INSERT INTO spread_log (week_start, ticker, model_name, upper_pct, "
        "lower_pct, actual_range_pct, outcome) VALUES (?, 'SPX', ?, ?, ?, ?, ?)",
        [
            ("2026-06-01", "M3", 0.02, 0.005, 0.015, "full_profit"),  # complete
            ("2026-06-08", "M3", 0.02, 0.005, None, None),            # unscored
            ("2026-06-15", "M3", None, None, 0.015, "full_profit"),   # no forecast
        ],
    )
    conn.commit()

    df = cal.load_completed_forecasts(conn)

    assert len(df) == 1
    assert df["week_start"].iloc[0] == pd.Timestamp("2026-06-01")


def test_daily_loader_filters_by_model():
    conn = _make_conn()
    conn.executemany(
        "INSERT INTO forecast_log_daily (session_date, ticker, model_name, "
        "upper_pct, actual_range_pct) VALUES (?, 'SPX', ?, ?, ?)",
        [
            ("2026-06-29", "M2_daily_vix", 0.008, 0.006),
            ("2026-06-29", "M1_daily_baseline", 0.009, 0.006),
            ("2026-06-30", "M2_daily_vix", 0.008, None),   # unscored
        ],
    )
    conn.commit()

    df_all = cal.load_completed_daily_forecasts(conn)
    df_m2 = cal.load_completed_daily_forecasts(conn, model_name="M2_daily_vix")

    assert len(df_all) == 2
    assert len(df_m2) == 1
    assert df_m2["model_name"].iloc[0] == "M2_daily_vix"


def test_weekly_pi_coverage_wrapper():
    conn = _make_conn()
    weeks = ["2026-01-05", "2026-01-12", "2026-01-19",
             "2026-01-26", "2026-02-02", "2026-02-09"]
    conn.executemany(
        "INSERT INTO spread_log (week_start, ticker, model_name, upper_pct, "
        "lower_pct, effective_range_pct, actual_range_pct, outcome) "
        "VALUES (?, 'SPX', 'M3', 0.02, 0.005, 0.023, 0.015, 'full_profit')",
        [(w,) for w in weeks],
    )
    conn.commit()

    cov = cal.weekly_pi_coverage(conn)

    assert cov["one_sided"] == pytest.approx(1.0)
    assert cov["buffer"]["breach_rate"] == pytest.approx(0.0)
