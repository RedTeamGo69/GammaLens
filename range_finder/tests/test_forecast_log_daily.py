"""0DTE forecast logging + scoring (range_finder.forecast_log_daily).

Pins the loop that makes the daily calibration series trustworthy:

  * logging is upsert-idempotent (same-morning rerun overwrites in place),
  * scoring joins the RIGHT session's range_pct from daily_spx, only for
    sessions strictly before today (a partial in-progress bar can never be
    recorded as an outcome), and skips already-scored rows,
  * a forecast for a session daily_spx doesn't have stays unscored.
"""
import sqlite3

import pytest

import range_finder.forecast_log_daily as fld


# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE forecast_log_daily (
            session_date     TEXT NOT NULL,
            ticker           TEXT NOT NULL DEFAULT 'SPX',
            model_name       TEXT NOT NULL,
            point_pct        REAL,
            lower_pct        REAL,
            upper_pct        REAL,
            spx_ref          REAL,
            vix1d_close      REAL,
            generated_at     TEXT,
            actual_range_pct REAL,
            scored_at        TEXT,
            PRIMARY KEY (session_date, ticker, model_name)
        )
    """)
    conn.execute("""
        CREATE TABLE daily_spx (
            session_date TEXT NOT NULL,
            ticker       TEXT NOT NULL DEFAULT 'SPX',
            range_pct    REAL,
            PRIMARY KEY (session_date, ticker)
        )
    """)
    return conn


def _forecast(point=0.008, lower=0.005, upper=0.012, ref=6000.0) -> dict:
    return {"point_pct": point, "lower_pct": lower, "upper_pct": upper,
            "spx_ref_close": ref}


# ── log_daily_forecast ─────────────────────────────────────────────────────────

def test_log_writes_bounds_only_row():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-07-01", "M2_daily_vix", _forecast(),
                           vix1d_close=13.02)

    row = conn.execute(
        "SELECT point_pct, lower_pct, upper_pct, spx_ref, vix1d_close, "
        "actual_range_pct FROM forecast_log_daily"
    ).fetchone()
    assert row[0] == pytest.approx(0.008)
    assert row[2] == pytest.approx(0.012)
    assert row[3] == pytest.approx(6000.0)
    assert row[4] == pytest.approx(13.02)
    assert row[5] is None                     # unscored at write time


def test_log_upsert_is_idempotent():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-07-01", "M2_daily_vix", _forecast(upper=0.012))
    fld.log_daily_forecast(conn, "2026-07-01", "M2_daily_vix", _forecast(upper=0.014))

    rows = conn.execute("SELECT upper_pct FROM forecast_log_daily").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.014)  # last write wins


def test_specs_keep_separate_rows():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-07-01", "M2_daily_vix", _forecast())
    fld.log_daily_forecast(conn, "2026-07-01", "M1_daily_baseline", _forecast())

    n = conn.execute("SELECT COUNT(*) FROM forecast_log_daily").fetchone()[0]
    assert n == 2


# ── score_daily_outcomes ───────────────────────────────────────────────────────

def test_scoring_joins_right_session():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-06-30", "M2_daily_vix", _forecast())
    fld.log_daily_forecast(conn, "2026-07-01", "M2_daily_vix", _forecast())
    conn.executemany(
        "INSERT INTO daily_spx (session_date, ticker, range_pct) VALUES (?, 'SPX', ?)",
        [("2026-06-30", 0.0074), ("2026-07-01", 0.0102)],
    )
    conn.commit()

    scored = fld.score_daily_outcomes(conn, before_date="2026-07-01")

    assert scored == 1                        # only the completed session
    rows = dict(conn.execute(
        "SELECT session_date, actual_range_pct FROM forecast_log_daily"
    ).fetchall())
    assert rows["2026-06-30"] == pytest.approx(0.0074)
    assert rows["2026-07-01"] is None         # today: still in progress


def test_scoring_skips_already_scored_rows():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-06-30", "M2_daily_vix", _forecast())
    conn.execute(
        "INSERT INTO daily_spx (session_date, ticker, range_pct) "
        "VALUES ('2026-06-30', 'SPX', 0.0074)")
    conn.commit()

    assert fld.score_daily_outcomes(conn, before_date="2026-07-02") == 1
    # daily_spx value changes later (e.g. bar revision) — scored rows stay put
    conn.execute("UPDATE daily_spx SET range_pct = 0.0999")
    conn.commit()
    assert fld.score_daily_outcomes(conn, before_date="2026-07-02") == 0

    val = conn.execute(
        "SELECT actual_range_pct FROM forecast_log_daily").fetchone()[0]
    assert val == pytest.approx(0.0074)


def test_scoring_leaves_missing_sessions_unscored():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-06-30", "M2_daily_vix", _forecast())
    # no daily_spx row for that session at all

    assert fld.score_daily_outcomes(conn, before_date="2026-07-02") == 0
    val = conn.execute(
        "SELECT actual_range_pct FROM forecast_log_daily").fetchone()[0]
    assert val is None


def test_scoring_ignores_null_range_pct_bars():
    conn = _make_conn()
    fld.log_daily_forecast(conn, "2026-06-30", "M2_daily_vix", _forecast())
    conn.execute(
        "INSERT INTO daily_spx (session_date, ticker, range_pct) "
        "VALUES ('2026-06-30', 'SPX', NULL)")
    conn.commit()

    assert fld.score_daily_outcomes(conn, before_date="2026-07-02") == 0
