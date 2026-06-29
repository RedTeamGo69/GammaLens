"""
Historical EM snapshot tracking — Postgres only.

DATABASE_URL must be set via Streamlit secrets or as an environment variable.
psycopg2 must be installed. If either is missing the module raises a clear
error at import / first-call time rather than silently degrading.

NOTE (2026-06): the `gex_snapshots` table and its read/write helpers
(save_snapshot / get_history / get_zero_gamma_trend / get_daily_summary) were
removed. The cron wrote one GEX snapshot row per ticker per day, but the GEX
history/trend view that consumed them was dropped in the UI redesign, leaving
the table a dead write. The EM snapshot path below IS still read (the spread
finder's frozen expected-move band), so it remains.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
_logger = logging.getLogger(__name__)


# ── Connection string resolution ──

_pg_conn_str = None

try:
    import streamlit as st
    _pg_conn_str = st.secrets.get("DATABASE_URL", "")
except Exception:
    pass

if not _pg_conn_str:
    _pg_conn_str = os.environ.get("DATABASE_URL", "")


def _require_postgres():
    """Raise a clear error if DATABASE_URL is missing or psycopg2 is unavailable."""
    if not _pg_conn_str:
        raise RuntimeError(
            "DATABASE_URL is not set. This app requires Postgres — set DATABASE_URL "
            "in Streamlit secrets or as an environment variable."
        )
    try:
        import psycopg2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "psycopg2 is not installed. This app requires Postgres — "
            "`pip install psycopg2-binary`."
        ) from e


# ── Postgres helpers ──

def _pg_get_connection():
    _require_postgres()
    import psycopg2
    conn = psycopg2.connect(_pg_conn_str, sslmode="require")
    conn.autocommit = True
    return conn


# ── Public API ──

def get_backend():
    """Legacy compatibility shim. Always returns 'postgres' now."""
    return "postgres"


# One-time schema init flag — `save_em_snapshot` previously issued a
# CREATE TABLE + 4 × ALTER TABLE + 2 × DROP/CREATE INDEX block on EVERY
# call, which cost ~9 round-trips per save (~200-300ms on Neon). This
# flag collapses the DDL to a single first-use invocation so steady-state
# saves are one INSERT.
_em_schema_initialized = False


def _ensure_em_snapshots_schema():
    """Create the em_snapshots table and run legacy migrations once per
    process. Subsequent calls are a no-op guarded by a module-level flag.
    All DDL is idempotent (IF NOT EXISTS / IF EXISTS + DO $$ BEGIN
    EXCEPTION), so multiple workers racing the first call is safe."""
    global _em_schema_initialized
    if _em_schema_initialized:
        return
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS em_snapshots (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL DEFAULT 'SPX',
                em_type TEXT NOT NULL DEFAULT 'daily',
                em_pts REAL,
                em_pct REAL,
                upper_level REAL,
                lower_level REAL,
                anchor_spot REAL,
                straddle_strike REAL,
                captured_at TEXT,
                PRIMARY KEY (ticker, date, em_type)
            )
        """)
        # Migrations for older schemas
        for col, typedef in [
            ("ticker", "TEXT NOT NULL DEFAULT 'SPX'"),
            ("em_pct", "REAL"),
            ("em_type", "TEXT NOT NULL DEFAULT 'daily'"),
            ("anchor_spot", "REAL"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE em_snapshots ADD COLUMN {col} {typedef};
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$
            """)
        # Migrate PK from old (date)-only to (ticker, date, em_type)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'em_snapshots_pkey'
                      AND conrelid = 'em_snapshots'::regclass
                      AND array_length(conkey, 1) = 1
                ) THEN
                    ALTER TABLE em_snapshots DROP CONSTRAINT em_snapshots_pkey;
                    ALTER TABLE em_snapshots ADD PRIMARY KEY (ticker, date, em_type);
                END IF;
            END $$
        """)
        # Drop old index and create new composite one
        cur.execute("DROP INDEX IF EXISTS idx_em_ticker_date")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_em_ticker_date_type
            ON em_snapshots(ticker, date, em_type)
        """)
        _em_schema_initialized = True
    finally:
        conn.close()


def save_em_snapshot(em_data, date_str, ticker="SPX", em_type="daily"):
    """Persist EM snapshot to Postgres so it survives across sessions on the same day."""
    _ensure_em_snapshots_schema()
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        # Compute anchor_spot from EM range midpoint
        upper = em_data.get("upper_level")
        lower = em_data.get("lower_level")
        anchor_spot = round((upper + lower) / 2, 2) if upper is not None and lower is not None else None
        cur.execute(
            """INSERT INTO em_snapshots (date, ticker, em_type, em_pts, em_pct, upper_level, lower_level, anchor_spot, straddle_strike, captured_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (ticker, date, em_type) DO NOTHING""",
            (
                date_str, ticker, em_type,
                em_data.get("expected_move_pts"),
                em_data.get("expected_move_pct"),
                upper, lower, anchor_spot,
                em_data.get("straddle", {}).get("strike"),
                datetime.now(NY_TZ).isoformat(),
            ),
        )
        # Daily EM snapshots are only ever read on their own day, so keep just
        # the latest per ticker — prune older daily rows so the table can't grow
        # unbounded. Weekly/monthly snapshots are retained (they're re-read for
        # historical reconstruction and the OpEx-cycle display).
        if em_type == "daily":
            cur.execute(
                "DELETE FROM em_snapshots "
                "WHERE ticker = %s AND em_type = 'daily' AND date < %s",
                (ticker, date_str),
            )
    finally:
        conn.close()


def get_em_snapshot(date_str, ticker="SPX", em_type="daily"):
    """Retrieve persisted EM snapshot for a given ticker/type/date, if any."""
    try:
        conn = _pg_get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT em_pts, em_pct, upper_level, lower_level, anchor_spot, straddle_strike, captured_at "
                "FROM em_snapshots WHERE date = %s AND ticker = %s AND em_type = %s",
                (date_str, ticker, em_type),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "expected_move_pts": row[0],
                    "expected_move_pct": row[1],
                    "upper_level": row[2],
                    "lower_level": row[3],
                    "anchor_spot": row[4],
                    "straddle": {"strike": row[5]},
                    "captured_at": row[6],
                }
        except Exception:
            # Fallback: older schema without em_type/anchor_spot
            conn = _pg_get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT em_pts, em_pct, upper_level, lower_level, straddle_strike, captured_at "
                "FROM em_snapshots WHERE date = %s AND ticker = %s",
                (date_str, ticker),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "expected_move_pts": row[0],
                    "expected_move_pct": row[1],
                    "upper_level": row[2],
                    "lower_level": row[3],
                    "anchor_spot": None,
                    "straddle": {"strike": row[4]},
                    "captured_at": row[5],
                }
    except Exception:
        pass
    return None


def get_weekly_em_date_key(now):
    """Return Monday's date string for the trading week the weekly EM
    refers to.

    Mon-Fri → this week's Monday (matches the cron's Monday-open capture).
    Sat-Sun → the UPCOMING Monday: the completed week's straddle has
    expired and find_weekly_expiration() already points at next Friday,
    so keying the lookup to last Monday would restore (and chart) a stale
    snapshot anchored at last week's spot. With the forward key the lookup
    simply misses and the UI falls back to the live next-week EM.
    """
    wd = now.weekday()  # 0=Mon
    delta_days = -wd if wd < 5 else (7 - wd)
    if hasattr(now, 'date'):
        monday = (now + timedelta(days=delta_days)).date()
    else:
        monday = now + timedelta(days=delta_days)
    return monday.strftime("%Y-%m-%d")


def get_monthly_em_date_key(now):
    """
    Return the OpEx-cycle key: the Monday following the most recent standard
    3rd-Friday OpEx (strictly before today). Stable across the whole cycle.

    The cycle runs from the Monday-after-OpEx through the NEXT 3rd Friday.
    On the 3rd Friday itself, the day is still in the *old* cycle (its
    standard options settle that morning) — the new cycle begins the Monday
    after. If that Monday is a market holiday the cron's freeze-day check
    will fire on Tuesday instead, but the DB key remains the Monday date so
    save and restore agree.
    """
    from datetime import date as _date, timedelta as _td
    import calendar as _cal

    today = now.date() if hasattr(now, 'date') else now

    # Walk back from today through up to 3 months to find the most recent
    # 3rd Friday that is strictly before today.
    year, month = today.year, today.month
    third_fri = None
    for _ in range(4):
        first_weekday = _cal.weekday(year, month, 1)  # 0=Mon
        first_fri_day = 1 + (4 - first_weekday) % 7
        candidate = _date(year, month, first_fri_day + 14)
        if candidate < today:
            third_fri = candidate
            break
        # Walk back one month
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1

    if third_fri is None:
        # Extreme fallback (shouldn't happen in practice)
        return today.replace(day=1).strftime("%Y-%m-%d")

    # 3rd Friday is always a Friday, so Monday-after = +3 days.
    cycle_open_mon = third_fri + _td(days=3)
    return cycle_open_mon.strftime("%Y-%m-%d")
