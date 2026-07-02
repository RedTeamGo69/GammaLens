# =============================================================================
# forecast_log_daily.py
# 0DTE forecast-vs-realized logging — the lean, READ-BACK successor to the
# removed spread_log_daily.
#
# The old table logged full plans (strikes, wings, credits) that nothing ever
# read, so it was deleted. This one logs exactly what the calibration audit
# (calibration.daily_pi_coverage) consumes and nothing else: the forecast
# bounds the daily HAR produced for a session, and — filled in the next
# morning from daily_spx already in the DB, zero network — the realized
# range. One deterministic row per market day, written by the daily cron.
# =============================================================================

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def log_daily_forecast(conn, session_date: str, model_name: str,
                       forecast: dict, ticker: str = "SPX",
                       vix1d_close: float = None) -> None:
    """Upsert one session's forecast bounds into forecast_log_daily.

    ``forecast`` is a forecast_next_session/forecast_next_week dict —
    only the bounds and reference are persisted (no strikes by design).
    Idempotent: a rerun of the cron the same morning overwrites in place.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO forecast_log_daily (
            session_date, ticker, model_name,
            point_pct, lower_pct, upper_pct,
            spx_ref, vix1d_close, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_date, ticker, model_name) DO UPDATE SET
            point_pct    = excluded.point_pct,
            lower_pct    = excluded.lower_pct,
            upper_pct    = excluded.upper_pct,
            spx_ref      = excluded.spx_ref,
            vix1d_close  = excluded.vix1d_close,
            generated_at = excluded.generated_at
    """, (
        session_date, ticker, model_name,
        forecast.get("point_pct"), forecast.get("lower_pct"),
        forecast.get("upper_pct"), forecast.get("spx_ref_close"),
        vix1d_close, now,
    ))
    conn.commit()
    log.info(f"forecast_log_daily: logged {session_date} ({ticker}/{model_name}) "
             f"point={forecast.get('point_pct')} upper={forecast.get('upper_pct')}")


def score_daily_outcomes(conn, before_date: str, ticker: str = "SPX") -> int:
    """Fill actual_range_pct for unscored rows whose session has completed.

    One correlated-subquery UPDATE joining daily_spx.range_pct already in
    the DB — zero network calls. Only sessions strictly before
    ``before_date`` (today) are scored, so a partial in-progress bar can
    never be recorded as an outcome; yesterday's bar is final by the time
    the next cron run lands here. Returns rows scored.
    """
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute("""
        UPDATE forecast_log_daily
        SET actual_range_pct = (
                SELECT d.range_pct FROM daily_spx d
                WHERE d.session_date = forecast_log_daily.session_date
                  AND d.ticker = forecast_log_daily.ticker
            ),
            scored_at = ?
        WHERE ticker = ?
          AND actual_range_pct IS NULL
          AND session_date < ?
          AND EXISTS (
                SELECT 1 FROM daily_spx d
                WHERE d.session_date = forecast_log_daily.session_date
                  AND d.ticker = forecast_log_daily.ticker
                  AND d.range_pct IS NOT NULL
            )
    """, (now, ticker, before_date))
    conn.commit()
    scored = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    if scored:
        log.info(f"forecast_log_daily: scored {scored} completed session(s)")
    return scored
