# =============================================================================
# spread_persistence.py
# Spread plan DB logging, outcome tracking, and pretty-printing.
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone

import yfinance as yf

log = logging.getLogger(__name__)


def init_spread_log_table(conn) -> None:
    """Ensure spread_log table exists.
    Now handled by db.init_all_tables() — kept for backwards compatibility."""
    pass  # Tables created in db.init_all_tables()


def log_spread_plan(
    conn,
    plan,
    wing_width_used: int = None,
    ticker: str = "SPX",
    model_name: str = None,
    lower_pct: float = None,
) -> None:
    """Persist a SpreadPlan to spread_log.

    ``model_name`` and ``lower_pct`` feed the PI-coverage calibration audit
    (range_finder/calibration.py): the spec that produced the forecast and
    its lower interval bound, so per-spec / two-sided coverage can be
    computed once outcomes are scored.
    """
    width = wing_width_used or plan.recommended_width

    call = next((s for s in plan.call_spreads if s.wing_width == width), None)
    put  = next((s for s in plan.put_spreads  if s.wing_width == width), None)

    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO spread_log (
            week_start, ticker, model_name, generated_at,
            spx_ref_close, point_pct, lower_pct, upper_pct, effective_range_pct,
            call_short, call_long, put_short, put_long,
            wing_width_used, buffer_pct, event_count, gex_flag,
            warnings, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(week_start, ticker) DO UPDATE SET
            model_name          = excluded.model_name,
            generated_at        = excluded.generated_at,
            spx_ref_close       = excluded.spx_ref_close,
            point_pct           = excluded.point_pct,
            lower_pct           = excluded.lower_pct,
            upper_pct           = excluded.upper_pct,
            effective_range_pct = excluded.effective_range_pct,
            call_short          = excluded.call_short,
            call_long           = excluded.call_long,
            put_short           = excluded.put_short,
            put_long            = excluded.put_long,
            wing_width_used     = excluded.wing_width_used,
            buffer_pct          = excluded.buffer_pct,
            event_count         = excluded.event_count,
            gex_flag            = excluded.gex_flag,
            warnings            = excluded.warnings,
            updated_at          = excluded.updated_at
    """, (
        plan.week_start,
        ticker,
        model_name,
        plan.generated_at,
        plan.spx_ref_close,
        plan.point_pct,
        lower_pct,
        plan.upper_pct,
        plan.effective_range_pct,
        call.short_strike if call else None,
        call.long_strike  if call else None,
        put.short_strike  if put  else None,
        put.long_strike   if put  else None,
        width,
        plan.buffer_pct,
        plan.event_count,
        plan.gex_flag,
        " | ".join(plan.warnings),
        now,
    ))
    conn.commit()
    log.info(f"Spread plan logged for {plan.week_start} ({ticker}, {model_name})")


def update_outcome(
    conn,
    week_start: str,
    actual_high: float,
    actual_low: float,
    credit_received: float = None,
    ticker: str = "SPX",
) -> str:
    """Fill in the actual outcome after the week expires."""
    row = conn.execute(
        "SELECT call_short, put_short, wing_width_used FROM spread_log "
        "WHERE week_start = ? AND ticker = ?",
        (week_start, ticker)
    ).fetchone()

    if not row:
        log.warning(f"No spread_log entry for {week_start} ({ticker})")
        return "not_found"

    call_short, put_short, width = row
    spx_ref = conn.execute(
        "SELECT spx_ref_close FROM spread_log WHERE week_start = ? AND ticker = ?",
        (week_start, ticker)
    ).fetchone()[0]

    actual_range_pct = (actual_high - actual_low) / spx_ref if spx_ref else None
    call_breached    = int(actual_high >= call_short) if call_short else 0
    put_breached     = int(actual_low  <= put_short)  if put_short  else 0

    if call_breached or put_breached:
        if credit_received and width:
            pnl_pts = credit_received - width
            outcome = "partial_loss" if pnl_pts > -width * 0.5 else "full_loss"
        else:
            outcome = "full_loss"
            pnl_pts = None
    else:
        outcome = "full_profit"
        pnl_pts = credit_received if credit_received else None

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE spread_log SET
            actual_high      = ?,
            actual_low       = ?,
            actual_range_pct = ?,
            call_breached    = ?,
            put_breached     = ?,
            outcome          = ?,
            pnl_pts          = ?,
            updated_at       = ?
        WHERE week_start = ? AND ticker = ?
    """, (
        actual_high, actual_low, actual_range_pct,
        call_breached, put_breached, outcome,
        pnl_pts, now,
        week_start, ticker,
    ))
    conn.commit()
    log.info(f"Outcome updated for {week_start} ({ticker}): {outcome}")
    return outcome


def _fetch_week_ohlc(conn, week_start: str,
                     friday_str: str) -> tuple[float, float, float | None] | None:
    """(high, low, monday_open|None) for one expired Mon-Fri SPX week.

    Source order: weekly_spx first (zero network, and it IS the series the
    HAR model trains on — exactly what PI coverage should be measured
    against), then Tradier /markets/history, then yfinance ^GSPC.
    """
    try:
        row = conn.execute(
            "SELECT spx_open, spx_high, spx_low FROM weekly_spx "
            "WHERE week_start = ?",
            (week_start,),
        ).fetchone()
        if row and row[1] is not None and row[2] is not None:
            return float(row[1]), float(row[2]), (float(row[0]) if row[0] else None)
    except Exception as e:
        log.warning(f"weekly_spx lookup failed for {week_start}: {e}")

    try:
        from range_finder.data_collector import _tradier_token
        token = _tradier_token()
        if token:
            from phase1.data_client import TradierDataClient
            days = TradierDataClient(token).get_history(
                "SPX", interval="daily", start=week_start, end=friday_str,
            )
            if days:
                highs = [float(d["high"]) for d in days]
                lows = [float(d["low"]) for d in days]
                opens = [float(d["open"]) for d in days]
                return max(highs), min(lows), (opens[0] if opens else None)
            log.warning(f"Tradier returned no SPX bars for week {week_start} — "
                        "falling back to yfinance")
    except Exception as e:
        log.warning(f"Tradier week fetch failed for {week_start}: {e} — "
                    "falling back to yfinance")

    # yfinance end date is exclusive — add 1 day
    fetch_end = (datetime.strptime(friday_str, "%Y-%m-%d")
                 + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download("^GSPC", start=week_start, end=fetch_end,
                      interval="1d", progress=False, timeout=30)
    if raw.empty:
        return None
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)
    first_open = float(raw["Open"].iloc[0]) if "Open" in raw.columns else None
    return float(raw["High"].max()), float(raw["Low"].min()), first_open


def update_expiration_outcome(week_start: str, conn, ticker: str = "SPX") -> str:
    """Auto-fetch the expired week's OHLC and update breach/outcome in spread_log.

    Uses daily bars for the expired week to compute actual_high, actual_low,
    then compares against call_short / put_short to determine the outcome.
    Scoring bars are SPX-only for now (the cron logs SPX plans only) —
    non-SPX rows are skipped rather than mis-scored against SPX prices.
    """
    if ticker != "SPX":
        log.warning(f"update_expiration_outcome: {ticker} scoring not wired yet "
                    "(SPX-only) — skipping")
        return "unsupported_ticker"

    row = conn.execute(
        "SELECT call_short, put_short, spx_ref_close, wing_width_used "
        "FROM spread_log WHERE week_start = ? AND ticker = ?",
        (week_start, ticker),
    ).fetchone()

    if not row:
        log.warning(f"No spread_log entry for {week_start} ({ticker})")
        return "not_found"

    call_short, put_short, spx_ref, width = row

    # Compute the week's Mon-Fri date range
    monday = datetime.strptime(week_start, "%Y-%m-%d")
    friday = monday + timedelta(days=4)
    friday_str = friday.strftime("%Y-%m-%d")

    log.info(f"Fetching SPX daily OHLC for {week_start} to {friday_str}")
    ohlc = _fetch_week_ohlc(conn, week_start, friday_str)
    if ohlc is None:
        log.error(f"No source returned data for week {week_start}")
        return "no_data"

    actual_high, actual_low, week_open = ohlc
    # Normalize by the week's Monday open when we have it — that matches the
    # model's range_pct = (high-low)/open target definition, so PI-coverage
    # comparisons against the forecast bounds are apples-to-apples. Fall
    # back to the plan's reference price (pre-migration behavior).
    denom = week_open or spx_ref
    actual_range_pct = (actual_high - actual_low) / denom if denom else None

    # Convention: touching the short strike counts as a breach (matches the
    # manual update_outcome() path and real-world assignment risk at expiry).
    call_breached = int(actual_high >= call_short) if call_short else 0
    put_breached = int(actual_low <= put_short) if put_short else 0

    # Four-valued outcome
    if call_breached and put_breached:
        outcome = "max_loss"
    elif call_breached:
        outcome = "call_loss"
    elif put_breached:
        outcome = "put_loss"
    else:
        outcome = "full_profit"

    # Conservative PnL estimate (actual credit not always available)
    if outcome == "full_profit":
        pnl_pts = None
    elif outcome == "max_loss":
        pnl_pts = -2 * width if width else None
    else:
        pnl_pts = -width if width else None

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE spread_log SET
            actual_high      = ?,
            actual_low       = ?,
            actual_range_pct = ?,
            call_breached    = ?,
            put_breached     = ?,
            outcome          = ?,
            pnl_pts          = ?,
            updated_at       = ?
        WHERE week_start = ? AND ticker = ?
    """, (
        actual_high, actual_low, actual_range_pct,
        call_breached, put_breached, outcome,
        pnl_pts, now,
        week_start, ticker,
    ))
    conn.commit()
    log.info(f"Expiration outcome for {week_start} ({ticker}): {outcome} "
             f"(high={actual_high:.2f}, low={actual_low:.2f})")
    return outcome


def get_spread_log(conn) -> list[dict]:
    """Read all spread_log rows as a list of dicts, newest first."""
    cur = conn.execute("SELECT * FROM spread_log ORDER BY week_start DESC")
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def print_spread_plan(plan) -> None:
    """Pretty-print the full spread plan to console."""
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  WEEKLY SPREAD PLAN  --  Week of {plan.week_start}")
    print(f"  Generated: {plan.generated_at[:19]} UTC")
    print(sep)

    print(f"\n  REFERENCE")
    print(f"    SPX Friday close  : {plan.spx_ref_close:>10,.2f}")
    if plan.spx_ref_open:
        print(f"    SPX Monday open   : {plan.spx_ref_open:>10,.2f}")
    print(f"    VIX implied range : {plan.vix_implied_pct*100:>9.2f}%")

    print(f"\n  FORECAST  ({plan.confidence_level}% CI)")
    print(f"    Point estimate    : +/-{plan.point_pct/2*100:.2f}%  "
          f"({plan.point_pct*100:.2f}% total)")
    print(f"    PI upper bound    :  {plan.upper_pct*100:.2f}%  total range")
    print(f"    Buffer applied    : +{plan.buffer_pct*100:.3f}%  ({plan.buffer_pts:.1f} pts)")
    print(f"    Effective range   :  {plan.effective_range_pct*100:.2f}%  total")
    print(f"    Effective upper   : {plan.effective_upper_px:>10,.2f}")
    print(f"    Effective lower   : {plan.effective_lower_px:>10,.2f}")

    print(f"\n  CONTEXT")
    print(f"    Events this week  : {plan.event_count}  "
          f"(FOMC={plan.has_fomc} CPI={plan.has_cpi} NFP={plan.has_nfp} OPEX={plan.has_opex})")
    print(f"    GEX regime        : {plan.gex_regime}")
    print(f"    Recommended width : {plan.recommended_width} pts")

    if plan.warnings:
        print(f"\n  WARNINGS")
        for w in plan.warnings:
            print(f"    {w}")

    print(f"\n{sep}\n")
