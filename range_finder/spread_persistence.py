# =============================================================================
# spread_persistence.py
# Spread plan DB logging, outcome tracking, and pretty-printing.
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone

import yfinance as yf

log = logging.getLogger(__name__)


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


def _fetch_week_ohlc(conn, week_start: str,
                     friday_str: str) -> tuple[float, float, float | None] | None:
    """(high, low, monday_open|None) for one expired Mon-Fri SPX week.

    Source order: weekly_spx first (zero network, and it IS the series the
    HAR model trains on — exactly what PI coverage should be measured
    against), then Tradier /markets/history, then yfinance ^GSPC.

    The weekly_spx row is trusted ONLY when its updated_at postdates the
    scored week's Friday. The Monday cron writes the current week's row at
    ~9:31 with a minutes-old partial bar; it normally gets refreshed the
    NEXT Monday before scoring, but if that refresh fails, scoring against
    the stale row records a Monday-morning "range" for a week that may have
    breached a strike — permanently corrupting outcome and coverage stats
    (a breach week scored full_profit). A stale row falls through to the
    network sources, which fetch the completed daily bars directly.
    """
    try:
        row = conn.execute(
            "SELECT spx_open, spx_high, spx_low, updated_at FROM weekly_spx "
            "WHERE week_start = ?",
            (week_start,),
        ).fetchone()
        if row and row[1] is not None and row[2] is not None:
            updated_at = str(row[3] or "")
            # ISO timestamps compare lexicographically: any update dated
            # after the week's Friday (i.e. Saturday or later) was written
            # from completed bars. The normal next-Monday refresh passes;
            # the partial Monday-of-the-scored-week write does not.
            if updated_at[:10] > friday_str:
                return (float(row[1]), float(row[2]),
                        (float(row[0]) if row[0] else None))
            log.warning(
                f"weekly_spx row for {week_start} was last updated "
                f"{updated_at[:10] or 'unknown'} (needed > {friday_str}) — "
                f"possibly a partial-week bar; fetching completed bars from "
                f"the network instead")
    except Exception as e:
        log.warning(f"weekly_spx lookup failed for {week_start}: {e}")

    try:
        from phase1 import credentials
        token = credentials.tradier_token()
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

    # Convention: touching the short strike counts as a breach (real-world
    # assignment risk at expiry).
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

    # Conservative PnL estimate (actual credit not always available).
    # Cash-settled European condor: the index expires at ONE price, so at
    # most ONE side can finish ITM — the worst realizable loss is a single
    # wing width (minus credit, unknown here), even when BOTH strikes were
    # touched intraweek. The old -2*width for "max_loss" was an impossible
    # loss that inflated every downstream loss statistic.
    if outcome == "full_profit":
        pnl_pts = None
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
