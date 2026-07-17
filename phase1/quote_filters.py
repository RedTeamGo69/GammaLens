from __future__ import annotations


def get_bid(row: dict) -> float:
    return float(row.get("bid", 0.0) or 0.0)


def get_ask(row: dict) -> float:
    return float(row.get("ask", 0.0) or 0.0)


def has_two_sided_quote(row: dict) -> bool:
    bid = get_bid(row)
    ask = get_ask(row)
    return bid > 0 and ask > 0


def is_crossed(row: dict) -> bool:
    """True only when bid > ask (genuinely crossed). Locked quotes (bid == ask) are valid."""
    if not has_two_sided_quote(row):
        return False
    return get_bid(row) > get_ask(row)


def is_locked(row: dict) -> bool:
    """True when bid == ask (tight/locked market). These are valid, high-quality quotes."""
    if not has_two_sided_quote(row):
        return False
    return get_bid(row) == get_ask(row)


def is_crossed_or_locked(row: dict) -> bool:
    """Backward-compatible wrapper. Prefer is_crossed() for filtering."""
    return is_crossed(row) or is_locked(row)


def quote_spread(row: dict) -> float | None:
    if not has_two_sided_quote(row):
        return None
    return get_ask(row) - get_bid(row)


def spread_is_reasonable(row: dict, max_spread: float) -> bool:
    spread = quote_spread(row)
    if spread is None:
        return False
    # Allow zero spread (locked quotes) — they represent tight markets
    return 0 <= spread <= max_spread


def quote_mid(row: dict) -> float | None:
    if not has_two_sided_quote(row):
        return None
    if is_crossed(row):
        return None
    return (get_bid(row) + get_ask(row)) / 2.0


def usable_for_parity(row: dict, max_spread: float) -> bool:
    if not has_two_sided_quote(row):
        return False
    if is_crossed(row):
        return False
    if not spread_is_reasonable(row, max_spread):
        return False
    mid = quote_mid(row)
    return mid is not None and mid > 0


def quote_quality_label(row: dict, max_spread: float) -> str:
    if not has_two_sided_quote(row):
        return "no_two_sided_quote"
    if is_crossed(row):
        return "crossed"
    if is_locked(row):
        return "locked"
    if not spread_is_reasonable(row, max_spread):
        return "wide_spread"
    mid = quote_mid(row)
    if mid is None or mid <= 0:
        return "bad_mid"
    return "usable"


def summarize_quote_quality(rows, max_spread: float) -> dict:
    """
    Returns counts of quote-quality labels for a list of option rows.
    """
    summary = {
        "total": 0,
        "usable": 0,
        "locked": 0,
        "no_two_sided_quote": 0,
        "crossed": 0,
        "crossed_or_locked": 0,  # backward compat aggregate
        "wide_spread": 0,
        "bad_mid": 0,
    }

    for row in rows:
        summary["total"] += 1
        label = quote_quality_label(row, max_spread)
        summary[label] = summary.get(label, 0) + 1

    # Backward compat: crossed_or_locked = crossed + locked
    summary["crossed_or_locked"] = summary["crossed"] + summary["locked"]

    return summary


# ── OCC root separation ──────────────────────────────────────────────────────
#
# Tradier's /markets/options/chains returns EVERY root listed on an
# underlying. On the 3rd Friday that means two contracts per SPX strike —
# the AM-settled monthly (root SPX, last trade Thursday close, settles to
# Friday-open SET) and the PM-settled weekly (root SPXW, trades through
# Friday close). NDX similarly lists NDX + NDXP; corporate-action-adjusted
# equity roots (SPY1 next to SPY) are the same shape. Every strike-keyed
# QUOTE consumer (parity, ATM straddle, spread quote maps) silently
# overwrites one root's row with the other's — mixing two different
# contracts' quotes into one spread or straddle.
#
# GEX aggregation must NOT use this filter: both roots' open interest carry
# dealer gamma, and the engine sums by strike rather than overwriting.

def preferred_root(roots: set[str]) -> "str | None":
    """Pick the root whose quotes strike-keyed consumers should use.

    Rule (ticker-agnostic): the shortest root is the base symbol; prefer the
    PM-settled companion base+"W" (SPXW) if listed, then base+"P" (NDXP),
    else the base itself (plain SPY over adjusted SPY1). Returns None when
    there is nothing to choose between (0 or 1 distinct roots).
    """
    distinct = {r for r in roots if r}
    if len(distinct) <= 1:
        return None
    base = min(distinct, key=len)
    if f"{base}W" in distinct:
        return f"{base}W"
    if f"{base}P" in distinct:
        return f"{base}P"
    return base


def filter_to_preferred_root(
    calls: "list[dict] | None",
    puts: "list[dict] | None",
) -> "tuple[list[dict], list[dict], str | None]":
    """Restrict both sides of a chain to a single OCC root for QUOTE use.

    The root is chosen once from the union of both sides so the call and put
    legs of any derived structure always price off the same contract. Rows
    without a "root" key (older cached entries, test fixtures) never trigger
    filtering — the chain passes through unchanged, preserving pre-root
    behavior. Returns (calls, puts, chosen_root); chosen_root is None when no
    filtering was needed.
    """
    calls = calls or []
    puts = puts or []
    roots = {(row.get("root") or "") for row in calls}
    roots |= {(row.get("root") or "") for row in puts}
    chosen = preferred_root(roots)
    if chosen is None:
        return calls, puts, None
    return (
        [row for row in calls if (row.get("root") or "") == chosen],
        [row for row in puts if (row.get("root") or "") == chosen],
        chosen,
    )
