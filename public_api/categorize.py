"""
Strategy categorization — grouped fills → the SAME category keys the
Pre-Flight tab computes for proposed trades.

The taxonomy (underlying type, DTE bucket, structure, direction) lives in
range_finder.preflight_engine and is imported, not duplicated — if the two
ever produced different keys for the same trade shape, the behavioral gate
would silently look up the wrong bucket.
"""
from __future__ import annotations

from range_finder.preflight_engine import (
    ProposedLeg,
    category_key as _category_key,
    ProposedTrade,
    structure_of_legs,
)
from public_api.grouping import StrategyGroup


def opening_pseudo_legs(group: StrategyGroup) -> tuple[ProposedLeg, ...]:
    """Aggregate a group's opening fills into one pseudo-leg per contract
    (direction from the net signed quantity), shaped for structure_of_legs."""
    by_contract: dict[str, dict] = {}
    for leg in group.legs_open:
        slot = by_contract.setdefault(leg.symbol, {
            "occ": leg.occ, "qty": 0.0, "security_type": leg.security_type})
        slot["qty"] += leg.quantity
    out = []
    for slot in by_contract.values():
        if abs(slot["qty"]) < 1e-9:
            continue
        occ = slot["occ"]
        out.append(ProposedLeg(
            option_type=occ.option_type if occ else "equity",
            direction="buy" if slot["qty"] > 0 else "sell",
            strike=occ.strike if occ else 0.0,
        ))
    return tuple(out)


def categorize(group: StrategyGroup) -> tuple[str, str]:
    """(structure, direction) for a grouped strategy."""
    return structure_of_legs(opening_pseudo_legs(group))


def category_key_for_group(group: StrategyGroup) -> str:
    """Full 4-dim key, e.g. ``index|3-7DTE|iron_condor|credit`` — byte-for-byte
    the key preflight_engine.category_key produces for the equivalent
    proposed trade."""
    trade = ProposedTrade(
        ticker=group.underlying,
        expiration=group.expiration or "",
        quantity=1,
        legs=opening_pseudo_legs(group),
    )
    return _category_key(trade, group.dte_at_open)
