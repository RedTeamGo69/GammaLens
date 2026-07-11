"""
Strategy grouping — pure inference over Public.com fill history.

Public's history rows carry NO order-id and NO open/close indicator, so
strategies are reconstructed from three signals:

  1. **Time clustering** — legs of one order fill within the same instant;
     fills of the same underlying within CLUSTER_WINDOW_SEC belong to one
     execution cluster (a 2s window because multi-fill executions of a
     single order can straddle a second boundary).
  2. **Running position per contract** — from flat, a fill opens (BUY →
     long, SELL → short); same-direction fills scale in; opposite-direction
     fills close toward flat. A fill that would cross THROUGH zero is
     ambiguous (position_flip) and is routed to the review queue, never
     force-grouped.
  3. **Expiry** — most defined-risk legs expire rather than trade out, so
     option groups whose every contract has expired (vs ``as_of``) close
     with reason "expired" and realized P&L = the opening cash flow.

Rolls: a cluster that flattens exactly one group while opening a different
expiration of the same underlying starts a NEW group tagged
``opened_via="roll"`` / ``rolled_from=<old id>`` (two strategies, linked).
Legged-in condors: two same-day, same-expiry credit verticals on opposite
sides of the same underlying merge into one ``opened_via="leg_in"`` group.

Known limitation (documented, not detectable without pre-history data): a
close of a position opened BEFORE the account's first fetched fill looks
identical to a fresh open and will be grouped as one. Public history starts
at the account's first transaction, so this only affects positions carried
into that boundary.

Cash-flow convention: Public's netAmount is the fee-inclusive cash amount;
sign conventions vary by record, so cash is normalized as +|netAmount| for
SELL and −|netAmount| for BUY. Realized P&L per strategy = Σ normalized cash
across all of its fills (premiums ± closes, fees included).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from public_api.occ import OccOption, parse_occ_symbol, underlying_of

GROUPING_VERSION = 1
CLUSTER_WINDOW_SEC = 2.0


# =============================================================================
# DATA SHAPES
# =============================================================================

@dataclass(frozen=True)
class FillLeg:
    txn_id: str
    ts: str                      # ISO timestamp (µs precision from Public)
    symbol: str                  # raw symbol (OCC for options)
    occ: OccOption | None        # parsed option, None for equities
    underlying: str
    side: str                    # "BUY" | "SELL"
    quantity: float              # signed: BUY > 0, SELL < 0
    cash_flow: float             # signed: SELL +, BUY − (fee-inclusive)
    fees: float
    security_type: str           # "OPTION" | "EQUITY" | ...


@dataclass
class StrategyGroup:
    strategy_id: str             # sha256(underlying|sorted opening txn ids)[:16]
    underlying: str
    legs_open: list[FillLeg]
    legs_close: list[FillLeg]
    opened_at: str
    closed_at: str | None
    status: str                  # "open" | "closed"
    close_reason: str | None     # "traded" | "expired" | None
    expiration: str | None       # option expiry (min across legs), None for equity
    dte_at_open: int | None
    realized_pnl: float | None   # None while open
    fees: float
    opened_via: str              # "order" | "roll" | "leg_in"
    rolled_from: str | None
    grouping_version: int = GROUPING_VERSION


@dataclass
class GroupingResult:
    strategies: list[StrategyGroup]
    ambiguous: list[dict]        # {txn_id, symbol, ts, reason, context}
    coverage: float              # grouped TRADE fills ÷ total TRADE fills


# =============================================================================
# FILL NORMALIZATION
# =============================================================================

def _to_leg(txn: dict) -> FillLeg | None:
    if txn.get("type") != "TRADE":
        return None
    symbol = str(txn.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    side = str(txn.get("side") or "").strip().upper()
    if side not in ("BUY", "SELL"):
        return None
    try:
        qty = abs(float(txn.get("quantity") or 0.0))
        net = abs(float(txn.get("netAmount") or 0.0))
        fees = abs(float(txn.get("fees") or 0.0))
    except (TypeError, ValueError):
        return None
    if qty <= 0:
        return None
    occ = parse_occ_symbol(symbol) if txn.get("securityType") == "OPTION" else None
    return FillLeg(
        txn_id=str(txn.get("id") or ""),
        ts=str(txn.get("timestamp") or ""),
        symbol=symbol,
        occ=occ,
        underlying=underlying_of(occ) if occ else symbol,
        side=side,
        quantity=qty if side == "BUY" else -qty,
        cash_flow=net if side == "SELL" else -net,
        fees=fees,
        security_type=str(txn.get("securityType") or ""),
    )


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _clusters(fills: list[FillLeg]) -> list[list[FillLeg]]:
    """Consecutive same-underlying fills within CLUSTER_WINDOW_SEC."""
    out: list[list[FillLeg]] = []
    last_by_und: dict[str, tuple[datetime | None, list[FillLeg]]] = {}
    for leg in fills:
        t = _parse_ts(leg.ts)
        prev = last_by_und.get(leg.underlying)
        if prev is not None:
            prev_t, cluster = prev
            if (t is not None and prev_t is not None
                    and (t - prev_t).total_seconds() <= CLUSTER_WINDOW_SEC):
                cluster.append(leg)
                last_by_und[leg.underlying] = (t, cluster)
                continue
        cluster = [leg]
        out.append(cluster)
        last_by_und[leg.underlying] = (t, cluster)
    out.sort(key=lambda c: c[0].ts)
    return out


# =============================================================================
# INTERNAL MUTABLE GROUP
# =============================================================================

class _Group:
    def __init__(self, underlying: str, opened_via: str = "order",
                 rolled_from: "_Group | None" = None):
        self.underlying = underlying
        self.opened_via = opened_via
        self.rolled_from = rolled_from
        self.legs: list[tuple[FillLeg, str]] = []       # (leg, "open" | "close")
        self.closed_at: str | None = None
        self.close_reason: str | None = None
        self.strategy_id: str = ""

    # -- convenience --------------------------------------------------------
    @property
    def open_legs(self) -> list[FillLeg]:
        return [l for l, phase in self.legs if phase == "open"]

    @property
    def close_legs(self) -> list[FillLeg]:
        return [l for l, phase in self.legs if phase == "close"]

    @property
    def opened_at(self) -> str:
        return min(l.ts for l in self.open_legs) if self.open_legs else ""

    @property
    def open_date(self) -> str:
        return self.opened_at[:10]

    @property
    def contracts(self) -> set[str]:
        return {l.symbol for l, _ in self.legs}

    @property
    def expirations(self) -> set[str]:
        return {l.occ.expiration for l in self.open_legs if l.occ}

    def opening_cash(self) -> float:
        return sum(l.cash_flow for l in self.open_legs)

    def option_types(self) -> set[str]:
        return {l.occ.option_type for l in self.open_legs if l.occ}


# =============================================================================
# THE GROUPER
# =============================================================================

def group_fills(
    transactions: list[dict],
    *,
    as_of: str | None = None,
    version: int = GROUPING_VERSION,
) -> GroupingResult:
    """Infer strategies from raw history transactions (see module docstring).

    ``as_of`` (ISO date): option groups whose every contract expired before
    this date are closed with reason "expired". Pass today's date in
    production; omit in tests that want purely fill-driven state.
    """
    fills = [l for l in (_to_leg(t) for t in transactions) if l is not None]
    fills.sort(key=lambda f: f.ts)
    total = len(fills)

    groups: list[_Group] = []
    ambiguous: list[dict] = []
    pos: dict[str, float] = {}            # contract symbol → signed position
    owner: dict[str, _Group] = {}         # contract symbol → open group

    def _flat(group: _Group) -> bool:
        return all(abs(pos.get(c, 0.0)) < 1e-9 for c in group.contracts)

    for cluster in _clusters(fills):
        opens: list[FillLeg] = []
        closes: list[FillLeg] = []
        for leg in cluster:
            p = pos.get(leg.symbol, 0.0)
            q = leg.quantity
            if abs(p) < 1e-9:
                opens.append(leg)                       # from flat: always an open
            elif (p > 0) == (q > 0):
                opens.append(leg)                       # scale-in
            elif abs(q) <= abs(p) + 1e-9:
                closes.append(leg)                      # (partial) close
            else:
                ambiguous.append({                      # would cross through zero
                    "txn_id": leg.txn_id, "symbol": leg.symbol, "ts": leg.ts,
                    "reason": "position_flip",
                    "context": f"fill qty {q:g} against position {p:g} would "
                               f"cross zero — split open/close intent unclear",
                })
                continue

        # Closes attach to their owning groups first (so a roll's close half
        # resolves before its open half looks for a rolled_from candidate).
        touched: set[int] = set()
        for leg in closes:
            g = owner.get(leg.symbol)
            if g is None:
                ambiguous.append({
                    "txn_id": leg.txn_id, "symbol": leg.symbol, "ts": leg.ts,
                    "reason": "orphan_close",
                    "context": "closing fill with no owning open group",
                })
                continue
            g.legs.append((leg, "close"))
            pos[leg.symbol] = pos.get(leg.symbol, 0.0) + leg.quantity
            touched.add(id(g))
            if _flat(g) and g.closed_at is None:
                g.closed_at = leg.ts
                g.close_reason = "traded"
                for c in g.contracts:
                    owner.pop(c, None)

        if opens:
            # Scale-in: if any opened contract is already owned, attach all
            # opens to that group instead of fabricating a second strategy.
            existing = next((owner[l.symbol] for l in opens if l.symbol in owner), None)
            if existing is not None:
                target = existing
            else:
                # Roll: this cluster flattened exactly one group and opens a
                # DIFFERENT expiration of the same underlying.
                rolled_src = None
                closed_here = [g for g in groups
                               if id(g) in touched and g.closed_at is not None
                               and g.close_reason == "traded"]
                if len(closed_here) == 1:
                    old = closed_here[0]
                    new_exps = {l.occ.expiration for l in opens if l.occ}
                    if new_exps and old.expirations and not (new_exps & old.expirations):
                        rolled_src = old
                target = _Group(
                    underlying=opens[0].underlying,
                    opened_via="roll" if rolled_src is not None else "order",
                    rolled_from=rolled_src,
                )
                groups.append(target)
            for leg in opens:
                target.legs.append((leg, "open"))
                pos[leg.symbol] = pos.get(leg.symbol, 0.0) + leg.quantity
                owner[leg.symbol] = target
                if target.closed_at is not None:      # re-opened via scale-in edge
                    target.closed_at = None
                    target.close_reason = None

    _merge_legged_in_condors(groups, owner)
    if as_of:
        _close_expired(groups, pos, owner, as_of)

    strategies = _finalize(groups)
    grouped_legs = sum(len(g.legs_open) + len(g.legs_close) for g in strategies)
    coverage = grouped_legs / total if total else 1.0
    return GroupingResult(strategies=strategies, ambiguous=ambiguous,
                          coverage=round(coverage, 4))


def _merge_legged_in_condors(groups: list[_Group], owner: dict) -> None:
    """Two same-day, same-expiry credit verticals on opposite sides of one
    underlying → one condor legged in over the day."""
    def _is_credit_vertical(g: _Group) -> bool:
        legs = g.open_legs
        return (len(legs) == 2 and all(l.occ for l in legs)
                and len(g.option_types()) == 1 and len(g.expirations) == 1
                and g.opening_cash() > 0
                and {l.side for l in legs} == {"BUY", "SELL"})

    by_key: dict[tuple, list[_Group]] = {}
    for g in groups:
        if g.opened_via == "order" and _is_credit_vertical(g):
            key = (g.underlying, g.open_date, next(iter(g.expirations)))
            by_key.setdefault(key, []).append(g)

    for candidates in by_key.values():
        calls = [g for g in candidates if g.option_types() == {"call"}]
        puts = [g for g in candidates if g.option_types() == {"put"}]
        if len(calls) == 1 and len(puts) == 1:
            first, second = sorted((calls[0], puts[0]), key=lambda g: g.opened_at)
            first.legs.extend(second.legs)
            first.opened_via = "leg_in"
            # both stay open/closed by position state; remap ownership
            for c in second.contracts:
                if owner.get(c) is second:
                    owner[c] = first
            if first.closed_at and second.closed_at:
                first.closed_at = max(first.closed_at, second.closed_at)
            elif first.closed_at or second.closed_at:
                first.closed_at, first.close_reason = None, None
            groups.remove(second)


def _close_expired(groups: list[_Group], pos: dict, owner: dict, as_of: str) -> None:
    for g in groups:
        if g.closed_at is not None or not g.open_legs:
            continue
        exps = [l.occ.expiration for l, phase in g.legs if l.occ]
        if not exps or len(exps) < len([l for l, _ in g.legs]):
            continue                       # equity or mixed — never expiry-closed
        if max(exps) < as_of:
            g.closed_at = max(exps)
            g.close_reason = "expired"
            for c in g.contracts:
                pos[c] = 0.0
                owner.pop(c, None)


def _finalize(groups: list[_Group]) -> list[StrategyGroup]:
    # ids first (post-merge), so rolled_from can reference them
    for g in groups:
        opening_ids = sorted(l.txn_id for l in g.open_legs)
        g.strategy_id = hashlib.sha256(
            (g.underlying + "|" + ",".join(opening_ids)).encode()
        ).hexdigest()[:16]

    out: list[StrategyGroup] = []
    for g in groups:
        if not g.open_legs:
            continue
        exps = sorted(g.expirations)
        expiration = exps[0] if exps else None
        dte = None
        if expiration and g.open_date:
            try:
                dte = (datetime.fromisoformat(expiration)
                       - datetime.fromisoformat(g.open_date)).days
            except ValueError:
                dte = None
        closed = g.closed_at is not None
        realized = (round(sum(l.cash_flow for l, _ in g.legs), 2)
                    if closed else None)
        out.append(StrategyGroup(
            strategy_id=g.strategy_id,
            underlying=g.underlying,
            legs_open=g.open_legs,
            legs_close=g.close_legs,
            opened_at=g.opened_at,
            closed_at=g.closed_at,
            status="closed" if closed else "open",
            close_reason=g.close_reason,
            expiration=expiration,
            dte_at_open=dte,
            realized_pnl=realized,
            fees=round(sum(l.fees for l, _ in g.legs), 2),
            opened_via=g.opened_via,
            rolled_from=(g.rolled_from.strategy_id
                         if isinstance(g.rolled_from, _Group) else None),
        ))
    out.sort(key=lambda s: s.opened_at)
    return out
