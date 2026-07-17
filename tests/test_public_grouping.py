"""Strategy grouping on synthetic Public.com fills (public_api.grouping).

No network, no DB — raw transaction dicts in, StrategyGroups out. The
load-bearing properties: same-second clustering, running-position open/close
inference, expiry closure (most defined-risk legs expire, not trade out),
roll linking, legged-in condor merging, position-flip ambiguity routed to
review (never force-grouped), and deterministic strategy ids.
"""
import pytest

from public_api.grouping import GROUPING_VERSION, group_fills
from public_api.occ import parse_occ_symbol, underlying_of


def fill(txn_id, ts, symbol, side, qty, net, sec="OPTION", fees=0.02):
    return {
        "id": txn_id, "timestamp": ts, "type": "TRADE", "subType": "TRADE",
        "accountNumber": "TESTACCT", "symbol": symbol, "securityType": sec,
        "side": side, "description": "",
        "quantity": str(-abs(qty) if side == "SELL" else abs(qty)),
        "netAmount": str(net), "principalAmount": str(abs(float(net))),
        "fees": str(fees),
    }


T0 = "2026-07-13T13:31:00.100000+00:00"
T0B = "2026-07-13T13:31:01.000000+00:00"     # 0.9s later — same cluster
T1 = "2026-07-13T17:31:00.000000+00:00"      # same day, hours later
T2 = "2026-07-15T14:00:00.000000+00:00"      # two days later

P745 = "XSP260717P00745000"
P750 = "XSP260717P00750000"
C765 = "XSP260717C00765000"
C770 = "XSP260717C00770000"


def condor_open(ts=T0):
    return [
        fill("t1", ts, P750, "SELL", 1, 90.00),
        fill("t2", ts, P745, "BUY", 1, 55.00),
        fill("t3", ts, C765, "SELL", 1, 80.00),
        fill("t4", ts, C770, "BUY", 1, 50.00),
    ]


# ── OCC parsing ───────────────────────────────────────────────────────────────

def test_occ_parse_basics():
    occ = parse_occ_symbol("SOFI260417P00016000")
    assert (occ.root, occ.expiration, occ.option_type, occ.strike) == \
        ("SOFI", "2026-04-17", "put", 16.0)
    occ2 = parse_occ_symbol(C765)
    assert (occ2.root, occ2.expiration, occ2.option_type, occ2.strike) == \
        ("XSP", "2026-07-17", "call", 765.0)


def test_occ_weekly_root_normalization_and_equity():
    occ = parse_occ_symbol("SPXW260713C06000000")
    assert occ.root == "SPXW"
    assert underlying_of(occ) == "SPX"
    assert parse_occ_symbol("AAPL") is None
    assert parse_occ_symbol(None) is None


# ── clustering + structure ────────────────────────────────────────────────────

def test_same_second_condor_is_one_group():
    res = group_fills(condor_open())
    assert len(res.strategies) == 1
    g = res.strategies[0]
    assert g.underlying == "XSP"
    assert len(g.legs_open) == 4
    assert g.status == "open"
    assert g.expiration == "2026-07-17"
    assert g.dte_at_open == 4
    assert res.coverage == 1.0
    assert res.ambiguous == []


def test_one_second_boundary_stays_one_cluster():
    txns = [fill("a", T0, P750, "SELL", 1, 90.00),
            fill("b", T0B, P745, "BUY", 1, 55.00)]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    assert len(res.strategies[0].legs_open) == 2


def test_money_movements_are_ignored():
    txns = condor_open() + [{
        "id": "d1", "timestamp": T0, "type": "MONEY_MOVEMENT",
        "subType": "DEPOSIT", "netAmount": "200.00", "direction": "INCOMING",
        "accountNumber": "TESTACCT", "description": "Deposit",
    }]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    assert res.coverage == 1.0


# ── open → close lifecycle ────────────────────────────────────────────────────

def test_open_then_close_realizes_pnl():
    txns = [fill("o1", T0, P750, "SELL", 1, 61.00),
            fill("c1", T2, P750, "BUY", 1, 35.00)]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    g = res.strategies[0]
    assert g.status == "closed"
    assert g.close_reason == "traded"
    assert g.realized_pnl == pytest.approx(61.00 - 35.00)
    assert len(g.legs_close) == 1


def test_partial_close_stays_open():
    txns = [fill("o1", T0, P750, "SELL", 2, 122.00),
            fill("c1", T2, P750, "BUY", 1, 35.00)]
    res = group_fills(txns)
    g = res.strategies[0]
    assert g.status == "open"
    assert g.realized_pnl is None


def test_expired_worthless_closes_with_opening_credit():
    res = group_fills(condor_open(), as_of="2026-07-20")
    g = res.strategies[0]
    assert g.status == "closed"
    assert g.close_reason == "expired"
    assert g.closed_at == "2026-07-17"
    assert g.realized_pnl == pytest.approx(90 - 55 + 80 - 50)   # credit kept


def test_before_expiry_group_stays_open():
    res = group_fills(condor_open(), as_of="2026-07-16")
    assert res.strategies[0].status == "open"


# ── ambiguity: never force-grouped ───────────────────────────────────────────

def test_position_flip_goes_to_review_queue():
    txns = [fill("o1", T0, P750, "SELL", 1, 61.00),
            fill("x1", T2, P750, "BUY", 2, 70.00)]     # would cross through zero
    res = group_fills(txns)
    assert len(res.ambiguous) == 1
    assert res.ambiguous[0]["reason"] == "position_flip"
    assert res.ambiguous[0]["txn_id"] == "x1"
    g = res.strategies[0]
    assert g.status == "open"                          # untouched by the flip
    assert res.coverage == pytest.approx(0.5)


# ── rolls ─────────────────────────────────────────────────────────────────────

def test_roll_links_two_strategies():
    jun = "SPY260619P00520000"
    jul = "SPY260717P00520000"
    txns = [
        fill("o1", T0, jun, "SELL", 1, 200.00),
        # one cluster: buy back June, sell July
        fill("r1", T2, jun, "BUY", 1, 120.00),
        fill("r2", T2, jul, "SELL", 1, 210.00),
    ]
    res = group_fills(txns)
    assert len(res.strategies) == 2
    old, new = res.strategies
    assert old.status == "closed" and old.close_reason == "traded"
    assert old.realized_pnl == pytest.approx(80.0)
    assert new.opened_via == "roll"
    assert new.rolled_from == old.strategy_id
    assert new.status == "open"


# ── legged-in condors ─────────────────────────────────────────────────────────

def test_legged_in_condor_merges_same_day_opposite_verticals():
    txns = [
        fill("p1", T0, P750, "SELL", 1, 90.00),
        fill("p2", T0, P745, "BUY", 1, 55.00),
        fill("c1", T1, C765, "SELL", 1, 80.00),
        fill("c2", T1, C770, "BUY", 1, 50.00),
    ]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    g = res.strategies[0]
    assert g.opened_via == "leg_in"
    assert len(g.legs_open) == 4


def test_multi_day_same_week_verticals_merge_as_leg_in():
    """The actual trading workflow: put spread sold Monday, call side legged
    in Wednesday (the Cockpit's Mon-Wed entry cutoff), same Friday expiry.
    grouping v2 merges these; v1's same-day window left them as two phantom
    verticals that never matched the position actually managed as a condor."""
    txns = [
        fill("p1", T0, P750, "SELL", 1, 90.00),    # Monday
        fill("p2", T0, P745, "BUY", 1, 55.00),
        fill("c1", T2, C765, "SELL", 1, 80.00),    # Wednesday, same week
        fill("c2", T2, C770, "BUY", 1, 50.00),
    ]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    g = res.strategies[0]
    assert g.opened_via == "leg_in"
    assert len(g.legs_open) == 4
    assert g.opened_at.startswith("2026-07-13")    # condor dates from first leg


def test_cross_week_verticals_stay_separate():
    """Opposite-side verticals in DIFFERENT weeks never merge, even ahead of
    a shared expiration — that's two positions, not a leg-in."""
    txns = [
        fill("p1", "2026-07-10T14:00:00.000000+00:00", P750, "SELL", 1, 90.00),  # prior Fri
        fill("p2", "2026-07-10T14:00:00.000000+00:00", P745, "BUY", 1, 55.00),
        fill("c1", T0, C765, "SELL", 1, 80.00),    # Monday of the next week
        fill("c2", T0, C770, "BUY", 1, 50.00),
    ]
    res = group_fills(txns)
    assert len(res.strategies) == 2
    assert all(g.opened_via == "order" for g in res.strategies)


# ── equities ─────────────────────────────────────────────────────────────────

def test_equity_round_trip():
    txns = [fill("e1", T0, "AAPL", "BUY", 10, 1000.00, sec="EQUITY"),
            fill("e2", T2, "AAPL", "SELL", 10, 1100.00, sec="EQUITY")]
    res = group_fills(txns, as_of="2026-08-01")
    g = res.strategies[0]
    assert g.underlying == "AAPL"
    assert g.status == "closed" and g.close_reason == "traded"
    assert g.realized_pnl == pytest.approx(100.0)
    assert g.expiration is None and g.dte_at_open is None


# ── determinism / bookkeeping ─────────────────────────────────────────────────

def test_strategy_ids_deterministic():
    a = group_fills(condor_open())
    b = group_fills(condor_open())
    assert a.strategies[0].strategy_id == b.strategies[0].strategy_id
    assert a.strategies[0].grouping_version == GROUPING_VERSION


def test_scale_in_attaches_to_existing_group():
    txns = [fill("o1", T0, P750, "SELL", 1, 61.00),
            fill("o2", T2, P750, "SELL", 1, 58.00)]
    res = group_fills(txns)
    assert len(res.strategies) == 1
    assert len(res.strategies[0].legs_open) == 2


# ── position-flip tracker reset (audit D46) ──────────────────────────────────

def test_position_flip_resets_tracker_so_later_fills_classify_correctly():
    """After a flip fill lands in review, the contract's tracked position must
    advance to reality (p+q). Hold +1 long, SELL 3 (crosses zero → flip/review,
    true book now -2), then SELL 1 more. With the reset, that SELL is a
    same-direction scale-in of the short (an OPEN) — without it, the stale +1
    book would mis-read the SELL 1 as a close. The flip fill carries the
    underlying key the review-queue writer persists."""
    txns = [
        fill("a", T0, C765, "BUY", 1, 50.00),                 # +1 long call
        fill("b", T1, C765, "SELL", 3, 240.00),               # -3 → crosses 0: flip
        fill("c", T2, C765, "SELL", 1, 80.00),                # more short vs true -2
    ]
    res = group_fills(txns)
    flips = [a for a in res.ambiguous if a["reason"] == "position_flip"]
    assert len(flips) == 1
    assert flips[0]["txn_id"] == "b"
    assert flips[0]["underlying"] == "XSP"      # underlying key present (D50)
    # Fill 'c' classified against the corrected -2 book as a same-direction
    # open — NOT flagged as an orphan close from a stale +1 book.
    assert not any(a["txn_id"] == "c" for a in res.ambiguous)
