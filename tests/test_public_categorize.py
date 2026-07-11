"""Categorization of grouped fills (public_api.categorize) — keys must be
byte-for-byte what preflight_engine.category_key yields for the equivalent
proposed trade, or the behavioral gate looks up the wrong bucket."""
import pytest

from public_api.categorize import (
    categorize,
    category_key_for_group,
    opening_pseudo_legs,
)
from public_api.grouping import group_fills
from range_finder.preflight_engine import (
    ProposedLeg,
    ProposedTrade,
    category_key,
)
from tests.test_public_grouping import T0, T2, condor_open, fill


def test_condor_group_matches_preflight_key():
    g = group_fills(condor_open()).strategies[0]
    assert categorize(g) == ("iron_condor", "credit")
    key = category_key_for_group(g)
    assert key == "index|3-7DTE|iron_condor|credit"

    # byte-for-byte parity with the proposed-trade path
    trade = ProposedTrade("XSP", "2026-07-17", 1, (
        ProposedLeg("put", "buy", 745.0), ProposedLeg("put", "sell", 750.0),
        ProposedLeg("call", "sell", 765.0), ProposedLeg("call", "buy", 770.0)))
    assert key == category_key(trade, g.dte_at_open)


def test_single_name_vertical_key():
    txns = [fill("v1", "2026-07-17T14:00:00+00:00", "AMD260731P00100000",
                 "SELL", 1, 150.00),
            fill("v2", "2026-07-17T14:00:00+00:00", "AMD260731P00095000",
                 "BUY", 1, 90.00)]
    g = group_fills(txns).strategies[0]
    assert g.dte_at_open == 14
    assert category_key_for_group(g) == "single_name|8-30DTE|vertical|credit"


def test_zero_dte_index_single_leg_key():
    txns = [fill("s1", "2026-07-13T14:12:00+00:00", "SPXW260713C06000000",
                 "BUY", 1, 220.00)]
    g = group_fills(txns).strategies[0]
    assert g.underlying == "SPX"          # SPXW root normalized
    assert g.dte_at_open == 0
    assert category_key_for_group(g) == "index|0DTE|single_leg|debit"


def test_equity_key():
    txns = [fill("e1", T0, "AAPL", "BUY", 10, 1000.00, sec="EQUITY")]
    g = group_fills(txns).strategies[0]
    assert categorize(g) == ("equity", "debit")
    assert category_key_for_group(g) == "single_name|unknown|equity|debit"


def test_pseudo_legs_aggregate_scale_ins():
    txns = [fill("o1", T0, "XSP260717P00750000", "SELL", 1, 61.00),
            fill("o2", T2, "XSP260717P00750000", "SELL", 1, 58.00)]
    g = group_fills(txns).strategies[0]
    legs = opening_pseudo_legs(g)
    assert len(legs) == 1
    assert legs[0].direction == "sell"
    assert legs[0].strike == pytest.approx(750.0)
