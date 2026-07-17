from phase1.quote_filters import (
    has_two_sided_quote,
    is_crossed_or_locked,
    is_crossed,
    is_locked,
    quote_spread,
    quote_mid,
    usable_for_parity,
)


def test_good_quote_is_usable():
    row = {"bid": 10.0, "ask": 10.5}
    assert has_two_sided_quote(row) is True
    assert is_crossed_or_locked(row) is False
    assert quote_spread(row) == 0.5
    assert quote_mid(row) == 10.25
    assert usable_for_parity(row, max_spread=2.0) is True


def test_crossed_quote_is_not_usable():
    row = {"bid": 10.5, "ask": 10.0}
    assert has_two_sided_quote(row) is True
    assert is_crossed(row) is True
    assert is_locked(row) is False
    assert is_crossed_or_locked(row) is True
    assert quote_mid(row) is None
    assert usable_for_parity(row, max_spread=2.0) is False


def test_locked_quote_is_usable():
    """Locked quotes (bid == ask) are tight markets and should be usable."""
    row = {"bid": 10.0, "ask": 10.0}
    assert has_two_sided_quote(row) is True
    assert is_crossed(row) is False
    assert is_locked(row) is True
    assert is_crossed_or_locked(row) is True  # backward compat still True
    assert quote_spread(row) == 0.0
    assert quote_mid(row) == 10.0
    assert usable_for_parity(row, max_spread=2.0) is True


def test_wide_spread_quote_is_not_usable():
    row = {"bid": 10.0, "ask": 13.5}
    assert usable_for_parity(row, max_spread=2.0) is False


# ── OCC root separation (SPX vs SPXW on 3rd Fridays) ─────────────────────────

from phase1.quote_filters import preferred_root, filter_to_preferred_root


def test_preferred_root_picks_pm_weekly_over_am_monthly():
    assert preferred_root({"SPX", "SPXW"}) == "SPXW"
    assert preferred_root({"NDX", "NDXP"}) == "NDXP"


def test_preferred_root_picks_plain_over_adjusted():
    # Corporate-action-adjusted roots (SPY1) lose to the standard contract.
    assert preferred_root({"SPY", "SPY1"}) == "SPY"


def test_preferred_root_noop_on_single_or_empty():
    assert preferred_root({"XSP"}) is None
    assert preferred_root(set()) is None
    assert preferred_root({""}) is None


def test_filter_to_preferred_root_mixed_third_friday_chain():
    """The classic 3rd-Friday hazard: a stale zero-bid AM-settled SPX row and
    a live PM-settled SPXW row at the same strike. The filter must keep ONLY
    the SPXW rows on both sides so strike-keyed maps never interleave the
    two contracts."""
    calls = [
        {"strike": 6000, "bid": 0.0, "ask": 0.0, "root": "SPX"},    # stale AM
        {"strike": 6000, "bid": 12.0, "ask": 12.5, "root": "SPXW"}, # live PM
        {"strike": 6005, "bid": 9.0, "ask": 9.6, "root": "SPXW"},
    ]
    puts = [
        {"strike": 6000, "bid": 11.0, "ask": 11.4, "root": "SPXW"},
        {"strike": 6000, "bid": 0.0, "ask": 55.0, "root": "SPX"},
    ]
    fc, fp, root = filter_to_preferred_root(calls, puts)
    assert root == "SPXW"
    assert all(r["root"] == "SPXW" for r in fc + fp)
    assert len(fc) == 2 and len(fp) == 1
    # The live quote survived; the stale AM row is gone.
    assert fc[0]["bid"] == 12.0


def test_filter_to_preferred_root_passthrough_without_root_keys():
    """Legacy cached entries / fixtures without a "root" key must pass
    through untouched (pre-root behavior preserved)."""
    calls = [{"strike": 6000, "bid": 1.0, "ask": 1.2}]
    puts = [{"strike": 6000, "bid": 1.1, "ask": 1.3}]
    fc, fp, root = filter_to_preferred_root(calls, puts)
    assert root is None
    assert fc == calls and fp == puts


def test_find_atm_straddle_ignores_stale_am_root(monkeypatch=None):
    """Behavioral end-to-end: with mixed roots, the straddle must price off
    the SPXW pair, not a chimera of SPX call + SPXW put."""
    from phase1.expected_move import find_atm_straddle
    calls = [
        {"strike": 6000, "bid": 30.0, "ask": 90.0, "root": "SPX"},   # junk-wide AM
        {"strike": 6000, "bid": 12.0, "ask": 12.5, "root": "SPXW"},
    ]
    puts = [
        {"strike": 6000, "bid": 11.0, "ask": 11.4, "root": "SPXW"},
        {"strike": 6000, "bid": 40.0, "ask": 41.0, "root": "SPX"},   # AM mid ≈ 40.5
    ]
    out = find_atm_straddle(calls, puts, spot=6001.0)
    assert out is not None
    assert out["strike"] == 6000
    # SPXW mids: 12.25 + 11.20 = 23.45. A root-mixed result would be ~52.75
    # (SPX call overwrite) or ~52.9 (SPX put overwrite).
    assert abs(out["straddle_price"] - 23.45) < 0.02
