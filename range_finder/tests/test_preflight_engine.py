"""Pre-Flight gates rule matrix (range_finder.preflight_engine).

Covers the trade taxonomy (category keys shared with the fill categorizer),
each gate's green/yellow/red/na paths — including the load-bearing
"no HAR model at this horizon" refusal to extrapolate — and the roll-up
precedence (red > yellow > green; na never blocks).
"""
import json

from range_finder.cockpit_config import CockpitConfig
from range_finder.preflight_engine import (
    ProposedLeg,
    ProposedTrade,
    category_key,
    check_behavioral,
    check_model,
    check_public_preflight,
    check_structure,
    dte_bucket,
    report_to_row,
    run_preflight,
    structure_of_legs,
    underlying_type,
)

CFG = CockpitConfig()
FRIDAY = "2026-07-17"
ANCHOR = 600.0
FORECAST = {"point_pct": 0.02, "upper_pct": 0.03}   # point band ±6, PI band ±9
EM_PTS = 6.0


def leg(t, d, k):
    return ProposedLeg(option_type=t, direction=d, strike=k)


def condor(call_s=606, call_l=611, put_s=594, put_l=589, ticker="XSP",
           exp=FRIDAY):
    return ProposedTrade(ticker=ticker, expiration=exp, quantity=1, legs=(
        leg("put", "buy", put_l), leg("put", "sell", put_s),
        leg("call", "sell", call_s), leg("call", "buy", call_l)))


def qrow(**kw):
    base = {}
    for side in ("call", "put"):
        bid = kw.get(f"{side}_bid")
        if bid is None:
            continue
        ask = kw.get(f"{side}_ask", bid + 0.2)
        base[f"{side}_bid"] = bid
        base[f"{side}_ask"] = ask
        base[f"{side}_oi"] = kw.get(f"{side}_oi", 400.0)
    return base


GOOD_CHAIN = {
    589.0: qrow(put_bid=3.5, put_ask=3.7),
    590.0: qrow(put_bid=4.0, put_ask=4.2),
    593.0: qrow(put_bid=4.6, put_ask=4.8),
    594.0: qrow(put_bid=5.0, put_ask=5.2),
    606.0: qrow(call_bid=5.0, call_ask=5.2),
    607.0: qrow(call_bid=4.6, call_ask=4.8),
    610.0: qrow(call_bid=4.0, call_ask=4.2),
    611.0: qrow(call_bid=3.5, call_ask=3.7),
}


# ── taxonomy ─────────────────────────────────────────────────────────────────

def test_underlying_types():
    assert underlying_type("SPX") == "index"
    assert underlying_type("XSP") == "index"
    assert underlying_type("QQQ") == "etf"
    assert underlying_type("AMD") == "single_name"
    assert underlying_type("TSLA") == "single_name"   # unknown → single name
    assert underlying_type("RUT") == "index"          # known index not in config


def test_dte_buckets():
    assert dte_bucket(0) == "0DTE"
    assert dte_bucket(2) == "1-2DTE"
    assert dte_bucket(5) == "3-7DTE"
    assert dte_bucket(14) == "8-30DTE"
    assert dte_bucket(45) == "30+DTE"
    assert dte_bucket(None) == "unknown"


def test_structure_classification_table():
    assert structure_of_legs(condor().legs) == ("iron_condor", "credit")
    # call credit = sell lower / buy higher; call debit = the reverse
    assert structure_of_legs((leg("call", "sell", 100), leg("call", "buy", 105))) == ("vertical", "credit")
    assert structure_of_legs((leg("call", "buy", 100), leg("call", "sell", 105))) == ("vertical", "debit")
    # put credit = sell higher / buy lower
    assert structure_of_legs((leg("put", "sell", 100), leg("put", "buy", 95))) == ("vertical", "credit")
    assert structure_of_legs((leg("put", "buy", 100), leg("put", "sell", 95))) == ("vertical", "debit")
    assert structure_of_legs((leg("call", "buy", 100),)) == ("single_leg", "debit")
    assert structure_of_legs((leg("put", "sell", 90),)) == ("single_leg", "credit")
    assert structure_of_legs((leg("call", "sell", 100), leg("put", "sell", 100))) == ("straddle", "credit")
    assert structure_of_legs((leg("call", "buy", 110), leg("put", "buy", 90))) == ("strangle", "debit")
    assert structure_of_legs((leg("call", "buy", 100), leg("call", "buy", 105),
                              leg("put", "sell", 95))) == ("other_multi", "unknown")


def test_category_keys():
    single = ProposedTrade("SPX", "2026-07-11", 1, (leg("call", "buy", 6000),))
    assert category_key(single, 0) == "index|0DTE|single_leg|debit"
    assert category_key(condor(), 5) == "index|3-7DTE|iron_condor|credit"
    vert = ProposedTrade("AMD", "2026-07-25", 1,
                         (leg("put", "sell", 100), leg("put", "buy", 95)))
    assert category_key(vert, 14) == "single_name|8-30DTE|vertical|credit"


# ── behavioral gate ──────────────────────────────────────────────────────────

def test_behavioral_na_without_history():
    assert check_behavioral("x", None, CFG).status == "na"


def test_behavioral_small_sample_yellow():
    g = check_behavioral("index|0DTE|single_leg|debit",
                         {"n_trades": 3, "total_pnl": -120.0, "win_rate": 0.33}, CFG)
    assert g.status == "yellow"
    assert any("Small sample" in r for r in g.reasons)
    assert any("net -$120" in r for r in g.reasons)


def test_behavioral_negative_expectancy_red():
    g = check_behavioral("index|0DTE|single_leg|debit",
                         {"n_trades": 15, "total_pnl": -430.0, "win_rate": 0.4}, CFG)
    assert g.status == "red"
    assert any("net -$430" in r for r in g.reasons)


def test_behavioral_positive_green():
    g = check_behavioral("index|3-7DTE|iron_condor|credit",
                         {"n_trades": 15, "total_pnl": 350.0, "win_rate": 0.73}, CFG)
    assert g.status == "green"


# ── model gate ───────────────────────────────────────────────────────────────

def test_model_never_extrapolates_off_the_weekly_horizon():
    trade = condor(exp="2026-08-21")   # 30+ DTE
    g = check_model(trade, FORECAST, ANCHOR, FRIDAY, CFG)
    assert g.status == "na"
    assert any("no HAR model at this horizon" in r for r in g.reasons)
    assert "legs" not in g.details       # proves no band math ran


def test_model_na_when_forecast_missing():
    g = check_model(condor(), None, ANCHOR, FRIDAY, CFG)
    assert g.status == "na"


def test_model_green_outside_pi_band():
    g = check_model(condor(call_s=610, put_s=590), FORECAST, ANCHOR, FRIDAY, CFG)
    assert g.status == "green"          # PI band is ±9 → 610/590 outside


def test_model_yellow_between_point_and_pi():
    g = check_model(condor(call_s=607, put_s=593), FORECAST, ANCHOR, FRIDAY, CFG)
    assert g.status == "yellow"


def test_model_red_inside_point_band():
    g = check_model(condor(call_s=604, put_s=590), FORECAST, ANCHOR, FRIDAY, CFG)
    assert g.status == "red"            # 604 < 606 point band edge
    assert any("INSIDE" in r for r in g.reasons)


def test_model_bands_follow_the_forecast_side_share():
    """Pre-Flight must judge strikes against the SAME per-side convention the
    Cockpit placed them with, or it reports the Cockpit's own proposal as
    sitting inside the point band every week (the /2 bug, fixed 2026-08-03)."""
    fc = dict(FORECAST, side_share_q=0.8357)
    # point band widens 600×0.02×0.8357 = ±10.03, PI band ±15.04
    strict = check_model(condor(call_s=607, put_s=593), fc, ANCHOR, FRIDAY, CFG)
    assert strict.status == "red"        # was "yellow" under the half-split
    wide = check_model(condor(call_s=616, put_s=584), fc, ANCHOR, FRIDAY, CFG)
    assert wide.status == "green"        # outside the widened PI band
    # and the legacy half-split is untouched when no share is supplied
    assert check_model(condor(call_s=607, put_s=593), FORECAST, ANCHOR,
                       FRIDAY, CFG).status == "yellow"


def test_model_na_for_pure_debit_trade():
    trade = ProposedTrade("XSP", FRIDAY, 1, (leg("call", "buy", 610),))
    g = check_model(trade, FORECAST, ANCHOR, FRIDAY, CFG)
    assert g.status == "na"


# ── structure gate ───────────────────────────────────────────────────────────

def test_structure_green_on_healthy_condor():
    g = check_structure(condor(), GOOD_CHAIN, EM_PTS, ANCHOR, CFG)
    assert g.status == "green"
    assert g.details["credit_ratio"] == 0.52


def test_structure_credit_floor_red():
    flat = {k: qrow(call_bid=0.48, call_ask=0.50, put_bid=0.48, put_ask=0.50)
            for k in (589.0, 594.0, 606.0, 611.0)}
    g = check_structure(condor(), flat, EM_PTS, ANCHOR, CFG)
    assert g.status == "red"
    assert any("Credit/width" in r for r in g.reasons)


def test_structure_breakeven_red_inside_one_em():
    chain = {596.0: qrow(put_bid=3.0, put_ask=3.2),
             591.0: qrow(put_bid=1.9, put_ask=2.0)}
    trade = ProposedTrade("XSP", FRIDAY, 1,
                          (leg("put", "sell", 596), leg("put", "buy", 591)))
    g = check_structure(trade, chain, EM_PTS, ANCHOR, CFG)
    assert g.status == "red"            # BE 595 → 5 pts = 0.83× EM
    assert any("inside 1×" in r for r in g.reasons)


def test_structure_breakeven_yellow_under_1_25_em():
    chain = {594.0: qrow(put_bid=3.0, put_ask=3.2),
             589.0: qrow(put_bid=1.9, put_ask=2.0)}
    trade = ProposedTrade("XSP", FRIDAY, 1,
                          (leg("put", "sell", 594), leg("put", "buy", 589)))
    g = check_structure(trade, chain, EM_PTS, ANCHOR, CFG)
    assert g.status == "yellow"         # BE 593 → 7 pts = 1.17× EM


def test_structure_bid_ask_thresholds():
    yellow_chain = {606.0: qrow(call_bid=4.7, call_ask=5.3)}       # 12% of mid
    trade = ProposedTrade("XSP", FRIDAY, 1, (leg("call", "sell", 606),))
    assert check_structure(trade, yellow_chain, None, None, CFG).status == "yellow"

    red_chain = {606.0: qrow(call_bid=2.0, call_ask=2.55)}          # ~24% of mid
    assert check_structure(trade, red_chain, None, None, CFG).status == "red"


def test_structure_oi_thresholds():
    trade = ProposedTrade("XSP", FRIDAY, 1, (leg("call", "sell", 606),))
    low = {606.0: qrow(call_bid=5.0, call_ask=5.2, call_oi=50.0)}
    assert check_structure(trade, low, None, None, CFG).status == "yellow"
    zero = {606.0: qrow(call_bid=5.0, call_ask=5.2, call_oi=0.0)}
    assert check_structure(trade, zero, None, None, CFG).status == "red"


def test_structure_em_missing_skips_breakeven_subcheck():
    g = check_structure(condor(), GOOD_CHAIN, None, ANCHOR, CFG)
    assert g.status == "green"
    assert any("skipped" in r for r in g.reasons)


def test_structure_na_without_chain():
    assert check_structure(condor(), None, EM_PTS, ANCHOR, CFG).status == "na"


# ── public gate ──────────────────────────────────────────────────────────────

def test_public_na_and_red_and_green():
    assert check_public_preflight(None).status == "na"
    rejected = check_public_preflight({"status": "rejected",
                                       "messages": ["insufficient buying power"]})
    assert rejected.status == "red"
    ok = check_public_preflight({"status": "ok", "buying_power_impact": 415.0,
                                 "estimated_credit": 85.0})
    assert ok.status == "green"
    assert any("415" in r for r in ok.reasons)


# ── roll-up + persistence row ────────────────────────────────────────────────

def _run(trade=None, **kw):
    args = dict(stats=None, forecast=FORECAST, anchor=ANCHOR,
                week_friday_exp=FRIDAY, chain_quotes=GOOD_CHAIN, em_pts=EM_PTS,
                public_response=None, cfg=CFG, dte=5,
                checked_at="2026-07-13T14:00:00+00:00")
    args.update(kw)
    return run_preflight(trade or condor(), **args)


def test_rollup_all_na_is_green():
    r = _run(trade=condor(exp="2026-08-21"), chain_quotes=None)
    assert r.verdict == "GREEN"
    assert all(g.status == "na" for g in r.gates)


def test_rollup_red_beats_yellow():
    r = _run(stats={"n_trades": 15, "total_pnl": -430.0, "win_rate": 0.4},
             trade=condor(call_s=607, put_s=593))      # model yellow
    assert r.verdict == "RED"


def test_rollup_yellow_without_red():
    r = _run(trade=condor(call_s=607, put_s=593))
    assert r.verdict == "YELLOW"


def test_rollup_green_path():
    r = _run(trade=condor(call_s=610, put_s=590))
    assert r.verdict == "GREEN"
    assert r.category == "index|3-7DTE|iron_condor|credit"


def test_report_to_row_shape():
    r = _run()
    row = report_to_row(r, "2026-07-13")
    assert row["verdict"] == r.verdict
    assert row["input_hash"] == r.input_hash
    assert row["category"] == r.category
    assert row["in_weekly_window"] == 1
    assert json.loads(row["trade_json"])["ticker"] == "XSP"
    assert json.loads(row["model_json"])["name"] == "model"

    off = _run(trade=condor(exp="2026-08-21"))
    assert report_to_row(off, "2026-07-13")["in_weekly_window"] == 0


def test_input_hash_stable_and_sensitive():
    assert _run().input_hash == _run().input_hash
    assert _run().input_hash != _run(trade=condor(call_s=610, put_s=590)).input_hash
