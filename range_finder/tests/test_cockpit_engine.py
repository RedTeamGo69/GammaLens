"""Cockpit decision engine rule matrix (range_finder.cockpit_engine).

Pure-function tests: every gate (VRP, tripwire, events, credit floor,
liquidity, data sufficiency), the outward-only GEX snap, strike rounding,
and the condor pricing math. Fixture geometry: anchor 600, weekly range
forecast 2% → expected-move half-range 6 pts → model shorts 606/594.
"""
import math

import pytest

from range_finder.cockpit_config import CockpitConfig
from range_finder.cockpit_engine import (
    build_leg_quote_map,
    compute_vrp,
    evaluate_cockpit,
    price_condor,
    snap_never_inward,
)
from range_finder.event_calendars import EventInfo

ANCHOR = 600.0
POINT_PCT = 0.02            # 2% weekly range forecast → em_pts = 6
FORECAST = {"point_pct": POINT_PCT, "upper_pct": 0.03, "lower_pct": 0.012}
GOOD_STRADDLE = 7.2         # implied range 2.4% → ratio 1.20 (clears the 1.10 gate)
FRIDAY = "2026-07-17"

FOMC = EventInfo(name="fomc", date="2026-07-15", tier=1, release_time_et="14:00")
PPI = EventInfo(name="ppi", date="2026-07-15", tier=2, release_time_et="08:30")


def decay_chain(lo=560, hi=641, oi=500.0, call_delta=0.12, put_delta=-0.13,
                iv=0.15):
    """Dense integer-strike chain with distance-decaying premium, so any
    strike the engine selects is listed and yields a healthy natural credit."""
    q = {}
    for k in range(lo, hi):
        m = max(0.05, 8.0 * math.exp(-abs(k - ANCHOR) / 15.0))
        bid, ask = round(m - 0.10, 2), round(m + 0.10, 2)
        q[float(k)] = {
            "call_bid": bid, "call_ask": ask, "call_mid": m, "call_oi": oi,
            "call_iv": iv, "call_delta": call_delta,
            "put_bid": bid, "put_ask": ask, "put_mid": m, "put_oi": oi,
            "put_iv": iv, "put_delta": put_delta,
        }
    return q


def run(**overrides):
    kwargs = dict(
        ticker="XSP", anchor=ANCHOR, spot=601.0, friday_exp=FRIDAY,
        forecast=FORECAST, straddle_em_pts=GOOD_STRADDLE, gex_levels=None,
        events=[], chain_quotes=decay_chain(), cfg=CockpitConfig(),
        strike_increment=1, wing_widths=[5, 10],
    )
    kwargs.update(overrides)
    return evaluate_cockpit(**kwargs)


def codes(verdict, severity=None):
    return [r.code for r in verdict.reasons
            if severity is None or r.severity == severity]


# ── happy path ────────────────────────────────────────────────────────────────

def test_trade_path_happy():
    v = run()
    assert v.verdict == "TRADE"
    assert codes(v, "gate") == []
    p = v.proposal
    assert p is not None
    assert (p.call_short, p.call_long, p.put_short, p.put_long) == (606, 611, 594, 589)
    assert p.credit == pytest.approx(2.64)
    assert p.credit_ratio == pytest.approx(0.528)
    assert p.max_profit == pytest.approx(264.0)
    assert p.max_loss == pytest.approx(236.0)
    assert p.breakeven_up == pytest.approx(608.64)
    assert p.breakeven_dn == pytest.approx(591.36)
    assert p.pop == pytest.approx(1 - 0.12 - 0.13)
    assert p.pop_source == "vendor"
    assert v.vrp["ratio"] == pytest.approx(1.20)
    assert "2 × ATM straddle" in v.vrp["convention"]


def test_trade_verdict_always_has_proposal():
    v = run()
    assert v.verdict == "TRADE" and v.proposal is not None


# ── VRP gate + units tripwire ────────────────────────────────────────────────

def test_vrp_thin_skips_but_still_proposes():
    v = run(straddle_em_pts=6.54)         # ratio 1.09 < 1.10
    assert v.verdict == "SKIP"
    assert "vrp_thin" in codes(v, "gate")
    assert v.proposal is not None          # SKIP still shows what was skipped
    assert v.vrp["ratio"] == pytest.approx(1.09)


def test_vrp_clears_threshold():
    v = run(straddle_em_pts=6.72)          # ratio 1.12
    assert v.verdict == "TRADE"
    assert "vrp_thin" not in codes(v)


def test_units_tripwire_low_fires_instead_of_vrp():
    v = run(straddle_em_pts=2.4)           # ratio 0.40 < 0.5 band
    assert v.verdict == "SKIP"
    assert "units_tripwire" in codes(v, "gate")
    assert "vrp_thin" not in codes(v)      # tripwire preempts the VRP verdict


def test_units_tripwire_high():
    v = run(straddle_em_pts=21.0)          # ratio 3.50 > 3.0 band
    assert "units_tripwire" in codes(v, "gate")


def test_tripwire_boundaries_inclusive():
    v_low = run(straddle_em_pts=3.0)       # ratio exactly 0.5 → inside band
    assert "units_tripwire" not in codes(v_low)
    assert "vrp_thin" in codes(v_low, "gate")   # 0.5 < 1.10 still gates VRP

    v_high = run(straddle_em_pts=18.0)     # ratio ~3.0 → inside band, VRP rich
    assert "units_tripwire" not in codes(v_high)
    assert v_high.verdict == "TRADE"


def test_compute_vrp_handles_missing_inputs():
    cfg = CockpitConfig()
    assert compute_vrp(None, ANCHOR, POINT_PCT, cfg)["ratio"] is None
    assert compute_vrp(7.2, None, POINT_PCT, cfg)["ratio"] is None
    assert compute_vrp(7.2, ANCHOR, None, cfg)["ratio"] is None


# ── event policy ─────────────────────────────────────────────────────────────

def test_fomc_skip_policy():
    v = run(events=[FOMC])
    assert v.verdict == "SKIP"
    assert "event_fomc_skip" in codes(v, "gate")
    gate_text = next(r.text for r in v.reasons if r.code == "event_fomc_skip")
    assert "14:00" in gate_text
    assert v.proposal is not None          # proposal still shown at normal k


def test_fomc_widen_policy_moves_strikes_out():
    v = run(events=[FOMC], cfg=CockpitConfig(event_policy="widen"))
    assert v.verdict == "TRADE"
    assert "event_fomc_skip" not in codes(v)
    assert "event_widen" in codes(v, "info")
    p = v.proposal
    # k_eff = 1.25 → base 600 ± 7.5 → rounded away to 608 / 592
    assert p.k_used == pytest.approx(1.25)
    assert (p.call_short, p.put_short) == (608, 592)


def test_tier2_event_warns_but_never_gates():
    v = run(events=[PPI])
    assert v.verdict == "TRADE"
    assert "event_tier2" in codes(v, "warning")
    assert codes(v, "gate") == []


# ── GEX snap: outward-only, tolerance-bounded ────────────────────────────────

def test_snap_outward_within_tolerance():
    # tolerance 0.5% of 600 = 3 pts; walls 2 pts outside the model shorts
    v = run(gex_levels={"call_wall": 608.0, "put_wall": 592.0})
    p = v.proposal
    assert (p.call_short, p.put_short) == (608, 592)
    assert p.call_snapped_to_wall and p.put_snapped_to_wall
    assert "gex_snap" in codes(v, "info")


def test_no_snap_beyond_tolerance():
    v = run(gex_levels={"call_wall": 612.0, "put_wall": 588.0})  # 6 pts > 3-pt tol
    p = v.proposal
    assert (p.call_short, p.put_short) == (606, 594)
    assert not p.call_snapped_to_wall and not p.put_snapped_to_wall


def test_wall_inside_shorts_never_pulls_inward():
    v = run(gex_levels={"call_wall": 604.0, "put_wall": 596.0})  # inside the shorts
    p = v.proposal
    assert (p.call_short, p.put_short) == (606, 594)             # model strikes kept
    assert "wall_inside_shorts" in codes(v, "warning")


def test_snap_never_inward_unit():
    ok = snap_never_inward(606.0, 608.0, "call", ANCHOR, 0.5)
    assert (ok.strike, ok.snapped, ok.inward_wall) == (608.0, True, False)

    far = snap_never_inward(606.0, 612.0, "call", ANCHOR, 0.5)
    assert (far.strike, far.snapped) == (606.0, False)

    inward_call = snap_never_inward(606.0, 604.0, "call", ANCHOR, 0.5)
    assert (inward_call.strike, inward_call.snapped, inward_call.inward_wall) == (606.0, False, True)

    inward_put = snap_never_inward(594.0, 596.0, "put", ANCHOR, 0.5)
    assert (inward_put.strike, inward_put.snapped, inward_put.inward_wall) == (594.0, False, True)

    none_wall = snap_never_inward(606.0, None, "call", ANCHOR, 0.5)
    assert (none_wall.strike, none_wall.snapped, none_wall.note) == (606.0, False, None)


# ── credit floor: SKIP, never tighten ────────────────────────────────────────

def test_credit_floor_skips_without_tightening_strikes():
    flat = {}
    for k in range(560, 641):
        flat[float(k)] = {
            "call_bid": 0.48, "call_ask": 0.50, "call_oi": 500.0,
            "call_iv": 0.15, "call_delta": 0.12,
            "put_bid": 0.48, "put_ask": 0.50, "put_oi": 500.0,
            "put_iv": 0.15, "put_delta": -0.13,
        }
    v = run(chain_quotes=flat)
    assert v.verdict == "SKIP"
    assert "credit_floor" in codes(v, "gate")
    p = v.proposal
    # strikes are the model strikes — proof the engine never chased credit inward
    assert (p.call_short, p.call_long, p.put_short, p.put_long) == (606, 611, 594, 589)
    assert p.credit_ratio < 0.20


# ── liquidity ────────────────────────────────────────────────────────────────

def test_missing_leg_side_gates_insufficient_liquidity():
    # Puts listed only at/above 600 → no put wing below the model short 594
    q = decay_chain()
    for k in list(q):
        if k < 600:
            for key in list(q[k]):
                if key.startswith("put_"):
                    del q[k][key]
    v = run(chain_quotes=q)
    assert v.verdict == "SKIP"
    assert "insufficient_liquidity" in codes(v, "gate")
    assert v.proposal is None


def test_low_oi_and_wide_spread_warn_but_dont_gate():
    q = decay_chain(oi=10.0)               # below the 100 floor everywhere
    v = run(chain_quotes=q)
    assert v.verdict == "TRADE"             # annotation only — gating is Pre-Flight's
    assert "low_oi" in codes(v, "warning")


# ── data sufficiency: every missing input is a named SKIP ────────────────────

def test_missing_anchor_gates():
    v = run(anchor=None)
    assert v.verdict == "SKIP"
    assert "no_anchor" in codes(v, "gate")
    assert v.proposal is None


def test_missing_forecast_gates():
    v = run(forecast=None)
    assert "no_forecast" in codes(v, "gate")
    assert v.proposal is None


def test_missing_straddle_gates_but_still_proposes():
    v = run(straddle_em_pts=None)
    assert v.verdict == "SKIP"
    assert "no_straddle" in codes(v, "gate")
    assert v.vrp["ratio"] is None
    assert v.proposal is not None           # strikes don't need the straddle


def test_missing_chain_gates():
    v = run(chain_quotes={})
    assert "no_chain" in codes(v, "gate")
    assert v.proposal is None


def test_missing_expiration_gates():
    v = run(friday_exp=None)
    assert "no_expiration" in codes(v, "gate")


def test_skip_always_has_a_gate_reason():
    scenarios = [
        run(straddle_em_pts=6.54),
        run(events=[FOMC]),
        run(anchor=None),
        run(chain_quotes={}),
    ]
    for v in scenarios:
        assert v.verdict == "SKIP"
        assert len(codes(v, "gate")) >= 1


# ── strike rounding at SPX-style increments ──────────────────────────────────

def test_strike_rounding_spx_increment():
    anchor = 6000.0
    q = {}
    for k in range(5800, 6205, 5):
        m = max(0.5, 60.0 * math.exp(-abs(k - anchor) / 120.0))
        q[float(k)] = {
            "call_bid": round(m - 0.5, 2), "call_ask": round(m + 0.5, 2),
            "call_oi": 500.0, "call_iv": 0.15, "call_delta": 0.12,
            "put_bid": round(m - 0.5, 2), "put_ask": round(m + 0.5, 2),
            "put_oi": 500.0, "put_iv": 0.15, "put_delta": -0.13,
        }
    v = run(ticker="SPX", anchor=anchor, spot=6001.0,
            forecast={"point_pct": 0.021}, straddle_em_pts=75.0,
            chain_quotes=q, strike_increment=5, wing_widths=[50])
    p = v.proposal
    # em_pts = 63 → call 6063 rounds UP to 6065, put 5937 rounds DOWN to 5935
    assert (p.call_short, p.put_short) == (6065, 5935)
    assert (p.call_long, p.put_long) == (6115, 5885)


# ── pricing + quote-map plumbing ─────────────────────────────────────────────

def test_price_condor_exact_math():
    q = {
        606.0: {"call_bid": 5.20, "call_ask": 5.40, "put_bid": 0.0, "put_ask": 0.1},
        611.0: {"call_bid": 3.80, "call_ask": 4.00},
        594.0: {"put_bid": 5.00, "put_ask": 5.20},
        589.0: {"put_bid": 3.70, "put_ask": 3.90},
    }
    econ = price_condor(606.0, 611.0, 594.0, 589.0, q)
    assert econ["call_credit"] == pytest.approx(1.20)   # 5.20 − 4.00
    assert econ["put_credit"] == pytest.approx(1.10)    # 5.00 − 3.90
    assert econ["credit"] == pytest.approx(2.30)
    assert econ["credit_ratio"] == pytest.approx(2.30 / 5.0)
    assert econ["breakeven_up"] == pytest.approx(608.30)
    assert econ["breakeven_dn"] == pytest.approx(591.70)
    assert econ["max_loss"] == pytest.approx((5.0 - 2.30) * 100)


def test_price_condor_missing_leg_returns_none():
    q = {
        606.0: {"call_bid": 5.20, "call_ask": 5.40},
        611.0: {"call_bid": 3.80, "call_ask": 4.00},
        594.0: {"put_bid": 5.00, "put_ask": 5.20},
        # 589 put missing entirely
    }
    assert price_condor(606.0, 611.0, 594.0, 589.0, q) is None


def test_price_condor_zero_bid_is_a_price_not_a_gap():
    q = {
        606.0: {"call_bid": 0.0, "call_ask": 0.10},
        611.0: {"call_bid": 0.0, "call_ask": 0.05},
        594.0: {"put_bid": 5.00, "put_ask": 5.20},
        589.0: {"put_bid": 3.70, "put_ask": 3.90},
    }
    econ = price_condor(606.0, 611.0, 594.0, 589.0, q)
    assert econ is not None
    assert econ["call_credit"] == 0.0
    assert econ["credit"] == pytest.approx(1.10)


def test_build_leg_quote_map_shape():
    calls = [{"strike": 606, "bid": 5.2, "ask": 5.4, "openInterest": 321,
              "impliedVolatility": 0.15, "vendorDelta": 0.18}]
    puts = [{"strike": 594, "bid": 5.0, "ask": 5.2, "openInterest": 240,
             "impliedVolatility": 0.16, "vendorDelta": -0.17}]
    q = build_leg_quote_map(calls, puts)
    assert q[606.0]["call_bid"] == 5.2
    assert q[606.0]["call_mid"] == pytest.approx(5.3)
    assert q[606.0]["call_oi"] == 321
    assert q[594.0]["put_delta"] == -0.17
    assert "put_bid" not in q[606.0]     # side keys only where the side is listed


# ── POP plumbing ─────────────────────────────────────────────────────────────

def test_pop_mixed_sources_and_unavailable():
    # put vendor delta zeroed → BS fallback from IV → "mixed"
    q = decay_chain(put_delta=0.0)
    v = run(chain_quotes=q, t_years=5 / 252)
    assert v.proposal.pop is not None
    assert v.proposal.pop_source == "mixed"

    # no vendor deltas, no IV → POP unavailable but never gates
    q2 = decay_chain(call_delta=0.0, put_delta=0.0, iv=0.0)
    v2 = run(chain_quotes=q2)
    assert v2.verdict == "TRADE"
    assert v2.proposal.pop is None
    assert v2.proposal.pop_source == "unavailable"


# ── entry guidance: sweet spot, leg-in, Wednesday cutoff ─────────────────────
# Fixture geometry: shorts 606/594 → sweet spot 600; EM 6 → tolerance band
# ±1.5 pts (0.25× EM) around 600.

def test_entry_full_condor_at_sweet_spot():
    v = run(spot=600.0)
    e = v.proposal.entry
    assert e is not None
    assert e.sweet_spot == 600.0
    assert (e.mode, e.lean) == ("full_condor", "balanced")
    assert "entry_full_condor" in codes(v, "info")
    assert "sell the whole condor" in e.note


def test_entry_tolerance_boundary_is_inclusive():
    v = run(spot=601.5)                      # exactly 0.25× EM from the mid
    assert v.proposal.entry.mode == "full_condor"


def test_entry_leg_in_call_lean():
    v = run(spot=603.0)                      # 3 pts above → outside the band
    e = v.proposal.entry
    assert (e.mode, e.lean) == ("leg_in", "call_side")
    assert e.distance_pts == pytest.approx(3.0)
    assert e.distance_em == pytest.approx(0.5)
    assert "CALL side" in e.note and "rich side" in e.note
    assert "entry_leg_in" in codes(v, "info")
    assert v.verdict == "TRADE"              # guidance, never a gate


def test_entry_leg_in_put_lean():
    v = run(spot=597.0)
    e = v.proposal.entry
    assert (e.mode, e.lean) == ("leg_in", "put_side")
    assert "PUT side" in e.note


def test_sweet_spot_follows_snapped_strikes_not_anchor():
    # Call wall snaps the short call 606→608; put side stays 594 → the
    # midpoint moves to 601 (the sweet spot is between the SHORTS).
    v = run(gex_levels={"call_wall": 608.0}, spot=601.0)
    assert v.proposal.call_short == 608
    assert v.proposal.entry.sweet_spot == 601.0


def test_entry_cutoff_gates_thursday():
    v = run(entry_weekday=3)                 # Thursday
    assert v.verdict == "SKIP"
    assert "entry_cutoff" in codes(v, "gate")
    text = next(r.text for r in v.reasons if r.code == "entry_cutoff")
    assert "Thursday" in text and "Wednesday" in text


def test_entry_cutoff_allows_wednesday_and_future_weeks():
    assert "entry_cutoff" not in codes(run(entry_weekday=2))   # Wednesday OK
    assert "entry_cutoff" not in codes(run(entry_weekday=0))   # Monday OK
    assert "entry_cutoff" not in codes(run(entry_weekday=None))  # next-week plan


def test_entry_cutoff_suppresses_entry_advice():
    # "Complete by Wednesday" next to a Thursday closed-window gate is
    # contradictory copy — past the cutoff the advice reason is dropped,
    # while the assessment still rides on the proposal for the chip strip.
    v = run(entry_weekday=3)
    assert "entry_cutoff" in codes(v, "gate")
    assert "entry_full_condor" not in codes(v)
    assert "entry_leg_in" not in codes(v)
    assert v.proposal.entry is not None


def test_side_economics_standalone_numbers():
    v = run()
    cs, ps = v.proposal.call_side, v.proposal.put_side
    # decay chain: each side's own credit is 5.26 − 3.94 = 1.32 on width 5
    assert cs["credit"] == pytest.approx(1.32)
    assert cs["credit_ratio"] == pytest.approx(0.264)
    assert cs["breakeven"] == pytest.approx(606 + 1.32)     # own BE, own credit
    assert cs["max_loss"] == pytest.approx((5 - 1.32) * 100)
    assert ps["breakeven"] == pytest.approx(594 - 1.32)
    assert cs["short_delta"] == pytest.approx(0.12)
    assert ps["short_delta"] == pytest.approx(-0.13)


# ── audit payload ────────────────────────────────────────────────────────────

def test_inputs_snapshot_for_audit():
    v = run(events=[FOMC], gex_levels={"call_wall": 608.0, "put_wall": 592.0,
                                       "zero_gamma": 599.0})
    assert v.inputs["anchor"] == ANCHOR
    assert v.inputs["call_wall"] == 608.0
    assert v.inputs["zero_gamma"] == 599.0
    assert v.inputs["events"][0]["name"] == "fomc"
    assert v.inputs["config"]["vrp_min_ratio"] == pytest.approx(1.10)
    assert v.inputs["k_used"] == v.inputs["k_configured"]  # skip policy: k unchanged
