"""Strike snapping + tier construction guards.

Regression tests for the chain-edge bug where a corrupt reference price
(e.g. SPX ref of 100 against a ~7,400 chain) made `_snap_to_chain_strike`
"snap" an un-listable put target to the chain's LOWEST listed strike —
fabricating deep-ITM spreads that the UI then rendered as if they were
valid far-OTM credit spreads.
"""
import pandas as pd  # noqa: F401  (matches repo test conventions)

from range_finder.spread_levels import (
    _snap_to_chain_strike,
    build_spread_side,
    build_spread_tiers,
    build_spread_plan,
)


def _chain(strikes):
    """Minimal chain-quotes dict with both sides quoted at every strike."""
    return {
        float(k): {
            "call_bid": 1.0, "call_ask": 1.2,
            "put_bid": 1.0, "put_ask": 1.2,
        }
        for k in strikes
    }


SPX_CHAIN = _chain(range(7000, 7801, 50))  # listed 7000..7800
SPX_CHAIN_FLOOR = 7000.0
SPX_CHAIN_CAP = 7800.0


# ─────────────────────────────────────────────────────────────────────────────
# _snap_to_chain_strike
# ─────────────────────────────────────────────────────────────────────────────

def test_snap_up_normal():
    assert _snap_to_chain_strike(7510, SPX_CHAIN, "call", "up") == 7550


def test_snap_down_normal():
    assert _snap_to_chain_strike(7290, SPX_CHAIN, "put", "down") == 7250


def test_snap_exact_match_kept():
    assert _snap_to_chain_strike(7300, SPX_CHAIN, "put", "down") == 7300


def test_snap_down_below_chain_floor_returns_target():
    # A put target below every listed strike must NOT come back as the
    # chain's bottom strike (that strike is closer to the money than the
    # caller asked for). The unsnapped target lets the spread builder
    # find no quotes and emit nothing — the honest outcome.
    assert _snap_to_chain_strike(97, SPX_CHAIN, "put", "down") == 97


def test_snap_up_above_chain_cap_returns_target():
    assert _snap_to_chain_strike(9000, SPX_CHAIN, "call", "up") == 9000


def test_snap_empty_chain_returns_target():
    assert _snap_to_chain_strike(7510, {}, "call", "up") == 7510


# ─────────────────────────────────────────────────────────────────────────────
# build_spread_side
# ─────────────────────────────────────────────────────────────────────────────

def test_spread_side_unlisted_short_yields_no_spreads():
    spreads = build_spread_side(
        "put", short_strike=95, wing_widths=[50, 100],
        spx_ref=7400.0, vix=18.0, chain_quotes=SPX_CHAIN,
    )
    assert spreads == []


def test_spread_side_listed_short_builds_rows():
    spreads = build_spread_side(
        "call", short_strike=7550.0, wing_widths=[50, 100],
        spx_ref=7400.0, vix=18.0, chain_quotes=SPX_CHAIN,
    )
    assert [s.long_strike for s in spreads] == [7600.0, 7650.0]
    for s in spreads:
        assert s.short_strike == 7550.0
        assert s.max_loss == round((s.wing_width - s.estimated_credit) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# build_spread_tiers — corrupt reference must not fabricate chain-edge puts
# ─────────────────────────────────────────────────────────────────────────────

def _forecast(point=0.0336, lower=0.0209, upper=0.0540):
    return {
        "point_pct": point,
        "lower_pct": lower,
        "upper_pct": upper,
        "vix_implied_pct": 0.0476,
        "model_vs_vix": point - 0.0476,
        "confidence_level": 80,
        "spx_ref_close": None,  # filled per test via build_spread_plan
    }


def _plan_for(ref, chain=None):
    fc = _forecast()
    fc["spx_ref_close"] = ref
    return build_spread_plan(
        forecast=fc, feature_row=None, week_start="2026-06-15",
        vix_level=20.0, ticker="SPX", chain_quotes=chain,
    )


def test_tiers_sane_reference_produces_otm_strikes():
    ref = 7400.0
    plan = _plan_for(ref, SPX_CHAIN)
    tiers = build_spread_tiers(
        forecast=_forecast(), plan=plan, spx_ref=ref, vix_level=20.0,
        chain_quotes=SPX_CHAIN, ticker="SPX",
    )
    assert len(tiers) == 4
    for t in tiers:
        assert t.call_short > ref
        assert t.put_short < ref
        assert SPX_CHAIN_FLOOR <= t.put_short <= SPX_CHAIN_CAP
        assert SPX_CHAIN_FLOOR <= t.call_short <= SPX_CHAIN_CAP


def test_tiers_corrupt_low_ref_does_not_fabricate_chain_floor_puts():
    # The original bug: ref=100 against a 7,000+ chain put the model put
    # target at ~97, which the old snap turned into the chain's lowest
    # listed strike — and the UI showed a "credit spread" short the
    # bottom of the chain. Post-fix the put side must stay unsnapped
    # (no listed strike at/below target) and build no spreads at all.
    ref = 100.0
    plan = _plan_for(ref, SPX_CHAIN)
    tiers = build_spread_tiers(
        forecast=_forecast(), plan=plan, spx_ref=ref, vix_level=20.0,
        chain_quotes=SPX_CHAIN, ticker="SPX",
    )
    for t in tiers:
        assert t.put_short < SPX_CHAIN_FLOOR, (
            f"{t.label}: put short {t.put_short} snapped INTO the chain"
        )
        assert t.put_spreads == []
        assert t.model_put_spreads == []


def test_tiers_corrupt_ref_suppresses_model_ladder_on_em_floored_side():
    # With ref=100 the model call target (~103) legally snaps UP to the
    # chain's bottom strike (200) and the EM floor rescues the FINAL call
    # short — but the "model strikes (before EM floor)" ladder would then
    # showcase a deep-ITM 200-short spread. A pre-floor call short below
    # the EM band can only mean the reference is corrupt, so that ladder
    # must be suppressed (final EM-floored strikes stay).
    ref = 100.0
    plan = _plan_for(ref, SPX_CHAIN)
    em = {"upper_level": 7549.0, "lower_level": 7360.0}
    tiers = build_spread_tiers(
        forecast=_forecast(), plan=plan, spx_ref=ref, vix_level=20.0,
        chain_quotes=SPX_CHAIN, ticker="SPX", weekly_em=em,
    )
    for t in tiers:
        assert t.call_short >= 7549.0          # EM floor still applied
        assert t.model_call_short is None      # garbage ladder suppressed
        assert t.model_call_spreads == []
        assert t.put_spreads == []             # honest-empty put side


def test_tiers_em_floor_moves_only_inside_strikes():
    ref = 7400.0
    plan = _plan_for(ref, SPX_CHAIN)
    em = {"upper_level": 7549.0, "lower_level": 7251.0}
    tiers = build_spread_tiers(
        forecast=_forecast(), plan=plan, spx_ref=ref, vix_level=20.0,
        chain_quotes=SPX_CHAIN, ticker="SPX", weekly_em=em,
    )
    for t in tiers:
        assert t.call_short >= 7549.0
        assert t.put_short <= 7251.0
        # model_* strikes are recorded only when the floor actually moved them
        if t.model_call_short is not None:
            assert t.model_call_short < 7549.0
        if t.model_put_short is not None:
            assert t.model_put_short > 7251.0


# ─────────────────────────────────────────────────────────────────────────────
# Nominal increment vs real chain grid — the AMD export-vs-screen mismatch
# ─────────────────────────────────────────────────────────────────────────────

def _amd_forecast():
    # ~Effective 12.4% range on AMD; mirrors the reported screenshot.
    return {
        "point_pct": 0.0748, "lower_pct": 0.0461, "upper_pct": 0.1213,
        "vix_implied_pct": 0.0373, "model_vs_vix": 0.0748 - 0.0373,
        "confidence_level": 80, "spx_ref_close": 522.84,
    }


# AMD lists a $2.5 grid far OTM: 556 is NOT listed, 557.5 is; 490 is on-grid.
AMD_CHAIN = _chain(round(450 + 2.5 * i, 2) for i in range(int((600 - 450) / 2.5) + 1))


def test_amd_export_chain_snaps_call_to_listed_strike():
    # The reported bug: with no chain, AMD's nominal strike_increment=1 rounds
    # the Effective call short to 556 (not a listed AMD strike); the live chain
    # snaps it to the tradeable 557.5 — exactly the screen vs export gap. Now
    # the export passes a chain, so both land on 557.5. The put side (490) is
    # already on the grid, so it agrees either way.
    ref = 522.84
    fc = _amd_forecast()
    plan = build_spread_plan(
        forecast=fc, feature_row=None, week_start="2026-06-29",
        vix_level=40.0, ticker="AMD", chain_quotes=None,
    )
    nochain = build_spread_tiers(
        forecast=fc, plan=plan, spx_ref=ref, vix_level=40.0,
        chain_quotes=None, ticker="AMD",
    )
    chained = build_spread_tiers(
        forecast=fc, plan=plan, spx_ref=ref, vix_level=40.0,
        chain_quotes=AMD_CHAIN, ticker="AMD",
    )
    eff_nc = next(t for t in nochain if t.label.startswith("Effective"))
    eff_c = next(t for t in chained if t.label.startswith("Effective"))

    assert eff_nc.call_short == 556.0      # nominal increment rounding
    assert eff_c.call_short == 557.5       # snapped to the real listed strike
    assert eff_nc.put_short == eff_c.put_short == 490.0  # already on the grid
