"""POP math (phase1.pop): BS delta sanity, vendor-first resolution, condor POP."""
import math

from phase1.pop import bs_delta, condor_pop, resolve_short_delta


# ── bs_delta ──────────────────────────────────────────────────────────────────

def test_bs_delta_atm_call_near_half():
    d = bs_delta(spot=100, strike=100, t_years=0.25, iv=0.20, r=0.04)
    assert 0.50 < d < 0.62  # slightly above 0.5 from the carry drift


def test_bs_delta_atm_put_near_minus_half():
    d = bs_delta(spot=100, strike=100, t_years=0.25, iv=0.20, r=0.04,
                 option_type="put")
    assert -0.50 < d < -0.38


def test_bs_delta_put_call_parity():
    call = bs_delta(100, 95, 0.10, 0.25, r=0.04, option_type="call")
    put = bs_delta(100, 95, 0.10, 0.25, r=0.04, option_type="put")
    assert math.isclose(call - put, 1.0, abs_tol=1e-9)  # q = 0


def test_bs_delta_deep_itm_and_otm():
    assert bs_delta(100, 50, 0.05, 0.20, option_type="call") > 0.99
    assert bs_delta(100, 150, 0.05, 0.20, option_type="call") < 0.01


def test_bs_delta_degenerate_inputs_return_none():
    assert bs_delta(100, 100, 0.0, 0.20) is None   # expired
    assert bs_delta(100, 100, 0.25, 0.0) is None   # no vol
    assert bs_delta(0, 100, 0.25, 0.20) is None    # no spot


# ── resolve_short_delta ───────────────────────────────────────────────────────

def test_resolve_prefers_vendor_delta():
    row = {"strike": 105, "impliedVolatility": 0.20, "vendorDelta": 0.23}
    d, src = resolve_short_delta(row, spot=100, t_years=0.02, option_type="call")
    assert d == 0.23
    assert src == "vendor"


def test_resolve_falls_back_to_bs_from_iv():
    row = {"strike": 105, "impliedVolatility": 0.20, "vendorDelta": 0.0}
    d, src = resolve_short_delta(row, spot=100, t_years=0.02, option_type="call")
    assert src == "bs_iv"
    assert 0.0 < d < 0.5  # OTM call


def test_resolve_unavailable_when_vendor_and_iv_missing():
    row = {"strike": 105, "impliedVolatility": 0.0, "vendorDelta": 0.0}
    d, src = resolve_short_delta(row, spot=100, t_years=0.02, option_type="call")
    assert d is None
    assert src == "unavailable"


def test_resolve_handles_absent_keys():
    d, src = resolve_short_delta({}, spot=100, t_years=0.02, option_type="put")
    assert d is None
    assert src == "unavailable"


# ── condor_pop ────────────────────────────────────────────────────────────────

def test_condor_pop_sixteen_delta_condor():
    assert math.isclose(condor_pop(0.16, -0.16), 0.68, abs_tol=1e-9)


def test_condor_pop_none_when_either_side_missing():
    assert condor_pop(None, -0.16) is None
    assert condor_pop(0.16, None) is None


def test_condor_pop_clamped_to_zero():
    assert condor_pop(0.70, -0.60) == 0.0
