from datetime import datetime

import pandas as pd

import phase1.gex_engine as gex_engine
from phase1.config import NY_TZ


def test_bs_gamma_vec_returns_expected_shape():
    out = gex_engine.bs_gamma_vec(
        S_arr=[5000, 5005],
        K_arr=[5000, 5010],
        T_arr=[1/365, 1/365],
        r=0.04,
        sigma_arr=[0.20, 0.22],
    )
    assert out.shape == (2, 2)


def test_get_gamma_regime_text_positive():
    info = gex_engine.get_gamma_regime_text(spot=5050, zero_gamma=5000)
    assert info["regime"] == "Positive Gamma"


def test_get_gamma_regime_text_negative():
    info = gex_engine.get_gamma_regime_text(spot=4950, zero_gamma=5000)
    assert info["regime"] == "Negative Gamma"


def test_find_key_levels_empty_df_returns_spot():
    empty = pd.DataFrame(columns=["strike", "net_gex"])
    levels = gex_engine.find_key_levels(empty, spot=5000)
    assert levels["call_wall"] == 5000
    assert levels["put_wall"] == 5000
    assert levels["zero_gamma"] == 5000


def test_calculate_all_basic_fake_client():
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            return {
                "status": "ok",
                "calls": [
                    {
                        "strike": 5000,
                        "openInterest": 100,
                        "impliedVolatility": 0.20,
                        "vendorGamma": 0.0,
                        "bid": 10.0,
                        "ask": 10.5,
                        "mid": 10.25,
                    }
                ],
                "puts": [
                    {
                        "strike": 5000,
                        "openInterest": 100,
                        "impliedVolatility": 0.20,
                        "vendorGamma": 0.0,
                        "bid": 10.0,
                        "ask": 10.5,
                        "mid": 10.25,
                    }
                ],
                "error": None,
            }

    client = FakeClient()

    # Pin `now` to a date before the 2026-03-20 expiration so the
    # expired-exp filter in calculate_all doesn't drop it when the test
    # is run on a wall clock that's past March 2026.
    fake_now = datetime(2026, 3, 10, 10, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client=client,
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert not gex_df.empty
    assert stats["used_option_count"] == 2
    assert len(all_options) == 2
    assert not strike_support_df.empty
    assert not expiration_support_df.empty
    # Volume amplification diagnostics: this fixture has OI=100, volume=0
    # on both legs, so size == OI everywhere → amplification ratio is 1.0
    # and no strikes are volume-dominated.
    assert stats["vol_amplification_ratio"] == 1.0
    assert stats["vol_dominated_strike_count"] == 0
    assert stats["vol_dominated_pct"] == 0.0


def test_calculate_all_flags_volume_amplification():
    """When today's volume dwarfs yesterday's OI on most strikes, the
    amplification ratio should exceed 1.0 and vol_dominated_strike_count
    should reflect the number of volume-dominated strikes."""
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            # OI = 10, volume = 500 — volume dominates by 50× on both legs
            return {
                "status": "ok",
                "calls": [{
                    "strike": 5000,
                    "openInterest": 10,
                    "volume": 500,
                    "impliedVolatility": 0.20,
                    "vendorGamma": 0.0,
                    "bid": 10.0, "ask": 10.5, "mid": 10.25,
                }],
                "puts": [{
                    "strike": 5000,
                    "openInterest": 10,
                    "volume": 500,
                    "impliedVolatility": 0.20,
                    "vendorGamma": 0.0,
                    "bid": 10.0, "ask": 10.5, "mid": 10.25,
                }],
                "error": None,
            }

    fake_now = datetime(2026, 3, 10, 10, 0, tzinfo=NY_TZ)
    _, stats, _, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )
    assert stats["vol_amplification_ratio"] == 50.0
    assert stats["vol_dominated_strike_count"] == 2
    assert stats["vol_dominated_pct"] == 1.0


def test_zero_gamma_sweep_details_flags_fallback_when_no_crossing():
    details = gex_engine.zero_gamma_sweep_details(
        all_options=[(5000, 100, 0.20, 1, 1/365)],
        spot=5000,
        r=0.04,
    )

    assert details["is_true_crossing"] is False
    assert details["zero_gamma_type"] == "Fallback node"
    assert details["method"] == "min_abs_fallback"


def test_zero_gamma_sweep_details_detects_true_crossing(monkeypatch):
    import numpy as np

    def fake_sweep(_all_options, prices, _r, r_curve=None):
        # _sweep_gex_at_prices returns total GEX (already Spot²-scaled).
        # r_curve is accepted but ignored — this test pins the crossing
        # location, not the rate path.
        return np.array([float(p - 100.0) for p in prices])

    import phase1.zero_gamma as zero_gamma_mod
    monkeypatch.setattr(zero_gamma_mod, "_sweep_gex_at_prices", fake_sweep)

    details = gex_engine.zero_gamma_sweep_details(
        all_options=[(1, 1, 1, 1, 1)],
        spot=95.0,
        r=0.04,
    )

    assert details["is_true_crossing"] is True
    assert details["zero_gamma_type"] == "True crossing"
    assert details["method"].startswith("crossing")

def test_calculate_all_iv_stats_fall_back_when_first_exp_expired():
    """When the nearest selected expiration has already settled, the
    call_iv / put_iv stats must come from the first LIVE expiration
    instead of silently reading 0.0 (the old behavior keyed the IV
    sample on target_exps[0] even when that exp was skipped)."""
    def _opt(iv):
        return {
            "strike": 5000,
            "openInterest": 100,
            "impliedVolatility": iv,
            "vendorGamma": 0.0,
            "bid": 10.0,
            "ask": 10.5,
            "mid": 10.25,
        }

    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            # Expired Wednesday chain carries a poison IV that must NOT
            # leak into stats; the live Friday chain carries 15%/17%.
            if exp == "2026-03-18":
                return {"status": "ok", "calls": [_opt(0.99)], "puts": [_opt(0.99)], "error": None}
            return {"status": "ok", "calls": [_opt(0.15)], "puts": [_opt(0.17)], "error": None}

    # Thursday 10am ET — the 2026-03-18 (Wed) expiration settled yesterday.
    fake_now = datetime(2026, 3, 19, 10, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-18", "2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert stats["expired_exp_count"] == 1
    assert abs(stats["call_iv"] - 15.0) < 1e-9
    assert abs(stats["put_iv"] - 17.0) < 1e-9


def test_calculate_all_after_hours_all_expirations_expired():
    """0DTE selected after the close: every expiration is dropped by the
    expired-exp guard. calculate_all must return an EMPTY-but-schema'd
    gex_df (plus coherent stats) instead of raising KeyError('net_gex')
    on the column-less frame — that crash masked the friendly 'all
    expirations have settled' notice in the app."""
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            raise AssertionError("expired expirations must be skipped before any chain fetch")

    # 8 PM ET on expiration day — the 2026-03-20 session settled at 4 PM.
    fake_now = datetime(2026, 3, 20, 20, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert gex_df.empty
    assert "net_gex" in gex_df.columns  # schema present even when empty
    assert stats["expired_exp_count"] == 1
    assert stats["used_option_count"] == 0
    assert stats["net_gex"] == 0.0
    assert all_options == []

    # The downstream pipeline (mirroring fetch_all_data) must also cope.
    levels = gex_engine.find_key_levels(gex_df, 5000, all_options=all_options, r=0.04)
    assert levels["zero_gamma"] == 5000
    regime = gex_engine.get_gamma_regime_text(5000, levels["zero_gamma"])
    assert regime["regime"] == "At Zero Gamma"
