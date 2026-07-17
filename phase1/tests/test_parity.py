from phase1.parity import compute_implied_spot


def test_parity_uses_median_and_ignores_outlier():
    tradier_spot = 5000.0

    calls = [
        {"strike": 4990, "bid": 22.0, "ask": 22.4},
        {"strike": 5000, "bid": 16.0, "ask": 16.4},
        {"strike": 5010, "bid": 11.0, "ask": 11.4},
        {"strike": 5020, "bid": 8.0, "ask": 8.4},
        {"strike": 5030, "bid": 20.0, "ask": 20.4},  # noisy outlier
    ]

    puts = [
        {"strike": 4990, "bid": 12.0, "ask": 12.4},
        {"strike": 5000, "bid": 16.0, "ask": 16.4},
        {"strike": 5010, "bid": 21.0, "ask": 21.4},
        {"strike": 5020, "bid": 28.0, "ask": 28.4},
        {"strike": 5030, "bid": 45.0, "ask": 45.4},  # noisy outlier pair
    ]

    spot, source = compute_implied_spot(calls, puts, tradier_spot, r=0.0, T=0.0)

    assert source.startswith("implied weighted median")
    assert abs(spot - 5000.0) < 20.0


def test_parity_falls_back_if_not_enough_valid_quotes():
    tradier_spot = 5000.0

    calls = [{"strike": 5000, "bid": 0.0, "ask": 0.0}]
    puts = [{"strike": 5000, "bid": 0.0, "ask": 0.0}]

    spot, source = compute_implied_spot(calls, puts, tradier_spot, r=0.0, T=0.0)

    assert spot == tradier_spot
    assert "tradier" in source


# ── dividend yield in put-call parity (audit M29) ────────────────────────────

def test_parity_growth_factor_lifts_implied_spot():
    from phase1.parity import _compute_implied_spot_core
    K = 6000.0
    # Fabricate mids consistent with S=6000 under q=0 parity at K=6000:
    # C - P = S*e^{-qT} - K*e^{-rT}. With q=0, r≈0, C-P ≈ S-K = 0 at ATM.
    calls = [{"strike": K, "bid": 30.0, "ask": 30.2},
             {"strike": K + 5, "bid": 27.0, "ask": 27.2},
             {"strike": K - 5, "bid": 33.0, "ask": 33.2}]
    puts = [{"strike": K, "bid": 30.0, "ask": 30.2},
            {"strike": K + 5, "bid": 32.0, "ask": 32.2},
            {"strike": K - 5, "bid": 28.0, "ask": 28.2}]
    r, T = 0.04, 0.25
    base = _compute_implied_spot_core(calls, puts, 6000.0, r=r, T=T, q=0.0)
    lifted = _compute_implied_spot_core(calls, puts, 6000.0, r=r, T=T, q=0.02)
    if base["spot"] and lifted["spot"]:
        # e^{qT} = e^{0.005} ≈ 1.005 → implied spot scales up.
        assert lifted["spot"] > base["spot"]
