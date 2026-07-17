"""IV unit heuristic bounds (audit G21).

Tradier usually sends decimal IV (0.18); some feeds slip percent (18.0). The
parser rescales (3, 300] as percent but rejects > 300 as unit-ambiguous
garbage rather than silently producing a near-zero decimal.
"""
from phase1.data_client import TradierDataClient as _T


def _iv(mid):
    return _T._parse_iv_from_greeks({"mid_iv": mid})


def test_decimal_iv_passthrough():
    assert _iv(0.18) == 0.18


def test_percent_iv_rescaled():
    assert _iv(18.0) == 0.18
    assert _iv(300.0) == 3.0          # boundary: still treated as percent


def test_garbage_iv_rejected_not_shrunk():
    # A corrupt 3500 must NOT become 35.0 (3500%) or 0.35 — it's dropped to
    # 0.0 (missing) so the synthetic-IV path takes over.
    assert _iv(3500.0) == 0.0


def test_zero_and_missing_iv():
    assert _iv(0) == 0.0
    assert _T._parse_iv_from_greeks({}) == 0.0
