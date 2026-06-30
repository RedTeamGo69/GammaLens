"""Export-path chain plumbing in ui_spread_finder.

The weekly multi-ticker Excel export builds non-active rows from persisted
state, so it lacks the active ticker's live ``data.chain_cache``. These cover
the helpers that let it fetch + snap to the SAME real listed strikes the live
tab uses:

  * ``_chain_entry_to_quotes`` shapes a Tradier chain entry into the
    strike -> {bid/ask} lookup the snapper expects,
  * ``_export_chain_quotes`` degrades cleanly to ({}, None) when no Tradier
    token is configured (so the export still builds, on nominal strikes).
"""
import ui_spread_finder as usf


def test_chain_entry_to_quotes_maps_both_sides():
    entry = {
        "status": "ok",
        "calls": [
            {"strike": 557.5, "bid": 1.10, "ask": 1.30},
            {"strike": 560.0, "bid": 0.0, "ask": 0.55},   # $0 bid is a valid quote
        ],
        "puts": [
            {"strike": 490.0, "bid": 0.80, "ask": 1.00},
        ],
    }
    q = usf._chain_entry_to_quotes(entry)

    assert q[557.5]["call_bid"] == 1.10
    assert q[557.5]["call_ask"] == 1.30
    assert q[560.0]["call_bid"] == 0.0     # kept, not dropped
    assert q[490.0]["put_bid"] == 0.80
    # A call-only strike carries no put keys (and vice-versa).
    assert "put_bid" not in q[557.5]
    assert "call_bid" not in q[490.0]


def test_chain_entry_to_quotes_empty_entry():
    assert usf._chain_entry_to_quotes({"calls": [], "puts": []}) == {}


def test_export_chain_quotes_no_token_degrades(monkeypatch):
    # No Tradier token -> ({}, None) so the export falls back to nominal
    # strikes instead of raising / blocking the whole workbook build.
    monkeypatch.setattr(usf, "_tradier_token", lambda: "")
    assert usf._export_chain_quotes("AMD", None) == ({}, None)
