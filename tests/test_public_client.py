"""Public client pure parts: OCC building, preflight payload/normalization,
credential resolution, and graceful auth failure (no live HTTP anywhere)."""
import pytest

import public_api.client as client_mod
from public_api.client import (
    PublicClient,
    build_multileg_payload,
    get_public_credentials,
    normalize_preflight_response,
)
from public_api.occ import parse_occ_symbol, to_occ_symbol


# ── OCC round trip ────────────────────────────────────────────────────────────

def test_to_occ_symbol_round_trip():
    sym = to_occ_symbol("XSP", "2026-07-17", "call", 765.0)
    assert sym == "XSP260717C00765000"
    occ = parse_occ_symbol(sym)
    assert (occ.root, occ.expiration, occ.option_type, occ.strike) == \
        ("XSP", "2026-07-17", "call", 765.0)


def test_to_occ_symbol_fractional_strike():
    assert to_occ_symbol("SOFI", "2026-04-17", "put", 16.5) == \
        "SOFI260417P00016500"


# ── preflight payload ─────────────────────────────────────────────────────────

def test_build_multileg_payload_shape():
    legs = build_multileg_payload("XSP", "2026-07-17", [
        {"option_type": "put", "direction": "buy", "strike": 745},
        {"option_type": "call", "direction": "sell", "strike": 765},
    ])
    assert len(legs) == 2
    put, call = legs
    assert put["instrument"]["symbol"] == "XSP260717P00745000"
    assert put["side"] == "BUY"
    assert put["openCloseIndicator"] == "OPEN"
    assert put["optionDetails"]["type"] == "PUT"
    assert call["side"] == "SELL"
    assert call["optionDetails"]["strikePrice"] == "765"
    assert call["optionDetails"]["optionExpireDate"] == "2026-07-17"


# ── response normalization ────────────────────────────────────────────────────

def test_normalize_ok_response_documented_fields():
    out = normalize_preflight_response({
        "estimatedCost": "412.50",
        "buyingPowerRequirement": 415.0,
        "marginRequirement": {"longInitialRequirement": 500.0},
        "strategyName": "Iron Condor",
    })
    assert out["status"] == "ok"
    assert out["estimated_cost"] == pytest.approx(412.50)
    assert out["buying_power_impact"] == pytest.approx(415.0)
    assert out["margin_requirement"] == pytest.approx(500.0)
    assert "Iron Condor" in out["messages"][0]


def test_normalize_margin_impact_fallbacks():
    out = normalize_preflight_response({
        "orderValue": 100.0,
        "marginImpact": {"marginUsageImpact": 250.0,
                         "initialMarginRequirement": 300.0},
    })
    assert out["estimated_cost"] == pytest.approx(100.0)
    assert out["buying_power_impact"] == pytest.approx(250.0)
    assert out["margin_requirement"] == pytest.approx(300.0)


def test_normalize_rejection_and_none():
    assert normalize_preflight_response(None) is None
    rej = normalize_preflight_response({"error": "insufficient buying power"})
    assert rej["status"] == "rejected"
    assert "insufficient buying power" in rej["messages"][0]


# ── credentials ───────────────────────────────────────────────────────────────

def test_get_public_credentials_env_fallback(monkeypatch):
    # Hermeticity: stub the st.secrets seam — the dev machine's real
    # secrets.toml must never leak into unit tests.
    monkeypatch.setattr(client_mod, "_read_st_secret", lambda key: "")
    monkeypatch.setenv("PUBLIC_API_SECRET", "s3cr3t")
    monkeypatch.setenv("PUBLIC_ACCOUNT_ID", "ACCT1")
    secret, account = get_public_credentials()
    assert (secret, account) == ("s3cr3t", "ACCT1")


def test_get_public_credentials_blank_when_unset(monkeypatch):
    monkeypatch.setattr(client_mod, "_read_st_secret", lambda key: "")
    monkeypatch.delenv("PUBLIC_API_SECRET", raising=False)
    monkeypatch.delenv("PUBLIC_ACCOUNT_ID", raising=False)
    secret, account = get_public_credentials()
    assert secret == "" and account == ""


def test_st_secrets_takes_precedence_over_env(monkeypatch):
    monkeypatch.setattr(client_mod, "_read_st_secret",
                        lambda key: {"PUBLIC_API_SECRET": "from-secrets",
                                     "PUBLIC_ACCOUNT_ID": "A2"}.get(key, ""))
    monkeypatch.setenv("PUBLIC_API_SECRET", "from-env")
    secret, account = get_public_credentials()
    assert (secret, account) == ("from-secrets", "A2")


# ── graceful auth failure (no network) ────────────────────────────────────────

def test_client_degrades_when_auth_unreachable(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("no network in tests")

    monkeypatch.setattr(client_mod.requests, "post", _boom)
    monkeypatch.setattr(client_mod.requests, "get", _boom)
    c = PublicClient("secret", "ACCT1")
    assert c._access_token() is None
    assert c.get_history_page() is None
    assert c.get_all_history() is None
    assert c.preflight_multileg(legs=[], quantity=1, limit_price=0.5) is None


def test_client_no_secret_short_circuits():
    c = PublicClient("", "ACCT1")
    assert c._access_token() is None
    assert c.get_history_page() is None
