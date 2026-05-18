import requests

from phase1 import data_client
from phase1.data_client import PublicDataClient


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _wire_fake_public(monkeypatch, *, fail_first_quote_401=False):
    """Patch requests.* with a minimal in-memory Public API.

    Returns a mutable `state` dict so tests can assert how many access
    tokens were minted (token/account caching, 401-refresh behaviour).
    """
    state = {"mint": 0, "did_401": False, "fail_401": fail_first_quote_401}

    def post(url, json=None, headers=None, timeout=None):
        if url.endswith("/userapiauthservice/personal/access-tokens"):
            state["mint"] += 1
            assert json["secret"] == "SECRET"
            return _FakeResp(200, {"accessToken": f"TOK{state['mint']}"})
        if url.endswith("/quotes"):
            if state["fail_401"] and not state["did_401"]:
                state["did_401"] = True
                return _FakeResp(401, {})
            return _FakeResp(200, {"quotes": [{
                "instrument": {"symbol": "SPX", "type": "INDEX"},
                "outcome": "SUCCESS",
                "last": "5123.45", "bid": "5123.0", "ask": "5124.0",
                "previousClose": "5100.10",
                "oneDayChange": {"change": "23.35", "percentChange": "0.46"},
            }]})
        if url.endswith("/option-expirations"):
            return _FakeResp(200, {"expirations": ["2026-05-22", "2026-05-18"]})
        if url.endswith("/option-chain"):
            assert json["instrument"] == {"symbol": "SPX", "type": "INDEX"}
            return _FakeResp(200, {"calls": [{
                "bid": "12.1", "ask": "12.5", "volume": 1500, "openInterest": 8200,
                "optionDetails": {"strikePrice": "5125", "midPrice": "12.3",
                                  "greeks": {"gamma": "0.0123",
                                             "impliedVolatility": "0.184"}},
            }], "puts": [{
                "bid": "9.8", "ask": "10.2", "volume": 900, "openInterest": 4100,
                "optionDetails": {"strikePrice": "5100", "midPrice": "10.0",
                                  "greeks": {"gamma": "0.0111",
                                             "impliedVolatility": "21.5"}},
            }]})
        raise AssertionError(f"unexpected POST {url}")

    def get(url, headers=None, timeout=None):
        if url.endswith("/userapigateway/trading/account"):
            return _FakeResp(200, {"accounts": [
                {"accountId": "ACC-1", "accountType": "BROKERAGE"}]})
        raise AssertionError(f"unexpected GET {url}")

    def request(method, url, json=None, headers=None, timeout=None):
        if method == "POST":
            return post(url, json=json, headers=headers, timeout=timeout)
        return get(url, headers=headers, timeout=timeout)

    monkeypatch.setattr(data_client.requests, "post", post)
    monkeypatch.setattr(data_client.requests, "get", get)
    monkeypatch.setattr(data_client.requests, "request", request)
    return state


def test_full_quote_normalizes_public_shape(monkeypatch):
    _wire_fake_public(monkeypatch)
    client = PublicDataClient(token="SECRET")

    assert client.get_spot_price("SPX") == 5123.45
    q = client.get_full_quote("SPX")
    assert q["symbol"] == "SPX"
    assert q["last"] == 5123.45
    assert q["prevclose"] == 5100.10
    assert q["change"] == 23.35
    assert q["change_pct"] == 0.46
    assert client._account_id == "ACC-1"


def test_expirations_are_sorted(monkeypatch):
    _wire_fake_public(monkeypatch)
    client = PublicDataClient(token="SECRET")
    assert client.get_expirations("SPX") == ["2026-05-18", "2026-05-22"]


def test_chain_parses_optiondetails_and_normalizes_iv(monkeypatch):
    _wire_fake_public(monkeypatch)
    client = PublicDataClient(token="SECRET")

    chain = client.get_chain_once("SPX", "2026-05-18")
    assert chain["status"] == "ok"

    call = chain["calls"][0]
    assert call["strike"] == 5125.0
    assert call["vendorGamma"] == 0.0123
    assert call["impliedVolatility"] == 0.184
    assert call["mid"] == 12.3
    assert call["openInterest"] == 8200.0

    put = chain["puts"][0]
    # 21.5 is percent-style and must be normalized to a 0.215 decimal.
    assert abs(put["impliedVolatility"] - 0.215) < 1e-9


def test_token_and_account_are_cached(monkeypatch):
    state = _wire_fake_public(monkeypatch)
    client = PublicDataClient(token="SECRET")

    client.get_spot_price("SPX")
    client.get_spot_price("SPX")
    client.get_expirations("SPX")
    # One mint total: token + account id are both cached across calls.
    assert state["mint"] == 1


def test_401_forces_one_token_refresh_then_succeeds(monkeypatch):
    state = _wire_fake_public(monkeypatch, fail_first_quote_401=True)
    client = PublicDataClient(token="SECRET")

    assert client.get_spot_price("SPX") == 5123.45
    assert state["did_401"] is True
    # Initial mint + one forced refresh after the 401.
    assert state["mint"] == 2


def test_parse_iv_from_greeks_handles_percent_style():
    client = PublicDataClient(token="dummy")
    greeks = {"impliedVolatility": 25.0}
    iv = client._parse_iv_from_greeks(greeks)
    assert iv == 0.25


def test_parse_iv_from_greeks_handles_decimal_style():
    client = PublicDataClient(token="dummy")
    greeks = {"impliedVolatility": 0.22}
    iv = client._parse_iv_from_greeks(greeks)
    assert iv == 0.22


def test_parse_iv_from_greeks_handles_missing():
    client = PublicDataClient(token="dummy")
    assert client._parse_iv_from_greeks({}) == 0.0
    assert client._parse_iv_from_greeks(None) == 0.0


def test_get_chain_cached_uses_cache(monkeypatch):
    client = PublicDataClient(token="dummy")

    call_count = {"n": 0}

    def fake_get_chain_with_retry(ticker, expiration, retries=0, sleep_sec=0):
        call_count["n"] += 1
        return {"status": "ok", "calls": [], "puts": [], "error": None}

    monkeypatch.setattr(client, "get_chain_with_retry", fake_get_chain_with_retry)

    r1 = client.get_chain_cached("SPX", "2026-03-20")
    r2 = client.get_chain_cached("SPX", "2026-03-20")

    assert r1["status"] == "ok"
    assert r2["status"] == "ok"
    assert call_count["n"] == 1


def test_clear_cache_empties_cache(monkeypatch):
    client = PublicDataClient(token="dummy")

    def fake_get_chain_with_retry(ticker, expiration, retries=0, sleep_sec=0):
        return {"status": "ok", "calls": [], "puts": [], "error": None}

    monkeypatch.setattr(client, "get_chain_with_retry", fake_get_chain_with_retry)

    client.get_chain_cached("SPX", "2026-03-20")
    assert len(client.chain_cache) == 1

    client.clear_cache()
    assert len(client.chain_cache) == 0
