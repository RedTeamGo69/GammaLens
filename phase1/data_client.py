from __future__ import annotations

import logging
import os
import random
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

_logger = logging.getLogger(__name__)

from phase1.config import (
    MAX_WORKERS,
    CHAIN_RETRIES,
    CHAIN_RETRY_SLEEP,
)
from phase1.ticker_config import get_config


_safe_float_coercion_count = 0
_coercion_lock = threading.Lock()

def safe_float(x, default=0.0):
    global _safe_float_coercion_count
    try:
        return float(x)
    except (TypeError, ValueError):
        with _coercion_lock:
            _safe_float_coercion_count += 1
        return default

def get_coercion_count():
    return _safe_float_coercion_count

def reset_coercion_count():
    global _safe_float_coercion_count
    with _coercion_lock:
        _safe_float_coercion_count = 0


_DEFAULT_BASE_URL = "https://api.public.com"
_AUTH_PATH = "/userapiauthservice/personal/access-tokens"
_ACCOUNT_PATH = "/userapigateway/trading/account"


class PublicAuthError(RuntimeError):
    """Raised when the Public secret cannot be exchanged for an access token."""


def _instrument_for(ticker: str) -> dict:
    """Build the Public {symbol, type} instrument object for a dashboard ticker.

    Public requires an explicit instrument type. Index products (SPX, XSP)
    are INDEX; ETFs/stocks (QQQ, AMZN, AMD) are EQUITY. Both come from
    ticker_config so adding a symbol stays a one-file change.
    """
    cfg = get_config(ticker)
    symbol = cfg.get("public_symbol") or (ticker or "SPX").upper()
    inst_type = cfg.get("public_type") or "EQUITY"
    return {"symbol": symbol, "type": inst_type}


class PublicDataClient:
    """
    Market-data client backed by the Public.com brokerage API.

    Auth is two-legged: a long-lived ``secret`` (generated from Public's
    settings page) is exchanged for a short-lived bearer access token. The
    token is minted lazily, cached, proactively refreshed shortly before it
    expires, and re-minted once on a 401. Every market-data endpoint is
    account-scoped, so the accountId is resolved once from
    ``/userapigateway/trading/account`` (or the ``PUBLIC_ACCOUNT_ID`` env
    override for multi-account secrets).

    The public method surface (get_spot_price / get_full_quote(s) /
    get_expirations / get_chain_* / prefetch_chains / clear_cache) returns
    normalized shapes the rest of the pipeline consumes; the GEX engine
    never sees a vendor-specific payload.
    """

    def __init__(self, token: str, base_url: str = _DEFAULT_BASE_URL):
        # `token` is the Public personal secret (not the bearer token —
        # that is minted from it lazily).
        self.secret = token
        self.base_url = base_url.rstrip("/")
        self.chain_cache = {}

        self._access_token = None
        self._token_expiry = 0.0  # epoch seconds; 0 → not yet minted
        self._account_id = os.environ.get("PUBLIC_ACCOUNT_ID", "").strip() or None
        self._auth_lock = threading.Lock()

        try:
            self._validity_min = max(2, int(os.environ.get("PUBLIC_TOKEN_VALIDITY_MIN", "120")))
        except (TypeError, ValueError):
            self._validity_min = 120

    # ── Authentication ───────────────────────────────────────────────────
    def _mint_token(self):
        if not self.secret:
            raise PublicAuthError("Public API secret is empty")
        try:
            resp = requests.post(
                f"{self.base_url}{_AUTH_PATH}",
                json={"secret": self.secret, "validityInMinutes": self._validity_min},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
        except (requests.RequestException, requests.Timeout) as e:
            raise PublicAuthError(f"Could not reach Public auth endpoint: {e}") from e

        if resp.status_code in (401, 403):
            raise PublicAuthError(
                f"Public rejected the API secret (HTTP {resp.status_code}). "
                "Generate a fresh secret at https://public.com/settings/security/api"
            )
        resp.raise_for_status()
        token = (resp.json() or {}).get("accessToken")
        if not token:
            raise PublicAuthError("Public auth response did not include an accessToken")
        self._access_token = token
        # Refresh a minute before the server-side expiry so an in-flight
        # request never races the boundary.
        self._token_expiry = time.time() + self._validity_min * 60 - 60

    def _ensure_token(self, force=False):
        with self._auth_lock:
            if force or not self._access_token or time.time() >= self._token_expiry:
                self._mint_token()
            return self._access_token

    def _ensure_account_id(self):
        if self._account_id:
            return self._account_id
        # Resolve outside the auth lock (token mint takes it too); the GET is
        # idempotent so a rare concurrent double-fetch is harmless.
        token = self._ensure_token()
        try:
            resp = requests.get(
                f"{self.base_url}{_ACCOUNT_PATH}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
        except (requests.RequestException, requests.Timeout) as e:
            raise PublicAuthError(f"Could not resolve Public account id: {e}") from e

        accounts = (resp.json() or {}).get("accounts") or []
        if not accounts:
            raise PublicAuthError("No Public accounts are associated with this secret")
        chosen = next((a for a in accounts if a.get("accountType") == "BROKERAGE"), accounts[0])
        acct = chosen.get("accountId")
        if not acct:
            raise PublicAuthError("Public account payload was missing an accountId")
        self._account_id = acct
        return acct

    def _request(self, method, path, *, json=None, timeout=10):
        """Authenticated request that re-mints the token once on a 401."""
        url = f"{self.base_url}{path}"
        for attempt in (0, 1):
            token = self._ensure_token(force=(attempt == 1))
            resp = requests.request(
                method,
                url,
                json=json,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if resp.status_code == 401 and attempt == 0:
                continue  # token likely expired early — force-refresh and retry
            resp.raise_for_status()
            return resp
        # Unreachable: attempt 1 either returns or raise_for_status() raises.
        raise PublicAuthError("Public API kept returning 401 after a token refresh")

    # ── Quotes ───────────────────────────────────────────────────────────
    def get_spot_price(self, ticker="SPX"):
        acct = self._ensure_account_id()
        r = self._request(
            "POST",
            f"/userapigateway/marketdata/{acct}/quotes",
            json={"instruments": [_instrument_for(ticker)]},
        )
        quotes = (r.json() or {}).get("quotes") or []
        if not quotes:
            return 0.0
        q = quotes[0]
        last = q.get("last")
        if last in (None, "", 0, "0"):
            last = q.get("previousClose", 0)
        return safe_float(last, 0.0)

    @staticmethod
    def _normalize_quote(q, fallback_symbol):
        # Public exposes last/bid/ask/previousClose plus a oneDayChange
        # object. It does NOT publish intraday open/high/low — those keys
        # are kept (zeroed) only so the dict shape is unchanged for callers;
        # downstream OHLC comes from yfinance, never the broker quote.
        inst = q.get("instrument") or {}
        odc = q.get("oneDayChange") or {}
        return {
            "symbol": inst.get("symbol", fallback_symbol),
            "last": safe_float(q.get("last", 0), 0.0),
            "prevclose": safe_float(q.get("previousClose", 0), 0.0),
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "bid": safe_float(q.get("bid", 0), 0.0),
            "ask": safe_float(q.get("ask", 0), 0.0),
            "change": safe_float(odc.get("change", 0), 0.0),
            "change_pct": safe_float(odc.get("percentChange", 0), 0.0),
        }

    def get_full_quotes(self, tickers):
        """
        Batch quote lookup. Returns {ticker: normalized_quote} for every
        ticker Public answered. Public's /quotes takes an `instruments`
        array and returns one entry per instrument in `quotes`.
        """
        acct = self._ensure_account_id()
        instruments = [_instrument_for(t) for t in tickers]
        r = self._request(
            "POST",
            f"/userapigateway/marketdata/{acct}/quotes",
            json={"instruments": instruments},
        )
        quotes = (r.json() or {}).get("quotes") or []

        by_symbol = {}
        for row in quotes:
            if not isinstance(row, dict):
                continue
            sym = (row.get("instrument") or {}).get("symbol")
            if sym:
                by_symbol[sym] = row

        out = {}
        for t in tickers:
            want = _instrument_for(t)["symbol"]
            row = by_symbol.get(want) or by_symbol.get((t or "").upper())
            if row is not None:
                out[t] = self._normalize_quote(row, t)
        return out

    def get_full_quote(self, ticker="SPX"):
        """
        Return the full quote dict with prevclose, last, bid, ask, etc.

        Useful for computing overnight moves and pre-market context.
        """
        quotes = self.get_full_quotes([ticker])
        if ticker in quotes:
            return quotes[ticker]
        # Public dropped the symbol from the response — return a zeroed quote
        # with the requested ticker label so callers keep the shape they expect.
        return self._normalize_quote({}, ticker)

    def get_expirations(self, ticker="SPX"):
        acct = self._ensure_account_id()
        r = self._request(
            "POST",
            f"/userapigateway/marketdata/{acct}/option-expirations",
            json={"instrument": _instrument_for(ticker)},
        )
        e = (r.json() or {}).get("expirations") or []
        if isinstance(e, str):
            e = [e]
        return sorted(e)

    @staticmethod
    def _parse_iv_from_greeks(greeks):
        if not greeks:
            return 0.0
        iv = safe_float(greeks.get("impliedVolatility"), 0.0)
        if iv > 3:
            # Some feeds quote IV percent-style (25.0) rather than decimal
            # (0.25). Normalize defensively.
            _logger.debug("IV normalization: %.4f → %.4f (divided by 100)", iv, iv / 100.0)
            iv /= 100.0
        return max(iv, 0.0)

    def get_chain_once(self, ticker, expiration):
        """
        Fetch one options chain.

        Returns dict with:
            status: "ok" or "failed"
            calls: [...]
            puts: [...]
            error: optional string
        """
        try:
            acct = self._ensure_account_id()
            r = self._request(
                "POST",
                f"/userapigateway/marketdata/{acct}/option-chain",
                json={"instrument": _instrument_for(ticker), "expirationDate": expiration},
            )
        except PublicAuthError as e:
            return {"status": "failed", "calls": [], "puts": [], "error": f"Auth error: {e}"}
        except (requests.RequestException, requests.Timeout) as e:
            return {"status": "failed", "calls": [], "puts": [], "error": str(e)}

        # --- Response shape validation ---
        try:
            d = r.json()
        except (ValueError, TypeError) as e:
            return {"status": "failed", "calls": [], "puts": [], "error": f"Invalid JSON: {e}"}

        if not isinstance(d, dict):
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": f"Unexpected response type: {type(d).__name__}",
            }

        def _parse_side(side):
            """Return a list of normalized rows, or None if the field is malformed."""
            block = d.get(side)
            if block is None:
                # Vendor returned no contracts on this side for this expiration.
                return []
            if isinstance(block, dict):
                # Single contract returned as a dict instead of a list.
                block = [block]
            if not isinstance(block, list):
                return None

            rows = []
            for o in block:
                if not isinstance(o, dict):
                    continue
                od = o.get("optionDetails") or {}
                if not isinstance(od, dict):
                    od = {}

                strike = round(safe_float(od.get("strikePrice", 0), 0.0), 2)
                if strike <= 0:
                    continue

                bid = safe_float(o.get("bid", 0), 0.0)
                ask = safe_float(o.get("ask", 0), 0.0)
                oi = safe_float(o.get("openInterest", 0), 0.0)
                volume = safe_float(o.get("volume", 0), 0.0)

                greeks = od.get("greeks") or {}
                if not isinstance(greeks, dict):
                    greeks = {}
                iv = self._parse_iv_from_greeks(greeks)
                vendor_gamma = safe_float(greeks.get("gamma", 0), 0.0)

                # Prefer Public's published midPrice; fall back to the
                # bid/ask midpoint only when both sides are quoted.
                mid = safe_float(od.get("midPrice"), 0.0)
                if mid <= 0:
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0

                rows.append({
                    "strike": strike,
                    "openInterest": oi,
                    "volume": volume,
                    "impliedVolatility": iv,
                    "vendorGamma": vendor_gamma,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                })
            return rows

        calls = _parse_side("calls")
        puts = _parse_side("puts")
        if calls is None or puts is None:
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": "Malformed 'calls'/'puts' field in option-chain response",
            }

        return {"status": "ok", "calls": calls, "puts": puts, "error": None}

    def get_chain_with_retry(self, ticker, expiration, retries=CHAIN_RETRIES, sleep_sec=CHAIN_RETRY_SLEEP):
        last_error = None
        for attempt in range(retries + 1):
            result = self.get_chain_once(ticker, expiration)
            if result["status"] == "ok":
                if attempt > 0:
                    print(f"    ✓ Recovered {expiration} on retry {attempt}")
                return result
            last_error = result.get("error")
            if attempt < retries:
                # Exponential backoff with jitter. Parallel threads would
                # otherwise retry in lockstep and re-hammer the API after a
                # 429; jitter de-synchronizes them.
                backoff = sleep_sec * (2 ** attempt)
                jitter = random.uniform(0, sleep_sec)
                time.sleep(backoff + jitter)

        return {
            "status": "failed",
            "calls": [],
            "puts": [],
            "error": last_error or "Unknown fetch failure",
        }

    def get_chain_cached(self, ticker, expiration):
        key = (ticker, expiration)
        if key not in self.chain_cache:
            self.chain_cache[key] = self.get_chain_with_retry(ticker, expiration)
        return self.chain_cache[key]

    def prefetch_chains(self, ticker, expirations):
        uncached = [e for e in expirations if (ticker, e) not in self.chain_cache]
        if not uncached:
            return

        print(f"  Fetching {len(uncached)} chains in parallel...")

        def _fetch(exp):
            return exp, self.get_chain_with_retry(ticker, exp)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch, e): e for e in uncached}
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    exp_res, result = future.result()
                    self.chain_cache[(ticker, exp_res)] = result
                except Exception as e:
                    self.chain_cache[(ticker, exp)] = {
                        "status": "failed",
                        "calls": [],
                        "puts": [],
                        "error": f"Thread exception: {e}",
                    }

    def clear_cache(self):
        self.chain_cache.clear()
