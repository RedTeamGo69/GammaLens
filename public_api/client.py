"""
Thin Public.com Trading API client — fills history + order preflight ONLY.

Endpoints verified against public.com/api/docs (2026-07):
  * POST /userapiauthservice/personal/access-tokens   {secret, validityInMinutes}
        → {accessToken}  (short-lived JWT, default 15 min)
  * GET  /userapigateway/trading/{accountId}/history  ?start&end&pageSize&nextToken
        → {transactions: [...], nextToken, pageSize}
  * POST /userapigateway/trading/{accountId}/preflight/multi-leg
        → estimated cost / buying-power / margin breakdown

House style: every call degrades gracefully (returns None + one logged
warning, never raises into the UI). Secrets resolve st.secrets → env and are
never logged. This module must never grow a market-data or order-placement
method — see the package docstring.
"""
from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger(__name__)

AUTH_URL = "https://api.public.com/userapiauthservice/personal/access-tokens"
TRADING_BASE = "https://api.public.com/userapigateway/trading"

SECRET_KEY_NAME = "PUBLIC_API_SECRET"
ACCOUNT_KEY_NAME = "PUBLIC_ACCOUNT_ID"

TOKEN_VALIDITY_MIN = 15          # Public's default; we refresh 60s early
DEFAULT_PAGE_SIZE = 100
MAX_PAGES = 200                  # 20k transactions — far above the ~550-leg history


def get_public_credentials() -> tuple[str, str]:
    """(secret, account_id) via the st.secrets → env chain; ("", "") when
    unconfigured. Import of streamlit is deferred so headless jobs
    (scheduled_snapshot, tests) never need it."""
    secret = ""
    account = ""
    try:
        import streamlit as st
        try:
            secret = st.secrets.get(SECRET_KEY_NAME, "") or ""
            account = st.secrets.get(ACCOUNT_KEY_NAME, "") or ""
        except Exception:
            pass
    except Exception:
        pass
    secret = secret or os.environ.get(SECRET_KEY_NAME, "")
    account = account or os.environ.get(ACCOUNT_KEY_NAME, "")
    return secret, account


class PublicClient:
    def __init__(self, secret: str, account_id: str, timeout: int = 15):
        self.secret = secret
        self.account_id = account_id
        self.timeout = timeout
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._auth_warned = False

    # ── auth ────────────────────────────────────────────────────────────────
    def _access_token(self) -> str | None:
        """Exchange the long-lived secret for a short-lived access token,
        cached until ~60s before expiry."""
        if self._token and time.time() < self._token_expires_at:
            return self._token
        if not self.secret:
            if not self._auth_warned:
                log.warning("Public API secret not configured "
                            f"({SECRET_KEY_NAME}) — Public features disabled.")
                self._auth_warned = True
            return None
        try:
            resp = requests.post(
                AUTH_URL,
                json={"secret": self.secret, "validityInMinutes": TOKEN_VALIDITY_MIN},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            token = (resp.json() or {}).get("accessToken")
            if not token:
                raise ValueError("no accessToken in response")
            self._token = token
            self._token_expires_at = time.time() + TOKEN_VALIDITY_MIN * 60 - 60
            return token
        except Exception as e:
            if not self._auth_warned:
                log.warning(f"Public API auth failed: {e}")
                self._auth_warned = True
            self._token = None
            return None

    def _headers(self) -> dict | None:
        token = self._access_token()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # ── history ─────────────────────────────────────────────────────────────
    def get_history_page(
        self,
        page_size: int = DEFAULT_PAGE_SIZE,
        next_token: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict | None:
        headers = self._headers()
        if headers is None or not self.account_id:
            return None
        params: dict = {"pageSize": page_size}
        if next_token:
            params["nextToken"] = next_token
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        try:
            resp = requests.get(
                f"{TRADING_BASE}/{self.account_id}/history",
                headers=headers, params=params, timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"Public history page failed: {e}")
            return None

    def get_all_history(
        self,
        start: str | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = MAX_PAGES,
    ) -> list[dict] | None:
        """Paginate to exhaustion. None when the FIRST page fails (auth /
        network — caller should treat Public as unavailable); a mid-stream
        failure returns what was fetched with a logged warning, because a
        partial idempotent sync is strictly better than none."""
        transactions: list[dict] = []
        next_token: str | None = None
        for page_num in range(max_pages):
            page = self.get_history_page(page_size=page_size,
                                         next_token=next_token, start=start)
            if page is None:
                if page_num == 0:
                    return None
                log.warning(f"Public history pagination stopped early at page "
                            f"{page_num} — proceeding with {len(transactions)} rows.")
                break
            transactions.extend(page.get("transactions") or [])
            next_token = page.get("nextToken")
            if not next_token:
                break
        else:
            log.warning(f"Public history hit the {max_pages}-page ceiling — "
                        f"pagination may be incomplete.")
        return transactions

    # ── order preflight (margin/cost estimate — NEVER placement) ────────────
    def preflight_multileg(
        self,
        *,
        legs: list[dict],
        quantity: int,
        limit_price: float | str,
        order_type: str = "LIMIT",
        expiration: dict | None = None,
    ) -> dict | None:
        """POST /preflight/multi-leg. ``legs`` use the documented shape:
        {instrument: {symbol, type}, side, openCloseIndicator, ratioQuantity,
        optionDetails: {baseSymbol, type, strikePrice, optionExpireDate}}."""
        headers = self._headers()
        if headers is None or not self.account_id:
            return None
        body = {
            "orderType": order_type,
            "expiration": expiration or {"timeInForce": "DAY"},
            "quantity": str(quantity),
            "limitPrice": str(limit_price),
            "legs": legs,
        }
        try:
            resp = requests.post(
                f"{TRADING_BASE}/{self.account_id}/preflight/multi-leg",
                headers=headers, json=body, timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"Public multi-leg preflight failed: {e}")
            return None
