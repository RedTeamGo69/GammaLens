# ADR 0001 — The forward study keeps its own market-data path

**Status:** accepted · 2026-09-26

## Context

The 2026-09-26 architecture review proposed putting all market data behind one module
(`range_finder/bar_sources.py`, `phase1/credentials.py`), with the forward study's
`TradierProvider` becoming a second adapter. The interactive app and the cron were
consolidated that way.

`range_finder/forward_test/provider.py` was deliberately left separate.

## Decision

The forward study keeps its own Tradier/Cboe/FRED access path. It is not an adapter of
`bar_sources`.

## Why

The two paths follow **opposite source policies**, so there is no single interface to share:

| | App / cron (`bar_sources`) | Forward study (`TradierProvider`) |
|---|---|---|
| Primary unavailable | Falls back to yfinance | **Fails closed**: the slot is marked unavailable/missed |
| Evidence | None | Every history call stores raw response text + SHA-256 receipt |
| HTTP | `TradierDataClient` per call | One retrying session with explicit status/backoff policy |
| Token | `phase1.credentials` (secrets → env) | Explicit argument (CLI must never read Streamlit secrets) |

Routing the study through `bar_sources` would silently add a yfinance fallback and drop the
receipts. Both would violate the study's protocol (`docs/weekly_forward_test.md`).

The provider's source is also part of the study's methodology hash
(`forward_test/config.py::methodology()`). Any edit starts a new comparison series.

The shared pieces that *are* policy-free are already reused: `TradierDataClient`'s base URL and
headers, `phase1.trading_week`, `cboe_data`, and `build_features(inputs=…, persist=False)`.

## Consequences

- Two market-data paths is intentional. Don't re-propose merging them without a new study
  protocol that accepts fallbacks.
- A fix to Tradier response parsing may need applying in both `phase1/data_client.py` and the
  provider.
