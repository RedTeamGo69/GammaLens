"""Executable closing marks for every spread the forward study froze.

The frozen opening chain prices only hold-to-close. A premium stop or profit
target needs what the spread cost to close during the week, so each session
near the close this quotes both short legs and the recommended-width wings of
every live forecast and appends them to ft_marks. Read-only Tradier quotes on
the study's own HTTP session (ADR 0001); nothing here is hash-pinned, so a
change starts no new comparison series.
"""
from datetime import timedelta
import json

from phase1.trading_week import NY, trading_week
from .provider import epoch_time

# Declared sampling time. cron-job.org dispatches forward_test_marks.yml at
# 15:45 ET; anything outside the last 20 minutes of a regular session is
# rejected, because a post-close quote is not an executable mark. On a 13:00
# early close the 15:45 run fails and that session goes unsampled unless a
# one-off 12:45 dispatch is added for the day.
SAMPLE_MINUTES_BEFORE_CLOSE = 20
QUOTE_BATCH = 400


def forecast_legs(payload):
    """(leg, OCC symbol) for both shorts and the recommended-width wings."""
    width = payload.get("recommended_width")
    for side in ("put", "call"):
        short = (payload.get("short_contracts") or {}).get(side) or {}
        wing = next((w for w in (payload.get("wings") or {}).get(side, [])
                     if width and abs(w.get("width", 0) - width) < 1e-9), {})
        for leg, contract in ((f"{side}_short", short), (f"{side}_wing", wing)):
            if contract.get("symbol"):
                yield leg, contract["symbol"]


def tradier_quotes(provider, symbols):
    """POST, because a session's few hundred OCC symbols overflow a GET query string."""
    out = {}
    for i in range(0, len(symbols), QUOTE_BATCH):
        response = provider.http.post(f"{provider.client.base_url}/markets/quotes",
                                      data={"symbols": ",".join(symbols[i:i + QUOTE_BATCH])}, timeout=20)
        response.raise_for_status()
        rows = (response.json().get("quotes") or {}).get("quote") or []
        for q in [rows] if isinstance(rows, dict) else rows:
            out[q["symbol"]] = q
    return out


def _price(value):
    return None if value is None else float(value)


def _stamp(value):
    # Tradier's derived NDX quote carries 0 dates; keep that as unknown, not 1970.
    return epoch_time(value).isoformat() if value else None


def sample_marks(store, quotes, study_ids, *, clock):
    """Append one sample for every unexpired current-week forecast of study_ids.

    quotes(symbols) -> {symbol: Tradier quote}. Calls are deduped by OCC symbol,
    since models and tiers share contracts. A contract Tradier did not quote is
    reported, not written, so a sample row always holds the vendor's answer.
    """
    now = clock()
    week = trading_week(now.astimezone(NY).date())
    session = next((s for s in week.sessions if s.day == now.astimezone(NY).date()), None)
    result = {"session": str(session.day) if session else None, "forecasts": 0, "symbols": 0, "rows": 0, "errors": []}
    if session is None:  # the Mon-Fri dispatch also fires on exchange holidays
        return {**result, "skipped": "no regular session today"}
    if not session.close - timedelta(minutes=SAMPLE_MINUTES_BEFORE_CLOSE) <= now < session.close:
        raise ValueError(f"Outside the declared marks window (last {SAMPLE_MINUTES_BEFORE_CLOSE} "
                         "minutes of the regular session)")
    legs = []  # (forecast_id, ticker, leg, symbol)
    for study_id in study_ids:
        for f in store.forecasts(study_id, week.key):
            payload = json.loads(f["payload_json"])
            if payload["expiration"] >= str(session.day):
                legs.extend((f["forecast_id"], f["ticker"], leg, symbol) for leg, symbol in forecast_legs(payload))
    result["forecasts"] = len({l[0] for l in legs})
    if not legs:
        return result
    symbols = sorted({l[3] for l in legs} | {l[1] for l in legs})
    book = quotes(symbols)
    collected_at = clock().isoformat()
    rows = []
    for fid, ticker, leg, symbol in legs:
        q, under = book.get(symbol), book.get(ticker) or {}
        if not q:
            continue
        rows.append({"forecast_id": fid, "leg": leg, "symbol": symbol,
                     "bid": _price(q.get("bid")), "ask": _price(q.get("ask")),
                     "bid_at": _stamp(q.get("bid_date")), "ask_at": _stamp(q.get("ask_date")),
                     "underlying_last": _price(under.get("last")), "underlying_at": _stamp(under.get("trade_date")),
                     "collected_at": collected_at})
    store.append_marks(rows)
    missing = sorted({l[3] for l in legs} - set(book))
    if missing:
        result["errors"].append(f"No Tradier quote for {len(missing)} contracts: {', '.join(missing[:10])}")
    result.update(symbols=len(symbols), rows=len(rows))
    return result
