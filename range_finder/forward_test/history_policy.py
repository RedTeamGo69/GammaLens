"""Reviewed training-history reconciliation, never an observation/price repair.

Tradier remains primary. Yahoo quote OHLC is a bounded corroborating source
and supplies whole replacement bars only for exact, independently reviewed
discrepancies. An unknown disagreement is unavailable, even if both bars have
valid shapes. The catalog is methodology-versioned; weekly refits do not edit it.
"""
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from range_finder.trading_week import NY

POLICY_VERSION = 'tradier-reviewed-whole-bars-v1'
PRICE_PRECISION = 0.0051  # half-cent reporting plus float32 representation
OHLC = ('open', 'high', 'low', 'close')
CATALOG = Path(__file__).with_name('history_resolutions.json')
SYMBOLS = {'SPX': ('^GSPC', 'INDEX'), 'SPY': ('SPY', 'ETF'),
           'AAPL': ('AAPL', 'EQUITY'), 'AMD': ('AMD', 'EQUITY')}


class HistoryUnavailable(ValueError):
    pass


def _fail(reason):
    raise HistoryUnavailable('UNAVAILABLE_HISTORY: '+reason)


def price_snapshot(row):
    return {key: None if pd.isna(row[key]) else float(row[key]) for key in OHLC}


def _shape(row):
    try:
        o, h, l, c = (float(row[k]) for k in OHLC)
        return all(np.isfinite(x) and x > 0 for x in (o,h,l,c)) and l <= min(o,c) <= max(o,c) <= h
    except (KeyError, TypeError, ValueError):
        return False


def _aligned(a, b, fields=OHLC):
    return all(pd.notna(a[k]) and pd.notna(b[k]) and abs(float(a[k])-float(b[k])) <= PRICE_PRECISION for k in fields)


def _frame(rows):
    if not rows:
        _fail('empty primary history')
    frame = pd.DataFrame(rows).set_index('date')
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None or frame.index.has_duplicates or (frame.index != frame.index.normalize()).any():
        _fail('invalid or duplicate primary date labels')
    if any(k not in frame for k in OHLC):
        _fail('missing primary OHLC columns')
    return frame.apply(pd.to_numeric, errors='coerce').sort_index()


def _coverage(frame, expected, label):
    if frame.index.has_duplicates:
        _fail(label+' duplicate dates')
    missing, extra = expected.difference(frame.index), frame.index.difference(expected)
    if len(missing) or len(extra):
        _fail(f'{label} coverage: {len(missing)} missing, {len(extra)} unexpected dates')


def fetch_yahoo_history(ticker, start, end, clock):
    """One GET, explicit session/basis, no auto-adjust, repair or retry loop."""
    from curl_cffi import requests
    symbol, _ = SYMBOLS[ticker]
    params = {'period1': int(datetime.combine(start, datetime.min.time(), NY).timestamp()),
              'period2': int(datetime.combine(end+timedelta(days=1), datetime.min.time(), NY).timestamp()),
              'interval':'1d', 'includePrePost':'false', 'events':'div,splits,capitalGains'}
    try:
        with requests.Session(impersonate='chrome') as session:
            response = session.get('https://query2.finance.yahoo.com/v8/finance/chart/'+quote(symbol, safe=''),
                                   params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            evidence = {'source':'Yahoo chart quote OHLC', 'symbol':symbol,
                'session':'regular', 'adjustment':'split only; no dividend adjustment',
                'retrieved_at':clock().isoformat(), 'params':params,
                'raw_sha256':sha256(response.content).hexdigest(), 'raw_response':response.text}
            return payload, evidence
    except Exception as exc:
        _fail('Yahoo corroboration unavailable ('+type(exc).__name__+')')


def _yahoo_frame(payload, ticker, expected):
    try:
        result = payload['chart']['result'][0]
        meta = result['meta']
        symbol, kind = SYMBOLS[ticker]
        if (payload['chart'].get('error') or meta.get('symbol') != symbol or
                meta.get('instrumentType') != kind or meta.get('currency') != 'USD' or
                meta.get('exchangeTimezoneName') != 'America/New_York'):
            _fail('Yahoo instrument/session identity mismatch')
        index = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert(NY).tz_localize(None).normalize()
        frame = pd.DataFrame(result['indicators']['quote'][0], index=index).sort_index()
        if any(k not in frame for k in OHLC):
            _fail('Yahoo OHLC columns missing')
        frame = frame.apply(pd.to_numeric, errors='coerce')
        _coverage(frame, expected, 'Yahoo daily')
        if not all(_shape(row) for _, row in frame.iterrows()):
            _fail('Yahoo invalid OHLC; no repair permitted')
        return frame
    except HistoryUnavailable:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _fail('Yahoo response invalid ('+type(exc).__name__+')')


def _choose(primary, alternative, ticker, cadence, resolutions, fields):
    accepted, decisions, warnings = primary.copy(), [], []
    for day, row in primary.iterrows():
        other = alternative.loc[day]
        if _shape(row) and _aligned(row, other, fields):
            if fields != OHLC and not _aligned(row, other):
                warnings.append({'date':str(day.date()), 'reason':'unused daily OHL differs; consumed close agrees',
                                 'primary':price_snapshot(row), 'alternative':price_snapshot(other)})
            continue
        candidates = [r for r in resolutions if r['ticker'] == ticker and r['cadence'] == cadence
                      and r['date'] == str(day.date()) and r['primary'] == price_snapshot(row)
                      and r['alternative'] == price_snapshot(other)]
        if len(candidates) != 1:
            _fail(f'{ticker} {cadence} {day.date()}: unreviewed OHLC/price disagreement')
        decision = candidates[0]
        source = decision['accepted_source']
        chosen = row if source == 'primary' else other if source == 'alternative' else None
        if chosen is None or not _shape(chosen):
            _fail('invalid reviewed whole-bar decision')
        # Evidence must substantiate the disputed consumed fields. Whole-bar
        # replacements also have whole-OHLC corroboration (possibly separate
        # official close evidence plus independent OHL on the same date).
        supported = set()
        for witness in decision['witnesses']:
            if (witness.get('interpolated') or not witness.get('source') or
                    not witness.get('raw_sha256') or not witness.get('retrieved_at')):
                _fail('invalid reviewed witness provenance')
            scope = tuple(witness['fields'])
            if not _aligned(chosen, witness['prices'], scope):
                _fail('reviewed witness does not corroborate chosen prices')
            supported.update(scope)
        required = set(OHLC if source == 'alternative' else fields)
        if not required <= supported:
            _fail('reviewed witness fields incomplete')
        if source == 'alternative':
            # Assign the entire source bar, never synthesize individual fields.
            accepted.loc[day] = other.reindex(accepted.columns)
        decisions.append(decision)
    return accepted, decisions, warnings


def validated_history(ticker, start, end, daily_start, primary_weekly, primary_daily,
                      alternative_payload, *, as_of, resolutions=None):
    """Pure validation/selection. No forecasts, DB writes or shortened windows."""
    if ticker not in SYMBOLS or start > daily_start or daily_start > end:
        _fail('invalid instrument/history window')
    expected = mcal.get_calendar('NYSE').valid_days(start_date=start, end_date=end).tz_localize(None)
    weekly_labels = pd.DatetimeIndex((expected-pd.to_timedelta(expected.weekday, unit='D')).unique())
    if not len(expected) or weekly_labels[0].date() < start:
        _fail('weekly history must start at the beginning of its exchange week')
    # Reject an incomplete final week; daily closes from an in-progress week
    # must not be promoted into a completed weekly training outcome.
    from range_finder.trading_week import trading_week
    if trading_week(end).sessions[-1].day != end:
        _fail('history ends before the final exchange session of its week')
    if as_of.tzinfo is None or trading_week(end).evaluation_close > as_of:
        _fail('history includes a week that has not completed as of retrieval')
    weekly, daily = _frame(primary_weekly), _frame(primary_daily)
    _coverage(weekly, weekly_labels, 'primary weekly')
    _coverage(daily, expected[expected >= pd.Timestamp(daily_start)], 'primary daily')
    alternative = _yahoo_frame(alternative_payload, ticker, expected)
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last'}
    if 'volume' in alternative:
        agg['volume'] = 'sum'
    alt_weekly = alternative.resample('W-MON', closed='left', label='left').agg(agg)
    _coverage(alt_weekly, weekly_labels, 'alternative weekly aggregation')
    if resolutions is None:
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
        if catalog['policy_version'] != POLICY_VERSION:
            _fail('history review catalog version mismatch')
        resolutions = catalog['resolutions']
    weekly, wd, _ = _choose(weekly, alt_weekly, ticker, 'weekly', resolutions, OHLC)
    daily, dd, warnings = _choose(daily, alternative.reindex(daily.index), ticker, 'daily', resolutions, ('close',))
    for monday, frame in daily.groupby(daily.index-pd.to_timedelta(daily.index.weekday, unit='D')):
        if abs(float(frame.iloc[-1]['close'])-float(weekly.at[monday,'close'])) > PRICE_PRECISION:
            _fail(f'{ticker} {monday.date()}: accepted daily/weekly final close mismatch')
    evidence = {'policy_version':POLICY_VERSION, 'status':'accepted',
                'start':str(start), 'daily_start':str(daily_start), 'end':str(end),
                'weekly_rows':len(weekly), 'daily_rows':len(daily),
                'primary_weekly':primary_weekly, 'primary_daily':primary_daily,
                'alternative_payload':alternative_payload, 'decisions':wd+dd,
                'unused_daily_ohl_warnings':warnings,
                'adjustment':'current split basis, no dividend adjustment',
                'daily_consumed_fields':['close'], 'weekly_consumed_fields':list(OHLC)}
    return weekly, daily, evidence
