"""Reproduce the explicitly reviewed catalog from preserved source evidence.

Offline review tool, never called by capture. New discrepancies require a new
review and methodology version; this script cannot infer a decision from a fit.
"""
import argparse
from datetime import date, timedelta
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pandas_market_calendars as mcal
from range_finder.forward_test.history_policy import (
    POLICY_VERSION, OHLC, _frame, _shape, _aligned, price_snapshot)
from scripts.analyze_forward_history import yahoo_frame

# Decisions are reviewed against independent price evidence, not forecast output.
KEEP_WEEKLY = {
    'SPX': '2018-04-30 2018-05-07 2018-05-14 2018-05-21 2018-07-09 2018-07-16 2020-09-28 2020-10-05'.split(),
    'SPY': '2016-06-20 2016-11-07 2017-01-23 2017-07-10 2017-10-30 2017-12-11'.split()}
REPLACE_WEEKLY = {'SPY':['2019-07-29'], 'AAPL':['2023-09-11'], 'AMD':['2019-07-08','2023-09-11']}
KEEP_DAILY = {'SPX':'2020-10-02 2020-10-05 2020-10-06 2020-10-07 2020-10-09 2021-08-11'.split()}
REPLACE_DAILY = {
    'SPX':['2020-11-03'], 'AAPL':['2023-09-11','2023-09-12'], 'AMD':['2023-09-11'],
    'SPY': '2020-09-17 2020-12-17 2021-03-18 2021-06-17 2021-09-16 2021-12-16 2022-03-17 2022-06-16 2022-09-15 2022-12-15 2023-03-16 2023-06-15 2023-09-14 2023-12-14 2024-03-14 2024-06-20 2024-09-19 2024-12-19 2025-03-20 2025-06-18 2025-09-18 2025-12-18 2026-03-19 2026-06-17'.split()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.evidence_root, args.output
    current = root/'forward-history-live-20260909'
    independent = root/'forward-history-resume-20260909'
    original = root/'forward-history-20260905'
    manifest = {r['file']:r for r in json.loads((current/'manifest.json').read_text())}
    for name, info in manifest.items():
        assert sha256((current/name).read_bytes()).hexdigest() == info['sanitized_sha256'], name
    rh = {}
    receipts = {}
    for path in [independent/'robinhood-stocks-full.json', *sorted(independent.glob('robinhood-spx-chunk-*.json'))]:
        payload = json.loads(path.read_text())
        receipt = {'source':'Robinhood historical market data connector', 'retrieved_at':payload['retrieved_at'],
                   'raw_sha256':sha256(path.read_bytes()).hexdigest(), 'file':path.name,
                   'encoding':'preserved connector response JSON', 'request':payload['params'],
                   'date_convention':'UTC midnight calendar labels; no New York date conversion'}
        receipts[path.name] = receipt
        for item in payload['result']['data']['results']:
            ticker = item['symbol']
            for bar in item['bars']:
                key = bar['begins_at'][:10]
                previous = rh.setdefault(ticker, {}).get(key)
                if previous:
                    assert previous[0] == bar, (ticker,key,'contradictory overlapping witness responses')
                rh[ticker][key] = (bar,receipt)

    def robinhood_witness(ticker, day, cadence, fields=OHLC):
        end = day + timedelta(days=6) if cadence == 'weekly' else day
        expected = mcal.get_calendar('NYSE').valid_days(start_date=day, end_date=end)
        rows, sources = [], {}
        for timestamp in expected:
            raw, receipt = rh[ticker][str(timestamp.date())]
            assert not raw.get('interpolated') and raw.get('session','reg') == 'reg'
            suffix = '_value' if ticker == 'SPX' else '_price'
            rows.append({'date':str(timestamp.date()), **{k:float(raw[k+suffix]) for k in OHLC}, 'raw':raw})
            sources[receipt['file']] = receipt
        frame = pd.DataFrame(rows)
        prices = {'open':frame.open.iloc[0], 'high':frame.high.max(), 'low':frame.low.min(), 'close':frame.close.iloc[-1]}
        receipt = next(iter(sources.values()))
        return {**receipt, 'symbol':ticker, 'session':'regular', 'adjustment':'price index' if ticker == 'SPX' else 'split only',
                'share_basis':'current', 'interpolated':False, 'fields':list(fields), 'prices':prices,
                'raw_rows':[r['raw'] for r in rows], 'receipts':list(sources.values())}

    def massive_witness(ticker, day, cadence):
        assert ticker == 'AAPL'
        path = independent/'massive-aapl-witness.json'
        payload = json.loads(path.read_text())
        frame = pd.read_csv(StringIO(payload['response']['structuredContent']['result'])).rename(columns=dict(zip('ohlc',OHLC)))
        frame.index = pd.to_datetime(frame.t, unit='ms', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
        end = day + timedelta(days=6) if cadence == 'weekly' else day
        expected = mcal.get_calendar('NYSE').valid_days(start_date=day, end_date=end).tz_localize(None)
        selected = frame.loc[expected]
        assert len(selected) == len(expected) and not selected.index.has_duplicates
        prices = {'open':selected.open.iloc[0], 'high':selected.high.max(), 'low':selected.low.min(), 'close':selected.close.iloc[-1]}
        return {'source':'Massive stock daily aggregates connector', 'symbol':ticker,
                'session':'daily qualifying-trade aggregate; whole OHLC agrees with regular-session candidate',
                'adjustment':'split only (adjusted=true), no dividend adjustment', 'share_basis':'current',
                'retrieved_at':payload['retrieved_at'],
                'raw_sha256':sha256(path.read_bytes()).hexdigest(),
                'encoding':'preserved connector response JSON containing normalized CSV', 'file':path.name,
                'request':payload['request'],
                'interpolated':False, 'fields':list(OHLC), 'prices':prices,
                'raw_rows':selected.to_dict('records')}

    def fred_witness(day):
        payload = json.loads((current/'fred-SP500.json').read_text())
        raw = next(r for r in payload['observations'] if r['date'] == str(day))
        return {**manifest['fred-SP500.json'], 'source':'FRED SP500, S&P Dow Jones Indices daily price-index close',
                'symbol':'SPX', 'session':'official daily index close', 'adjustment':'price index',
                'interpolated':False, 'fields':['close'], 'prices':{'close':float(raw['value'])}, 'raw_rows':[raw]}

    resolutions = []
    for ticker in ('SPX','SPY','AAPL','AMD'):
        alternative, _ = yahoo_frame(json.loads((current/f'yahoo-{ticker}-1d.json').read_text()))
        alt_weekly = alternative.resample('W-MON', closed='left', label='left').agg({'open':'first','high':'max','low':'min','close':'last'})
        for cadence, fields, keep, replace in [('weekly',OHLC,KEEP_WEEKLY,REPLACE_WEEKLY), ('daily',('close',),KEEP_DAILY,REPLACE_DAILY)]:
            primary = _frame(json.loads((current/f'tradier-{ticker}-{cadence}.json').read_text())['history']['day'])
            alt = alt_weekly if cadence == 'weekly' else alternative
            actual = {str(d.date()) for d,r in primary.iterrows() if not _shape(r) or not _aligned(r,alt.loc[d],fields)}
            assert actual == set(keep.get(ticker,[])+replace.get(ticker,[])), (ticker,cadence,actual)
            for label in sorted(actual):
                day = date.fromisoformat(label)
                chosen = 'primary' if label in keep.get(ticker,[]) else 'alternative'
                prices = primary.loc[label] if chosen == 'primary' else alt.loc[label]
                if ticker == 'AAPL' and label == '2023-09-11':
                    witnesses = [massive_witness(ticker,day,cadence)]
                elif ticker == 'SPX' and label == '2020-11-03':
                    witnesses = [robinhood_witness(ticker,day,cadence,OHLC[:3]), fred_witness(day)]
                else:
                    witnesses = [robinhood_witness(ticker,day,cadence)]
                for witness in witnesses:
                    assert _aligned(prices,witness['prices'],witness['fields']), (ticker,cadence,label,witness['prices'],price_snapshot(prices))
                reason = ('Independently corroborated primary bar; conflicting Yahoo values rejected' if chosen == 'primary' else
                          'Independent whole-OHLC witness corroborates regular-session Yahoo quote bar')
                if ticker == 'SPY' and cadence == 'daily':
                    reason += '; primary pre-dividend close uses a cash adjustment absent from OHL; no dividend arithmetic applied'
                if ticker == 'SPX' and label == '2020-11-03':
                    reason = 'FRED S&P DJI canonical close supersedes conflicting primary/RH close; RH corroborates candidate OHL; accept whole Yahoo bar'
                resolutions.append({'ticker':ticker,'cadence':cadence,'date':label,'accepted_source':chosen,'reason':reason,
                    'primary':price_snapshot(primary.loc[label]), 'alternative':price_snapshot(alt.loc[label]),
                    'primary_receipt':manifest[f'tradier-{ticker}-{cadence}.json'],
                    'alternative_receipt':manifest[f'yahoo-{ticker}-1d.json'], 'witnesses':witnesses})
    output.write_text(json.dumps({'policy_version':POLICY_VERSION, 'review_date':'2026-09-08 ET',
        'convention':'Current split basis; regular-session price returns, no dividend adjustment; Monday exchange weeks',
        'precision_policy':'0.0051 USD comparison tolerance for half-cent reporting plus float32; input prices never rounded',
        'unknown_discrepancy':'UNAVAILABLE_HISTORY; explicit review required', 'resolutions':resolutions},indent=2,allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'catalog':str(output),'decisions':len(resolutions),'whole_bar_replacements':sum(r['accepted_source']=='alternative' for r in resolutions)}))


if __name__ == '__main__':
    main()
