"""Actual source-bar regression windows; no network or production writes."""
from copy import deepcopy
from datetime import date, datetime, timedelta
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from range_finder.forward_test.history_policy import (CATALOG, OHLC, SYMBOLS,
    HistoryUnavailable, _choose, _frame, price_snapshot, validated_history)
from range_finder.forward_test.provider import valid_ohlc
from range_finder.trading_week import NY, trading_week


@lru_cache
def archived_windows():
    path = Path(__file__).parent/'fixtures/forward_history_windows.json.gz'
    return json.loads(gzip.decompress(path.read_bytes()))


def yahoo_payload(ticker, frame):
    symbol, kind = SYMBOLS[ticker]
    return {'chart':{'error':None,'result':[{
        'meta':{'symbol':symbol,'instrumentType':kind,'currency':'USD','exchangeTimezoneName':'America/New_York'},
        'timestamp':[int(pd.Timestamp(d).tz_localize(NY).timestamp())+9*3600+30*60 for d in frame.index],
        'indicators':{'quote':[{k:frame[k].tolist() for k in frame.columns}]}}]}}


def validate(ticker, data=None, **kwargs):
    fixture = archived_windows()
    data = data or fixture['tickers'][ticker]
    return validated_history(ticker, date.fromisoformat(data['weekly'][0]['date']),
        date.fromisoformat(fixture['end']), date.fromisoformat(fixture['daily_start']),
        data['weekly'], data['daily'], data['yahoo'],
        as_of=kwargs.pop('as_of',datetime.fromisoformat(fixture['as_of'])), **kwargs)


@pytest.mark.parametrize('ticker,nweekly,ndecisions,nreplaced',[
    ('SPX',535,15,1),('SPY',534,31,25),('AAPL',326,3,3),('AMD',534,3,3)])
def test_actual_full_windows_are_covered_with_independent_whole_bar_decisions(ticker,nweekly,ndecisions,nreplaced):
    before = deepcopy(archived_windows()['tickers'][ticker])
    weekly,daily,evidence = validate(ticker)
    assert len(weekly)==nweekly and len(daily)==1521
    assert str(daily.index[0].date())=='2020-08-17' and str(daily.index[-1].date())=='2026-09-04'
    assert all(valid_ohlc(r.to_dict()) for frame in (weekly,daily) for _,r in frame.iterrows())
    assert len(evidence['decisions'])==ndecisions
    assert sum(d['accepted_source']=='alternative' for d in evidence['decisions'])==nreplaced
    for decision in evidence['decisions']:
        accepted = weekly if decision['cadence']=='weekly' else daily
        assert price_snapshot(accepted.loc[decision['date']])==decision[decision['accepted_source']]
        assert all(len(w['raw_sha256'])==64 and w['raw_rows'] and not w.get('interpolated') for w in decision['witnesses'])
    assert archived_windows()['tickers'][ticker]==before


def test_cash_dividends_and_split_basis_are_not_computed_price_repairs():
    _,daily,evidence=validate('SPY')
    raw=_frame(archived_windows()['tickers']['SPY']['daily'])
    bad=[str(d.date()) for d,r in raw.iterrows() if not valid_ohlc(r.to_dict())]
    assert len(bad)==8
    for day in bad:
        d=next(r for r in evidence['decisions'] if r['cadence']=='daily' and r['date']==day)
        assert price_snapshot(daily.loc[day])==d['alternative']
    aapl,adaily,_=validate('AAPL')
    # Aug 31, 2020 split uses current-share OHLC on both sides of the boundary.
    assert 100 < adaily.at[pd.Timestamp('2020-08-28'),'close'] < 150
    assert 100 < adaily.at[pd.Timestamp('2020-08-31'),'close'] < 150
    assert 100 < aapl.at[pd.Timestamp('2020-08-24'),'open'] < 150


def test_recorded_missing_spx_ohlc_cannot_recur_unreviewed():
    data=deepcopy(archived_windows()['tickers']['SPX'])
    row=next(r for r in data['daily'] if r['date']=='2026-03-19')
    # Upstream corrected these fields in the current evidence. A recurrence
    # must still stop admission, despite a matching independently seen close.
    row.update(open='NaN',high='NaN',low='NaN')
    with pytest.raises(HistoryUnavailable,match='SPX daily 2026-03-19: unreviewed'):
        validate('SPX',data)


@pytest.mark.parametrize('mutation,reason',[
    ('missing','1 missing'),('duplicate','duplicate primary'),('future','unexpected dates'),
    ('unknown_close','unreviewed'),('unknown_weekly_high','unreviewed'),
    ('total_return','unreviewed'),('wrong_symbol','identity mismatch'),
    ('wrong_zone','identity mismatch'),('unavailable','response invalid')])
def test_unknown_missing_incompatible_or_future_data_remains_unavailable(mutation,reason):
    data=deepcopy(archived_windows()['tickers']['SPY'])
    result=data['yahoo']['chart']['result'][0]
    if mutation=='missing':data['daily'].pop(-3)
    elif mutation=='duplicate':data['daily'].append(deepcopy(data['daily'][-3]))
    elif mutation=='future':data['daily'].append({**data['daily'][-1],'date':'2026-09-08'})
    elif mutation=='unknown_close':data['daily'][-3]['close']+=.10
    elif mutation=='unknown_weekly_high':data['weekly'][-3]['high']+=.10
    elif mutation=='total_return':
        result['indicators']['quote'][0]['close']=[v*.99 for v in result['indicators']['quote'][0]['close']]
        # Keep candidate bar shapes valid while deliberately changing its basis.
        result['indicators']['quote'][0]['low']=[v*.98 for v in result['indicators']['quote'][0]['low']]
    elif mutation=='wrong_symbol':result['meta']['symbol']='SPX'
    elif mutation=='wrong_zone':result['meta']['exchangeTimezoneName']='UTC'
    elif mutation=='unavailable':data['yahoo']={'chart':{'error':'unavailable','result':None}}
    with pytest.raises(HistoryUnavailable,match=reason):validate('SPY',data)


def test_incomplete_week_and_future_week_cannot_become_training_history():
    with pytest.raises(HistoryUnavailable,match='has not completed'):
        validate('SPY',as_of=trading_week(date(2026,8,31)).evaluation_close-timedelta(seconds=1))
    data=archived_windows()['tickers']['SPY']
    with pytest.raises(HistoryUnavailable,match='before the final exchange session'):
        validated_history('SPY',date(2016,6,13),date(2026,9,3),date(2020,8,15),
            data['weekly'],data['daily'],data['yahoo'],as_of=datetime.fromisoformat(archived_windows()['as_of']))


def test_adjusted_close_is_not_consumed_and_final_daily_weekly_must_agree():
    data=deepcopy(archived_windows()['tickers']['AMD'])
    before=validate('AMD',data)
    data['yahoo']['chart']['result'][0]['indicators']['adjclose']=[{'adjclose':[1]*3000}]
    after=validate('AMD',data)
    pd.testing.assert_frame_equal(before[1],after[1])
    # A coherent but incompatible daily/weekly pair cannot pass merely because
    # both have individual witnesses: the final session must still reconcile.
    resolutions=deepcopy(json.loads(CATALOG.read_text())['resolutions'])
    last=data['daily'][-1];last['close']-=.1
    alt=data['yahoo']['chart']['result'][0]['indicators']['quote'][0]
    alternative={k:alt[k][-1] for k in OHLC}
    resolutions.append({'ticker':'AMD','cadence':'daily','date':last['date'],
        'primary':price_snapshot(last),'alternative':alternative,'accepted_source':'primary',
        'witnesses':[{'source':'isolated contradictory fixture','raw_sha256':'a'*64,'retrieved_at':'2026-09-09T00:00:00Z',
                      'fields':['close'],'prices':{'close':last['close']}}]})
    with pytest.raises(HistoryUnavailable,match='daily/weekly final close mismatch'):
        validate('AMD',data,resolutions=resolutions)


def test_changed_witness_or_missing_provenance_cannot_authorize_replacement():
    for mutation in ('prices','raw_sha256','interpolated'):
        catalog=deepcopy(json.loads(CATALOG.read_text())['resolutions'])
        decision=next(r for r in catalog if r['ticker']=='AAPL' and r['cadence']=='weekly')
        witness=decision['witnesses'][0]
        if mutation=='prices':witness['prices']['open']+=1
        elif mutation=='raw_sha256':witness['raw_sha256']=None
        else:witness['interpolated']=True
        with pytest.raises(HistoryUnavailable,match='witness'):validate('AAPL',resolutions=catalog)


def test_corroboration_failure_is_bounded_and_sanitized(monkeypatch):
    from curl_cffi import requests
    from range_finder.forward_test.history_policy import fetch_yahoo_history
    attempts=[]
    class Session:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,**kwargs):
            attempts.append((url,kwargs))
            raise RuntimeError('fixture sensitive transport detail')
    monkeypatch.setattr(requests,'Session',Session)
    with pytest.raises(HistoryUnavailable,match=r'Yahoo corroboration unavailable \(RuntimeError\)') as exc:
        fetch_yahoo_history('SPY',date(2020,8,17),date(2026,9,4),lambda:datetime.fromisoformat(archived_windows()['as_of']))
    assert 'sensitive' not in str(exc.value) and len(attempts)==1
    assert attempts[0][1]['timeout']==20 and attempts[0][1]['params']['includePrePost']=='false'


def test_methodology_archives_source_policy_and_catalog_hash():
    from hashlib import sha256
    from range_finder.forward_test.config import methodology
    _,config=methodology()
    assert config['history_policy']=='tradier-reviewed-whole-bars-v1'
    assert config['source_hashes']['forward_test/history_resolutions.json']==sha256(CATALOG.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


@pytest.mark.parametrize('ticker',('SPX','SPY','AAPL','AMD'))
def test_ui_headless_parity_with_actual_accepted_history_inputs(ticker):
    from range_finder.tests.forward_fixtures import Clock, prepared_fixture
    from range_finder.forward_test.config import MODELS
    from range_finder.forward_test.capture import capture_model
    from range_finder.forward_test.provider import frame_records
    from range_finder.feature_builder import build_features
    from range_finder.har_model import fit_production_model, train_window_min_date
    from range_finder.recommendations import build_recommendations,chain_entry_to_quotes
    from ui_spread_finder import _tier_bands_from_tiers
    # All fitting is isolated regression work. Synthetic volatility/quotes are
    # explicitly fixture inputs; accepted historical OHLC is the recorded data.
    week=trading_week(date(2026,9,7));clock=Clock(week.capture_start)
    p=prepared_fixture(ticker,week,clock)
    weekly,daily,evidence=validate(ticker)
    base=weekly.rename(columns={k:'spx_'+k for k in weekly.columns})
    base['range_pct']=(base.spx_high-base.spx_low)/base.spx_open
    base['log_range']=np.log(base.range_pct)
    base['spx_return']=(base.spx_close-base.spx_open)/base.spx_open
    base['vix_close']=20+np.sin(np.arange(len(base))*.17)
    raw={k:pd.DataFrame(v).set_index('date') if v else pd.DataFrame() for k,v in p['raw_inputs'].items()}
    for f in raw.values():f.index=pd.to_datetime(f.index)
    raw.update(weekly=base,daily=daily[['close']].rename(columns={'close':'spx_close'}))
    p.update(weekly=base,features=build_features(None,ticker=ticker,inputs=raw,as_of=clock(),persist=False))
    p['raw_inputs']={**{k:frame_records(f) for k,f in raw.items()},'history_evidence':evidence}
    for model in MODELS:
        inputs,frozen=capture_model(p,model,week,'isolated-reviewed-methodology',clock())
        cols=inputs['feature_columns']
        minimum=pd.Timestamp(train_window_min_date(as_of=clock()))
        training=p['features'][(p['features'].index<pd.Timestamp(week.monday)) & (p['features'].index>=minimum)]
        fit=fit_production_model(training,cols)
        _,_,tiers=build_recommendations(result=fit,feature_row=p['features'].loc[pd.Timestamp(week.monday)],
            feature_cols=cols,reference=p['reference'],vix=p['vix'],ticker=ticker,
            week_start=str(week.monday),side_share_q=inputs['side_share']['q'],chain_quotes=chain_entry_to_quotes(p['chain']))
        assert _tier_bands_from_tiers(tiers)=={r['tier']:(r['put_short'],r['call_short']) for r in frozen}
        assert inputs['raw_inputs']['history_evidence']==evidence
