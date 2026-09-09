"""Read-only release evidence. Never invokes capture, model fitting or migrations.

Uses explicit environment credentials, or --secrets-file for the intended app.
Only redacted capability/identity summaries are printed or written.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pandas_market_calendars as mcal
import psycopg2
from range_finder.forward_test.config import UNIVERSE, utcnow
from range_finder.forward_test.provider import TradierProvider, epoch_time, valid_ohlc
from range_finder.forward_test.store import Store
from range_finder.forward_test.capture import contract_chain
from range_finder.trading_week import NY, trading_week, listed_week_expiration
from range_finder.cboe_data import fetch_cboe_index_history
from range_finder.event_calendars import FOMC_DATES, CPI_DATES, NFP_DATES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--secrets-file')
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-database-fingerprint')
    args=parser.parse_args()
    cfg=tomllib.loads(Path(args.secrets_file).read_text()) if args.secrets_file else os.environ
    now=utcnow()
    week=trading_week(now.astimezone(NY).date())
    if now>=week.capture_end:week=trading_week(week.monday+timedelta(days=7))
    prior=trading_week(week.monday-timedelta(days=7))
    required_prior=prior
    while prior.evaluation_close > now:
        prior=trading_week(prior.monday-timedelta(days=7))
    end=prior.sessions[-1].day
    start=week.monday-timedelta(days=int(6*365.25)+30)
    report={'checked_at':now.isoformat(),'mode':'read_only_no_forecasts',
            'target_week':str(week.monday),'capture_start':week.capture_start.isoformat(),
            'capture_end_exclusive':week.capture_end.isoformat(),'evaluation_close':week.evaluation_close.isoformat(),
            'history_review_through':str(end),
            'required_history_through_at_capture':str(required_prior.sessions[-1].day),
            'future_history_pending':required_prior.evaluation_close > now,
            'sources':{},'tickers':{},'blockers':[],
            'session_dependent':['Opening-session quote freshness and open anchors',
                                 'Observed transport/feed latency during a live session',
                                 'Complete first-session minute coverage after the close']}
    for name,dates in [('FOMC',FOMC_DATES),('CPI',CPI_DATES),('NFP',NFP_DATES)]:
        report['sources'][name]={'calendar_last_event':max(dates),'covers_target_week':max(dates)>=str(week.sessions[-1].day)}
        if not report['sources'][name]['covers_target_week']:report['blockers'].append(name+' calendar expired')
    # Identity inspection uses the selected application's URL, never an
    # unrelated inherited DATABASE_URL when --secrets-file is supplied.
    url=cfg.get('DATABASE_URL')
    parsed=urlparse(url or '')
    report['database']={'host':parsed.hostname,'database':parsed.path.lstrip('/'),
                        'target_fingerprint':hashlib.sha256(f'{parsed.hostname}/{parsed.path}'.encode()).hexdigest()[:16]}
    if args.expected_database_fingerprint and report['database']['target_fingerprint'] != args.expected_database_fingerprint:
        raise SystemExit('Configured database does not match the reviewed production target fingerprint')
    try:
        conn=psycopg2.connect(url,connect_timeout=15)
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_schema(), clock_timestamp(), current_setting('transaction_read_only')")
            identity=cur.fetchone()
            report['database'].update(connected_database=identity[0],schema=identity[1],clock=identity[2].isoformat(),read_only=identity[3])
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
            tables=[r[0] for r in cur.fetchall()]
            report['database']['forward_tables']=[t for t in tables if t.startswith('ft_')]
            report['database']['gamma_lens_tables_present']=all(t in tables for t in ['weekly_spx','weekly_underlying','gex_snapshots','saved_models'])
            depths={}
            for ticker in UNIVERSE:
                if ticker=='SPX':cur.execute('SELECT COUNT(*),MIN(week_start),MAX(week_start) FROM weekly_spx')
                else:cur.execute('SELECT COUNT(*),MIN(week_start),MAX(week_start) FROM weekly_underlying WHERE ticker=%s',(ticker,))
                n,first,last=cur.fetchone()
                depths[ticker]={'rows':n,'first':str(first),'last':str(last)}
            report['database']['legacy_weekly_depth']=depths
        conn.rollback()
        conn.close()
    except Exception as exc:
        report['blockers'].append('Database read-only identity check: '+type(exc).__name__)
    history_store=Store.postgres(url)
    history_store.conn.set_session(readonly=True)
    provider=TradierProvider(cfg.get('TRADIER_TOKEN'),clock=utcnow,history_loader=history_store.legacy_weekly)
    try:
        report['tradier_endpoint']=provider.client.base_url
        report['market_clock']=provider._get('clock').get('clock')
        # No account IDs/names/balances are recorded. Authentication to the
        # Brokerage API verifies account access, not measured current latency.
        response=provider.http.get(provider.client.base_url+'/user/profile',timeout=20)
        response.raise_for_status()
        accounts=response.json().get('profile',{}).get('account',[])
        if isinstance(accounts,dict):accounts=[accounts]
        report['brokerage_account_access']={'authenticated':True,'account_count':len(accounts)}
        quotes=provider.quotes
        for ticker in (*UNIVERSE,'VIX'):
            q=quotes.get(ticker,{})
            stamp=epoch_time(q.get('trade_date',0))
            report['tickers'][ticker]={'quote_symbol':q.get('symbol'),'quote_type':q.get('type'),
                'open':q.get('open'),'last':q.get('last'),'trade_at':stamp.isoformat(),
                'age_seconds_at_check':round((utcnow()-stamp).total_seconds(),1),
                'opening_freshness_certified':False,
                'provider_delay_fields':{k:v for k,v in q.items() if 'delay' in k.lower() or 'realtime' in k.lower()}}
            if q.get('symbol')!=ticker or not q.get('open') or not q.get('last'):
                report['blockers'].append(ticker+' quote missing/invalid')
        for ticker in UNIVERSE:
            item=report['tickers'][ticker]
            try:
                weekly,daily,evidence=provider.training_history(ticker,week,review_end=end)
                evidence_path=Path(args.output).with_name(Path(args.output).stem+'-'+ticker+'-history.json')
                evidence_path.write_text(json.dumps(evidence,indent=2,allow_nan=False),encoding='utf-8')
                for interval,frame in [('weekly',weekly),('daily',daily)]:
                    item[interval+'_history']={'rows':len(frame),'first':str(frame.index.min().date()),
                        'last':str(frame.index.max().date()),'valid':True,'policy':evidence['policy_version']}
                item['history_decisions']={
                    'reviewed':len(evidence['decisions']),
                    'whole_bar_replacements':sum(d['accepted_source']=='alternative' for d in evidence['decisions']),
                    'unused_daily_ohl_warnings':len(evidence['unused_daily_ohl_warnings']),
                    'full_evidence':evidence_path.name,'sha256':hashlib.sha256(evidence_path.read_bytes()).hexdigest()}
            except Exception as exc:
                item['history_error']=type(exc).__name__+': '+str(exc).split('?')[0][:180]
                report['blockers'].append(ticker+' history: '+item['history_error'])
            # Capabilities remain independently testable when history fails.
            try:
                expirations=provider.client.get_expirations(ticker)
                expiration=listed_week_expiration(expirations,week)
                if not expiration:raise ValueError('No listed final-session expiry')
                chain,root=contract_chain(provider.client.get_chain_once(ticker,expiration),ticker,expiration)
                item['contracts']={'expiration':expiration,'root':root,'calls':len(chain['calls']),
                    'puts':len(chain['puts']),'standard_size':100,'sample_symbol':chain['calls'][0]['symbol']}
                # A completed session probes minute-history capability without
                # reclassifying any historic bar as a prospective forecast.
                minute_start=prior.sessions[-1].open+timedelta(minutes=15)
                observation=provider.observe(ticker,prior.sessions[-1],minute_start)
                item['minute_history']={'session':str(end),'bars':len(observation['minutes']),
                    'error':observation.get('minute_error'),'daily_valid':valid_ohlc(observation.get('daily'))}
                if observation.get('minute_error') or not observation['minutes']:
                    report['blockers'].append(ticker+' minute history unavailable')
            except Exception as exc:
                item['error']=type(exc).__name__+': '+str(exc).split('?')[0][:180]
                report['blockers'].append(ticker+' source/contract check: '+item['error'])
        for index in ('VIX','VIX9D','VIX3M'):
            try:
                frame=fetch_cboe_index_history(index)
                report['sources'][index]={'rows':len(frame),'first':str(frame.index.min().date()),
                    'last':str(frame.index.max().date()),'prior_final_session_present':pd.Timestamp(end) in frame.index}
                if pd.Timestamp(end) not in frame.index:report['blockers'].append(index+' prior final-session data not yet available')
            except Exception as exc:report['blockers'].append(index+' source: '+type(exc).__name__)
        from fredapi import Fred
        fred=Fred(api_key=cfg.get('FRED_API_KEY'))
        for series in ('DGS10','DGS2','DFF'):
            try:
                values=fred.get_series(series,observation_start=start,observation_end=end).dropna()
                report['sources'][series]={'rows':len(values),'first':str(values.index.min().date()),'last':str(values.index.max().date())}
                if values.empty or values.index.max().date()<end-timedelta(days=7):report['blockers'].append(series+' stale/missing')
            except Exception as exc:report['blockers'].append(series+' source: '+type(exc).__name__)
    finally:
        provider.close()
        history_store.close()
    report['capabilities_pass']=not report['blockers']
    text=json.dumps(report,indent=2,default=str)
    Path(args.output).write_text(text,encoding='utf-8')
    print(text)
    return 0 if report['capabilities_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
