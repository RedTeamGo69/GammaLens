"""The scheduler's unchanged CLI command targets the new registration plus
scoring-only studies, never a retired one."""
from datetime import date
import sqlite3
import sys

import forward_test
from range_finder.forward_test.config import DEFAULT_STUDY_ID, SCORING_ONLY_STUDIES, UNIVERSE
from range_finder.forward_test.store import Store
from range_finder.tests.forward_fixtures import Clock, FixtureProvider
from phase1.trading_week import trading_week


def test_default_cli_targets_restart_only(tmp_path, monkeypatch):
    from range_finder.forward_test import config, provider
    path = tmp_path/'restart.sqlite'
    def connect(*args):
        return Store(sqlite3.connect(path, isolation_level=None))
    clock = Clock(trading_week(date(2026,9,21)).evaluation_close)
    store = connect()
    store.migrate()
    store.register(DEFAULT_STUDY_ID, '2026-10-05', clock(), {'universe':UNIVERSE})
    (scoring_only, _), = SCORING_ONLY_STUDIES.items()
    store.register(scoring_only, '2026-09-28', clock(), {'universe':('SPY', 'NDX')})
    store.register('spread-finder-weekly-v1', '2026-09-14', clock(), {'fixture':True})
    store.close()
    fixture = FixtureProvider(clock)
    fixture.close = lambda: None
    monkeypatch.setattr(Store, 'postgres', connect)
    monkeypatch.setattr(config, 'utcnow', clock)
    monkeypatch.setattr(provider, 'TradierProvider', lambda *a, **kw: fixture)
    monkeypatch.setattr(sys, 'argv', ['forward_test.py', 'run'])
    forward_test.main()
    store = connect()
    try:
        runs = store.query('SELECT study_id,status FROM ft_runs')
        assert runs == [{'study_id':DEFAULT_STUDY_ID, 'status':'ok'},
                        {'study_id':scoring_only, 'status':'ok'}]
        assert not store.query('SELECT * FROM ft_slots')
    finally:
        store.close()
