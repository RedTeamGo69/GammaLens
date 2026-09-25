"""The scheduler's unchanged CLI command must target the new registration."""
from datetime import date
import sqlite3
import sys

import forward_test
from range_finder.forward_test.config import DEFAULT_STUDY_ID, UNIVERSE
from range_finder.forward_test.store import Store
from range_finder.tests.forward_fixtures import Clock, FixtureProvider
from range_finder.trading_week import trading_week


def test_default_cli_targets_restart_only(tmp_path, monkeypatch):
    from range_finder.forward_test import config, provider
    path = tmp_path/'restart.sqlite'
    def connect(*args):
        return Store(sqlite3.connect(path, isolation_level=None))
    clock = Clock(trading_week(date(2026,9,21)).evaluation_close)
    store = connect()
    store.migrate()
    store.register(DEFAULT_STUDY_ID, '2026-09-28', clock(), {'universe':UNIVERSE})
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
        assert runs == [{'study_id':DEFAULT_STUDY_ID, 'status':'ok'}]
        assert not store.query('SELECT * FROM ft_slots')
    finally:
        store.close()
