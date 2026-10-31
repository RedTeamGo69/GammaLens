from datetime import date, timedelta
from io import BytesIO
import sqlite3

import pytest

from phase1.trading_week import trading_week
from range_finder.forward_test.marks import sample_marks
from range_finder.forward_test.results import (build_workbook, condor_managed_return, condor_return,
                                               load_results)
from range_finder.forward_test.runner import run_study
from range_finder.forward_test.store import Store
from range_finder.tests.forward_fixtures import Clock, FixtureProvider

# Entry at natural fills: credit = (1.0 - 0.4) + (0.8 - 0.3) = 1.1, max risk = 5 - 1.1.
ROW = {"close_eligible": True, "recommended_width": 5.0, "put_short": 95.0, "call_short": 105.0,
       "final_close": 100.0,
       "short_contracts": {"put": {"bid": 1.0}, "call": {"bid": 0.8}},
       "wings": {"put": [{"width": 5.0, "ask": 0.4}], "call": [{"width": 5.0, "ask": 0.3}]}}
CREDIT, RISK = 1.1, 3.9


def sample(at, put_short_ask, call_short_ask, put_wing_bid=0.0, call_wing_bid=0.0):
    return [{"collected_at": at, "leg": "put_short", "ask": put_short_ask, "bid": None},
            {"collected_at": at, "leg": "call_short", "ask": call_short_ask, "bid": None},
            {"collected_at": at, "leg": "put_wing", "bid": put_wing_bid, "ask": None},
            {"collected_at": at, "leg": "call_wing", "bid": call_wing_bid, "ask": None}]


def test_stop_exits_at_first_mark_past_the_credit_multiple():
    marks = sample("2026-11-02T19:48:00+00:00", 1.0, 0.5) + sample("2026-11-03T19:48:00+00:00", 2.6, 0.2, 0.3)
    # Day 1 debit 1.5 sits between target (0.55) and stop (2.2); day 2 debit 2.5 stops out.
    assert condor_managed_return({**ROW, "marks": marks}) == pytest.approx((CREDIT - 2.5) / RISK)
    # Stop disabled: held to a close inside the shorts, keeping the full credit.
    assert condor_managed_return({**ROW, "marks": marks}, stop=None) == pytest.approx(CREDIT / RISK)


def test_target_exits_and_the_earliest_trigger_wins():
    marks = sample("2026-11-04T19:48:00+00:00", 3.0, 0.5) + sample("2026-11-03T19:48:00+00:00", 0.3, 0.2)
    # The 11-03 target (debit 0.5 <= 0.55) precedes the 11-04 stop, whatever the row order.
    assert condor_managed_return({**ROW, "marks": marks}) == pytest.approx((CREDIT - 0.5) / RISK)


def test_no_trigger_or_untradable_sample_falls_back_to_hold_to_close():
    incomplete = [m for m in sample("2026-11-03T19:48:00+00:00", 9.0, 9.0) if m["leg"] != "call_wing"]
    for marks in ([], sample("2026-11-03T19:48:00+00:00", 1.0, 0.5), incomplete):
        row = {**ROW, "final_close": 106.0, "marks": marks}
        assert condor_managed_return(row) == pytest.approx(condor_return(row)) == pytest.approx((CREDIT - 1) / RISK)
    # A credit multiple means nothing without a credit.
    debit_entry = {**ROW, "short_contracts": {"put": {"bid": 0.1}, "call": {"bid": 0.1}}}
    assert condor_return(debit_entry) is not None and condor_managed_return(debit_entry) is None


@pytest.fixture
def frozen(tmp_path):
    week = trading_week(date(2026, 9, 7))  # Labor Day -> Tuesday
    store = Store(sqlite3.connect(tmp_path / 'isolated.sqlite', isolation_level=None))
    store.migrate()
    store.register('fixture', str(week.monday), week.capture_start, {'fixture': True})
    clock = Clock(week.capture_start + timedelta(seconds=20))
    assert run_study(store, FixtureProvider(clock), 'fixture', clock=clock, model_version='fixture-v1')['captured_models'] == 16
    yield store, week, clock
    store.close()


def test_sampler_dedupes_quotes_and_marks_are_immutable(frozen):
    store, week, clock = frozen
    calls = []

    def quotes(symbols):
        calls.append(symbols)
        stamp = int(clock().timestamp() * 1000)
        unquoted = next(s for s in symbols if len(s) > 10)
        book = {s: {"symbol": s, "bid": 1.0, "ask": 1.2, "bid_date": stamp, "ask_date": stamp}
                for s in symbols if s != unquoted}
        return {**book, "SPY": {"symbol": "SPY", "last": 101.0, "trade_date": stamp}}

    clock.value = week.sessions[0].close - timedelta(days=1)  # Labor Day: a holiday dispatch is a no-op
    assert sample_marks(store, quotes, ['fixture'], clock=clock)['skipped'] and not calls
    clock.value = week.sessions[0].open + timedelta(hours=2)
    with pytest.raises(ValueError, match='marks window'):
        sample_marks(store, quotes, ['fixture'], clock=clock)
    clock.value = week.sessions[0].close - timedelta(minutes=12)
    result = sample_marks(store, quotes, ['fixture', 'not-registered'], clock=clock)
    assert len(calls) == 1 and len(calls[0]) == len(set(calls[0])) == result['symbols']
    assert result['forecasts'] == 64 and result['rows'] > 64
    assert result['errors']  # the one symbol the fake did not quote is reported, not written
    rows = store.query("SELECT * FROM ft_marks")
    assert len(rows) == result['rows'] and {r['leg'] for r in rows} == {'put_short', 'call_short', 'put_wing', 'call_wing'}
    assert all(r['bid'] == 1.0 and r['ask'] == 1.2 and r['bid_at'] for r in rows)
    assert all(r['underlying_last'] == 101.0 for r in rows if r['symbol'].startswith('SPY'))
    for sql in ("UPDATE ft_marks SET bid=0", "DELETE FROM ft_marks"):
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            with store.transaction():
                store.execute(sql)
    loaded = [r for r in load_results(store, 'fixture') if r['marks']]
    assert sum(len(r['marks']) for r in loaded) == result['rows']
    from openpyxl import load_workbook
    sheet = load_workbook(BytesIO(build_workbook(loaded)))['Breach details']
    assert [c.value for c in sheet[4]][10:12] == ['Condor return', 'Condor managed return']
