"""Primary history is retained unchanged, without a second provider's veto."""
from copy import deepcopy
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from range_finder.forward_test.history_policy import (
    HistoryUnavailable, _frame, validated_tradier_history)
from range_finder.tests.test_forward_history_policy import archived_windows
from range_finder.trading_week import trading_week


def validate(ticker, data=None, *, as_of=None):
    fixture = archived_windows()
    data = data or fixture['tickers'][ticker]
    return validated_tradier_history(ticker, date.fromisoformat(data['weekly'][0]['date']),
        date.fromisoformat(fixture['end']), date.fromisoformat(fixture['daily_start']),
        data['weekly'], data['daily'], as_of=as_of or datetime.fromisoformat(fixture['as_of']))


@pytest.mark.parametrize('ticker', ('SPX', 'SPY', 'AAPL', 'AMD'))
def test_archived_primary_prices_are_kept_without_replacements(ticker):
    data = deepcopy(archived_windows()['tickers'][ticker])
    # Even an unavailable corroborator cannot affect primary-source admission.
    data['yahoo'] = None
    before = deepcopy(data)
    weekly, daily, evidence = validate(ticker, data)
    pd.testing.assert_frame_equal(weekly, _frame(data['weekly']))
    pd.testing.assert_frame_equal(daily, _frame(data['daily']))
    assert data == before and not evidence['decisions']
    assert evidence['corroboration_required'] is False
    assert evidence['primary_daily'] == data['daily']


def test_spy_disputed_2020_close_is_accepted_as_supplied():
    data = deepcopy(archived_windows()['tickers']['SPY'])
    row = next(r for r in data['daily'] if r['date'] == '2020-09-17')
    row['close'] += .1
    _, daily, _ = validate('SPY', data)
    assert daily.at[pd.Timestamp('2020-09-17'), 'close'] == row['close']


@pytest.mark.parametrize('mutation,reason', [
    ('missing', '1 missing'), ('duplicate', 'duplicate primary'),
    ('future', 'unexpected dates'), ('weekly_high', 'invalid primary OHLC'),
    ('daily_close', 'invalid primary close'), ('mismatched_close', 'daily/weekly final close mismatch')])
def test_primary_data_defects_still_block_admission(mutation, reason):
    data = deepcopy(archived_windows()['tickers']['SPY'])
    if mutation == 'missing': data['daily'].pop(-3)
    elif mutation == 'duplicate': data['daily'].append(deepcopy(data['daily'][-3]))
    elif mutation == 'future': data['daily'].append({**data['daily'][-1], 'date':'2026-09-08'})
    elif mutation == 'weekly_high': data['weekly'][-3]['high'] = 0
    elif mutation == 'daily_close': data['daily'][-3]['close'] = 'NaN'
    else: data['daily'][-1]['close'] -= .1
    with pytest.raises(HistoryUnavailable, match=reason):
        validate('SPY', data)


def test_unclosed_week_cannot_enter_primary_training():
    with pytest.raises(HistoryUnavailable, match='has not completed'):
        validate('SPY', as_of=trading_week(date(2026,8,31)).evaluation_close-timedelta(seconds=1))
