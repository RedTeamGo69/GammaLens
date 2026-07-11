"""Category stats + idempotent sync orchestration (public_api.stats / .sync)."""
import pytest

from public_api.grouping import group_fills
from public_api.stats import (
    _STATS_COLS,
    compute_category_stats,
    load_category_stats,
    upsert_category_stats,
)
from public_api.sync import sync_public_history
from tests.test_public_grouping import condor_open, fill


class FakeConn:
    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows or []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def commit(self):
        pass


def _closed_strategies():
    # winner: condor expires worthless (+65 credit kept)
    win = group_fills(condor_open(), as_of="2026-07-20").strategies[0]
    # loser: short put bought back higher
    lose_txns = [fill("l1", "2026-06-01T14:00:00+00:00",
                      "XSP260605P00700000", "SELL", 1, 80.00),
                 fill("l2", "2026-06-03T14:00:00+00:00",
                      "XSP260605P00700000", "BUY", 1, 200.00)]
    lose = group_fills(lose_txns, as_of="2026-06-10").strategies[0]
    # open position: excluded from stats
    open_g = group_fills([fill("o1", "2026-07-10T14:00:00+00:00",
                               "XSP260724P00700000", "SELL", 1, 90.00)],
                         as_of="2026-07-11").strategies[0]
    return win, lose, open_g


def test_compute_category_stats_closed_only():
    win, lose, open_g = _closed_strategies()
    cats = {win.strategy_id: "index|3-7DTE|iron_condor|credit",
            lose.strategy_id: "index|3-7DTE|single_leg|credit",
            open_g.strategy_id: "index|8-30DTE|single_leg|credit"}
    stats = compute_category_stats([win, lose, open_g], cats)

    assert set(stats) == {"index|3-7DTE|iron_condor|credit",
                          "index|3-7DTE|single_leg|credit"}   # open excluded
    condor = stats["index|3-7DTE|iron_condor|credit"]
    assert condor["n_trades"] == 1
    assert condor["total_pnl"] == pytest.approx(65.0)
    assert condor["win_rate"] == 1.0

    single = stats["index|3-7DTE|single_leg|credit"]
    assert single["total_pnl"] == pytest.approx(-120.0)
    assert single["worst_loss"] == pytest.approx(-120.0)
    assert single["n_wins"] == 0
    assert single["avg_hold_days"] == pytest.approx(2.0)


def test_compute_stats_aggregates_same_category():
    win, lose, _ = _closed_strategies()
    cats = {win.strategy_id: "bucket", lose.strategy_id: "bucket"}
    stats = compute_category_stats([win, lose], cats)["bucket"]
    assert stats["n_trades"] == 2
    assert stats["total_pnl"] == pytest.approx(65.0 - 120.0)
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["avg_win"] == pytest.approx(65.0)
    assert stats["avg_loss"] == pytest.approx(-120.0)


def test_upsert_category_stats_replaces_wholesale():
    fc = FakeConn()
    n = upsert_category_stats(fc, "TESTACCT", {"a|b|c|d": {"n_trades": 1}})
    assert n == 1
    assert "DELETE FROM trade_category_stats" in fc.calls[0][0]
    insert_sql, params = fc.calls[1]
    assert insert_sql.count("?") == len(params) == len(_STATS_COLS)


def test_load_category_stats_none_when_empty():
    assert load_category_stats(FakeConn(), "index|0DTE|single_leg|debit") is None


def test_load_category_stats_parses_row():
    row = tuple(None for _ in _STATS_COLS)
    fc = FakeConn(rows=[row])
    out = load_category_stats(fc, "x")
    assert set(out) == set(_STATS_COLS)


# ── sync orchestration ────────────────────────────────────────────────────────

class FakeClient:
    account_id = "TESTACCT"

    def __init__(self, txns):
        self._txns = txns

    def get_all_history(self):
        return self._txns


def test_sync_public_history_end_to_end():
    fc = FakeConn()
    res = sync_public_history(fc, FakeClient(condor_open()),
                              as_of="2026-07-20")
    assert res is not None
    assert res.trade_fills == 4
    assert res.strategies == 1
    assert res.closed == 1                     # expired via as_of
    assert res.coverage == 1.0
    assert res.categories == 1
    assert res.errors == []
    sql_text = " ".join(sql for sql, _ in fc.calls)
    assert "INSERT INTO strategy_history" in sql_text
    assert "trade_category_stats" in sql_text


def test_sync_returns_none_when_unreachable():
    class DeadClient:
        account_id = "TESTACCT"

        def get_all_history(self):
            return None

    assert sync_public_history(FakeConn(), DeadClient()) is None
