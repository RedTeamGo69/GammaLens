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
    executed = [c[0] for c in fc.calls]
    # Atomic swap: BEGIN, DELETE, INSERT, COMMIT.
    assert executed[0] == "BEGIN"
    assert "DELETE FROM trade_category_stats" in executed[1]
    insert_sql, params = fc.calls[2]
    assert insert_sql.count("?") == len(params) == len(_STATS_COLS)
    assert "COMMIT" in executed


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


# ── reconciliation: superseded-id cleanup + review-queue auto-resolve ─────────

import sqlite3


def _sqlite_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE strategy_history (
            strategy_id TEXT PRIMARY KEY, account_id TEXT, underlying TEXT,
            category TEXT, status TEXT, close_reason TEXT, opened_at TEXT,
            closed_at TEXT, expiration TEXT, dte_at_open INTEGER,
            n_legs INTEGER, legs_json TEXT, realized_pnl REAL, fees REAL,
            opened_via TEXT, rolled_from TEXT, grouping_version INTEGER,
            updated_at TEXT)
    """)
    conn.execute("""
        CREATE TABLE fill_review_queue (
            txn_id TEXT PRIMARY KEY, account_id TEXT, symbol TEXT,
            underlying TEXT, ts TEXT, reason TEXT, context TEXT,
            resolved INTEGER, updated_at TEXT)
    """)
    return conn


def test_delete_superseded_strategies_removes_id_churn_orphans():
    from public_api.store import delete_superseded_strategies
    conn = _sqlite_conn()
    for sid in ("old_vertical", "current_condor", "other_current"):
        conn.execute(
            "INSERT INTO strategy_history (strategy_id, account_id, status) "
            "VALUES (?, ?, 'open')", (sid, "ACCT"))
    # 'old_vertical' predates a leg-in merge — the fresh full sync only
    # regenerated the two current ids.
    removed = delete_superseded_strategies(
        conn, "ACCT", ["current_condor", "other_current"])
    assert removed == 1
    left = {r[0] for r in conn.execute(
        "SELECT strategy_id FROM strategy_history").fetchall()}
    assert left == {"current_condor", "other_current"}


def test_delete_superseded_refuses_empty_keep_set():
    from public_api.store import delete_superseded_strategies
    conn = _sqlite_conn()
    conn.execute(
        "INSERT INTO strategy_history (strategy_id, account_id, status) "
        "VALUES ('x', 'ACCT', 'open')", ())
    assert delete_superseded_strategies(conn, "ACCT", []) == 0
    assert conn.execute("SELECT COUNT(*) FROM strategy_history").fetchone()[0] == 1


def test_delete_superseded_scoped_to_account():
    from public_api.store import delete_superseded_strategies
    conn = _sqlite_conn()
    conn.execute("INSERT INTO strategy_history (strategy_id, account_id) "
                 "VALUES ('mine_old', 'ACCT')", ())
    conn.execute("INSERT INTO strategy_history (strategy_id, account_id) "
                 "VALUES ('theirs', 'OTHER')", ())
    delete_superseded_strategies(conn, "ACCT", ["mine_new"])
    left = {r[0] for r in conn.execute(
        "SELECT strategy_id FROM strategy_history").fetchall()}
    assert left == {"theirs"}


def test_resolve_stale_review_items_drains_grouped_fills():
    from public_api.store import resolve_stale_review_items
    conn = _sqlite_conn()
    for txn in ("t_still_odd", "t_now_grouped", "t_already_done"):
        conn.execute(
            "INSERT INTO fill_review_queue (txn_id, account_id, resolved) "
            "VALUES (?, 'ACCT', ?)",
            (txn, 1 if txn == "t_already_done" else 0))
    # Latest complete sync still flags only t_still_odd as ambiguous.
    n = resolve_stale_review_items(conn, "ACCT", ["t_still_odd"])
    assert n == 1
    rows = dict(conn.execute(
        "SELECT txn_id, resolved FROM fill_review_queue").fetchall())
    assert rows == {"t_still_odd": 0, "t_now_grouped": 1, "t_already_done": 1}


def test_sync_reconciles_superseded_rows_end_to_end():
    """Leg-in id churn scenario through the real sync: a pre-merge vertical
    row sits in the DB; after a full sync whose grouping merged everything
    into one condor, the orphan is gone and the condor row remains."""
    from public_api.sync import sync_public_history
    conn = _sqlite_conn()
    conn.execute(
        "INSERT INTO strategy_history (strategy_id, account_id, status) "
        "VALUES ('stale_pre_merge_id', 'TESTACCT', 'open')", ())
    # Columns mirror public_api.stats._STATS_COLS so the real upsert lands.
    conn.execute("""
        CREATE TABLE trade_category_stats (
            account_id TEXT, category TEXT, n_trades INTEGER,
            n_wins INTEGER, win_rate REAL, total_pnl REAL,
            avg_pnl REAL, avg_win REAL, avg_loss REAL, worst_loss REAL,
            best_trade REAL, avg_hold_days REAL,
            first_trade_date TEXT, last_trade_date TEXT,
            last_computed_at TEXT, updated_at TEXT,
            PRIMARY KEY (account_id, category))
    """)
    res = sync_public_history(conn, FakeClient(condor_open()),
                              as_of="2026-07-20")
    assert res is not None and res.errors == []
    assert res.superseded_removed == 1
    ids = {r[0] for r in conn.execute(
        "SELECT strategy_id FROM strategy_history").fetchall()}
    assert "stale_pre_merge_id" not in ids
    assert len(ids) == 1                       # the freshly-synced condor
