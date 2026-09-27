"""Postgres schema-initialization lock behavior.

The lock is TRANSACTION-scoped (pg_advisory_xact_lock inside one explicit
BEGIN…COMMIT). DATABASE_URL points at Neon's pooled endpoint — PgBouncer in
transaction mode — and the wrapper runs in autocommit, so a SESSION lock's
acquire and unlock could land on different server connections: unlock
reported "not held", the lock leaked, and every Monday weekly setup since
2026-08-31 failed (fail-fast then cancelled the SPX job). A transaction pins
one server connection and releases the lock at COMMIT/ROLLBACK.
"""
import pytest

import range_finder.db as db


class _RawCursor:
    def __init__(self, events, *, acquire_error=None):
        self.events = events
        self.acquire_error = acquire_error
        self._result = None

    def execute(self, sql, params=None):
        verb = sql.strip().split()[0].upper()
        if verb in ("BEGIN", "COMMIT", "ROLLBACK"):
            self.events.append((verb.lower(), sql, params))
            return None
        if "pg_advisory_xact_lock" in sql:
            self.events.append(("acquire", sql, params))
            if self.acquire_error is not None:
                raise self.acquire_error
            self._result = (None,)
            return None
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._result


class _Connection:
    def __init__(self, events, **cursor_kwargs):
        self.events = events
        self.cursor_kwargs = cursor_kwargs

    def cursor(self):
        return db.PGCursor(_RawCursor(self.events, **self.cursor_kwargs))


def test_advisory_lock_is_transaction_scoped_and_casts_key_to_bigint():
    events = []

    db._acquire_init_advisory_lock(_Connection(events))

    kind, sql, params = events[0]
    assert kind == "acquire"
    assert sql == "SELECT pg_advisory_xact_lock(CAST(%s AS bigint))"
    # Preserve the global adapter behavior; the SQL's explicit bigint cast is
    # the narrow correction for this overload-sensitive Postgres function.
    assert params == (float(db._INIT_ADVISORY_LOCK_KEY),)


def test_init_runs_lock_and_body_in_one_transaction(monkeypatch):
    events = []
    conn = _Connection(events)
    monkeypatch.setattr(db, "_init_all_tables_body", lambda _conn: events.append(("body",)))

    db.init_all_tables(conn)

    assert [event[0] for event in events] == ["begin", "acquire", "body", "commit"]


def test_init_rolls_back_when_schema_body_fails(monkeypatch):
    events = []
    conn = _Connection(events)

    def fail_body(_conn):
        events.append(("body",))
        raise ValueError("DDL failed")

    monkeypatch.setattr(db, "_init_all_tables_body", fail_body)

    with pytest.raises(ValueError, match="DDL failed"):
        db.init_all_tables(conn)

    # ROLLBACK both undoes partial DDL and releases the xact lock.
    assert [event[0] for event in events] == ["begin", "acquire", "body", "rollback"]


def test_init_does_not_swallow_lock_acquisition_failure(monkeypatch):
    events = []
    conn = _Connection(events, acquire_error=RuntimeError("no bigint overload"))
    body_called = False

    def body(_conn):
        nonlocal body_called
        body_called = True

    monkeypatch.setattr(db, "_init_all_tables_body", body)

    with pytest.raises(RuntimeError, match="Failed to acquire Postgres schema init lock"):
        db.init_all_tables(conn)

    assert body_called is False
    assert [event[0] for event in events] == ["begin", "acquire", "rollback"]


def test_body_commit_call_is_not_relied_on(monkeypatch):
    """psycopg2's commit() is a no-op under autocommit, so the explicit
    COMMIT must come from init_all_tables itself, after the body."""
    events = []
    conn = _Connection(events)

    def body(_conn):
        events.append(("body",))

    monkeypatch.setattr(db, "_init_all_tables_body", body)
    db.init_all_tables(conn)
    assert events[-1][0] == "commit"
