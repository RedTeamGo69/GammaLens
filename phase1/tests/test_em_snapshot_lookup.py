"""get_em_snapshot must never return a row of another EM type.

A legacy fallback re-ran the lookup WITHOUT the em_type filter whenever the
first query errored; on a Monday the daily and weekly rows share a date, so a
weekly lookup could come back with the (much narrower) daily EM.
"""
import phase1.gex_history as gh


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


class _FailingCursor:
    def execute(self, sql, params):
        raise RuntimeError("statement timeout")


class _RowCursor:
    def __init__(self):
        self.params = None

    def execute(self, sql, params):
        self.params = params

    def fetchone(self):
        return (40.0, 0.6, 6740.0, 6660.0, 6700.0, 6700.0, "2026-10-12T09:31:00")


def test_query_error_is_absence_not_another_row(monkeypatch):
    conns = []
    monkeypatch.setattr(gh, "_pg_get_connection",
                        lambda: conns.append(_Conn(_FailingCursor())) or conns[-1])

    assert gh.get_em_snapshot("2026-10-12", ticker="SPX", em_type="weekly") is None
    assert len(conns) == 1 and conns[0].closed


def test_lookup_filters_by_em_type(monkeypatch):
    cursor = _RowCursor()
    monkeypatch.setattr(gh, "_pg_get_connection", lambda: _Conn(cursor))

    snap = gh.get_em_snapshot("2026-10-12", ticker="SPX", em_type="weekly")

    assert cursor.params == ("2026-10-12", "SPX", "weekly")
    assert snap["expected_move_pts"] == 40.0 and snap["straddle"] == {"strike": 6700.0}
