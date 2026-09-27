"""One Postgres connect path, resolving DATABASE_URL at call time.

range_finder/db.py and phase1/gex_history.py each froze DATABASE_URL at
import (reading st.secrets from domain modules) and each carried its own
copy of the "is Postgres configured?" check.
"""
import sys
import types

import pytest

import phase1.pg as pg


@pytest.fixture
def fake_psycopg2(monkeypatch):
    calls = []

    class _Conn:
        autocommit = False

    def connect(url, **kw):
        calls.append((url, kw))
        return _Conn()

    monkeypatch.setitem(sys.modules, "psycopg2", types.SimpleNamespace(connect=connect))
    return calls


def test_missing_database_url_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("phase1.credentials.database_url", lambda: "")
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        pg.require_database_url()


def test_connect_resolves_url_per_call_with_ssl(monkeypatch, fake_psycopg2):
    urls = iter(["postgres://a", "postgres://b"])
    monkeypatch.setattr("phase1.credentials.database_url", lambda: next(urls))
    first = pg.connect()
    pg.connect(autocommit=False)
    assert fake_psycopg2 == [("postgres://a", {"sslmode": "require"}),
                             ("postgres://b", {"sslmode": "require"})]
    assert first.autocommit is True


def test_forward_test_url_prefers_its_own_setting(monkeypatch):
    import phase1.credentials as creds
    values = {"FORWARD_TEST_DATABASE_URL": "postgres://ft", "DATABASE_URL": "postgres://main"}
    monkeypatch.setattr(creds, "secret", lambda name: values.get(name, ""))
    assert creds.forward_test_database_url() == "postgres://ft"
    values.pop("FORWARD_TEST_DATABASE_URL")
    assert creds.forward_test_database_url() == "postgres://main"
