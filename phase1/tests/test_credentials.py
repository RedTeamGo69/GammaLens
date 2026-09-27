"""Credentials resolve at call time: secrets first, then environment."""
import phase1.credentials as creds


class _Secrets(dict):
    pass


def test_environment_is_used_when_secrets_lack_the_key(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", _Secrets(), raising=False)
    monkeypatch.setenv("TRADIER_TOKEN", " env-tok ")
    assert creds.tradier_token() == "env-tok"


def test_secrets_win_over_environment(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", _Secrets(FRED_API_KEY="sec"), raising=False)
    monkeypatch.setenv("FRED_API_KEY", "env")
    assert creds.fred_api_key() == "sec"


def test_missing_everywhere_is_empty(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", _Secrets(), raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert creds.database_url() == ""


def test_resolution_happens_per_call(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "secrets", _Secrets(), raising=False)
    monkeypatch.setenv("TRADIER_TOKEN", "first")
    assert creds.tradier_token() == "first"
    monkeypatch.setenv("TRADIER_TOKEN", "second")
    assert creds.tradier_token() == "second"
