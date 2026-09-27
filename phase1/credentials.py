"""One resolver for API keys and connection strings.

Streamlit secrets first (Streamlit Cloud), then the environment (cron,
CLI, local shells). Resolved at CALL time, never at import: a module-level
read froze whatever was visible when the module was first imported, and
tests had to patch each module's private copy.

Callers go through the module attribute (``credentials.tradier_token()``),
so a test patches exactly one place.
"""
import os


def secret(name: str) -> str:
    """``name`` from st.secrets, else the environment, else ``""``."""
    value = ""
    try:
        import streamlit as st
        value = st.secrets.get(name, "") or ""
    except Exception:
        pass
    return str(value or os.environ.get(name, "")).strip()


def tradier_token() -> str:
    return secret("TRADIER_TOKEN")


def fred_api_key() -> str:
    return secret("FRED_API_KEY")


def database_url() -> str:
    return secret("DATABASE_URL")


def forward_test_database_url() -> str:
    """The forward study's database: its own FORWARD_TEST_DATABASE_URL when
    set, else the app's DATABASE_URL."""
    return secret("FORWARD_TEST_DATABASE_URL") or database_url()
