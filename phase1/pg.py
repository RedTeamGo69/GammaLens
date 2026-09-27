"""The one raw Postgres connect path (Neon requires TLS).

DATABASE_URL resolves at call time through ``phase1.credentials``; callers
wrap the returned psycopg2 connection as they need (range_finder.db's
placeholder-translating wrapper, gex_history's plain autocommit use).
"""
from phase1 import credentials


def require_database_url() -> str:
    """DATABASE_URL, or a clear error when Postgres isn't configured."""
    url = credentials.database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. This app requires Postgres — set DATABASE_URL "
            "in Streamlit secrets or as an environment variable."
        )
    try:
        import psycopg2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "psycopg2 is not installed. This app requires Postgres — "
            "`pip install psycopg2-binary`."
        ) from e
    return url


def connect(url: "str | None" = None, *, autocommit: bool = True):
    """A new psycopg2 connection to ``url`` (default: DATABASE_URL).

    Autocommit by default so read paths never sit ``idle in transaction``
    (which keeps a Neon compute from auto-suspending); callers that need a
    transaction send BEGIN/COMMIT themselves.
    """
    import psycopg2
    conn = psycopg2.connect(url or require_database_url(), sslmode="require")
    conn.autocommit = autocommit
    return conn
