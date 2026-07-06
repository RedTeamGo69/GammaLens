"""Unit tests for ui_url_state — the URL-backed nav-state persistence that
lets ticker / tab / expiration / recents / export-extras survive a Streamlit
Cloud session reset.

Targets the pure helpers (``_parse_list`` / ``_serialize_list`` /
``_rehydrate`` / ``_compute_url``) with plain dicts standing in for
``st.session_state`` and ``st.query_params`` — no Streamlit runtime needed
(the module only imports streamlit lazily inside the two entry points).
"""
import ui_url_state as us

# session_state keys the module manages, for round-trip assertions.
_NAV_KEYS = ("active_ticker", "_tab_last", "_exp_last",
             "recent_tickers", "_sf_xlsx_extra")


def test_parse_list_basic():
    assert us._parse_list("AAPL,TSLA,NVDA") == ["AAPL", "TSLA", "NVDA"]


def test_parse_list_normalizes_and_dedupes():
    # lower-cases -> upper, trims whitespace, drops empties and duplicates
    assert us._parse_list(" aapl , TSLA ,, aapl ") == ["AAPL", "TSLA"]


def test_parse_list_empty_inputs():
    assert us._parse_list("") == []
    assert us._parse_list(None) == []


def test_parse_list_is_bounded():
    raw = ",".join(f"T{i}" for i in range(100))
    assert len(us._parse_list(raw)) == us._MAX_LIST_ITEMS


def test_serialize_list():
    assert us._serialize_list(["AAPL", "TSLA"]) == "AAPL,TSLA"
    assert us._serialize_list([]) == ""
    assert us._serialize_list(None) == ""


def test_rehydrate_seeds_all_managed_keys():
    session = {}
    params = {"t": "AMD", "tab": "spread", "exp": "week",
              "recents": "AMD,NVDA", "xtra": "TSLA,GOOGL"}
    us._rehydrate(session, params)
    assert session["active_ticker"] == "AMD"
    assert session["_tab_last"] == "spread"
    assert session["_exp_last"] == "week"
    assert session["recent_tickers"] == ["AMD", "NVDA"]
    assert session["_sf_xlsx_extra"] == ["TSLA", "GOOGL"]


def test_rehydrate_never_clobbers_live_session():
    # A live widget interaction earlier this run wins over the URL.
    session = {"active_ticker": "SPX", "recent_tickers": ["QQQ"]}
    us._rehydrate(session, {"t": "AMD", "recents": "AMD,NVDA"})
    assert session["active_ticker"] == "SPX"
    assert session["recent_tickers"] == ["QQQ"]


def test_rehydrate_is_once_per_session():
    session = {}
    us._rehydrate(session, {"t": "AMD"})
    assert session["active_ticker"] == "AMD"
    # second call is a no-op even though the URL changed
    us._rehydrate(session, {"t": "TSLA"})
    assert session["active_ticker"] == "AMD"


def test_rehydrate_with_no_params_invents_nothing():
    session = {}
    us._rehydrate(session, {})
    assert session == {us._REHYDRATED_FLAG: True}


def test_compute_url_serializes_current_state():
    session = {"active_ticker": "AMD", "_tab_last": "spread", "_exp_last": "week",
               "recent_tickers": ["AMD", "NVDA"], "_sf_xlsx_extra": ["TSLA"]}
    assert us._compute_url(session) == {
        "t": "AMD", "tab": "spread", "exp": "week",
        "recents": "AMD,NVDA", "xtra": "TSLA",
    }


def test_compute_url_omits_empty_and_missing():
    session = {"active_ticker": "SPX", "recent_tickers": [], "_sf_xlsx_extra": []}
    params = us._compute_url(session)
    assert params == {"t": "SPX"}
    assert "recents" not in params and "xtra" not in params and "exp" not in params


def test_round_trip_url_to_fresh_session():
    session = {"active_ticker": "AMD", "_tab_last": "spread", "_exp_last": "opex",
               "recent_tickers": ["AMD", "NVDA"], "_sf_xlsx_extra": ["TSLA", "GOOGL"]}
    params = us._compute_url(session)
    fresh = {}
    us._rehydrate(fresh, params)
    for k in _NAV_KEYS:
        assert fresh[k] == session[k]
