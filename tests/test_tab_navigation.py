from pathlib import Path

from streamlit.testing.v1 import AppTest


FIXTURE = Path(__file__).parent / "fixtures" / "navigation_app.py"


def test_tab_callback_commits_route_before_render_and_prevents_deselect():
    at = AppTest.from_string('''
import streamlit as st
from ui_controls import render_tab_control
st.text(st.session_state.get("_tab_last", "gex"))
render_tab_control()
''').run()
    at.button_group(key="tab_seg").set_value("forward").run()
    assert at.text[0].value == "forward"
    at.button_group(key="tab_seg").set_value(None).run()
    assert at.text[0].value == "forward"
    assert at.button_group(key="tab_seg").value == "forward"
    at.button_group(key="tab_seg").set_value("gex").run()
    assert at.text[0].value == "gex"


def test_gex_forward_gex_preserves_controls_and_export():
    at = AppTest.from_file(str(FIXTURE)).run(timeout=30)
    assert not at.exception
    assert any(code.value.startswith("GL1") for code in at.code)
    at.button_group(key="exp_pills").set_value("week").run()
    calls = at.session_state["fixture_market_calls"]
    for _ in range(2):
        at.button_group(key="tab_seg").set_value("forward").run()
        assert not at.exception
        assert any(title.value == "Forward Test" for title in at.title)
        assert at.text_input(key="ticker_search")
        assert at.button_group(key="exp_pills").value == "week"
        assert at.session_state["fixture_market_calls"] == calls
        assert not at.code
        assert at.query_params["tab"] == ["forward"]
        at.button_group(key="ft_view").set_value("Detailed data").run()
        at.multiselect(key="ft_filter_ticker").select("AMD").run()
        assert at.session_state["fixture_market_calls"] == calls
        at.button_group(key="tab_seg").set_value("gex").run()
        assert not at.exception
        assert not at.title
        assert not at.multiselect
        assert any(code.value.startswith("GL1") for code in at.code)
        assert at.text_input(key="ticker_search")
        assert at.button_group(key="exp_pills").value == "week"
        assert at.query_params["tab"] == ["gex"]
        calls += 1
        assert at.session_state["fixture_market_calls"] == calls
    at.button_group(key="tab_seg").set_value("spread").run()
    assert not at.exception
    assert at.title[0].value == "Spread Finder fixture"
    assert not at.code
    at.button_group(key="tab_seg").set_value("forward").run()
    assert not at.exception
    assert at.title[0].value == "Forward Test"


def test_forward_deep_link_keeps_controls_without_market_access(monkeypatch):
    import streamlit_app
    import ui_forward_test
    monkeypatch.setattr(streamlit_app, "get_credentials", lambda: ("", ""))
    monkeypatch.setattr(ui_forward_test, "load_snapshot", lambda: ([], [], []))
    at = AppTest.from_string("from streamlit_app import main\nmain()")
    at.query_params.update(tab="forward", t="UNVALIDATED", exp="week")
    at.run(timeout=30)
    assert not at.exception
    assert any(title.value == "Forward Test" for title in at.title)
    assert at.text_input(key="ticker_search")
    assert not any("API Token" in control.label for control in at.text_input)
    at.button_group(key="tab_seg").set_value("gex").run()
    assert not at.exception
    assert not at.title
    assert any("API Token" in control.label for control in at.text_input)
    at.button_group(key="tab_seg").set_value("forward").run()
    assert any(title.value == "Forward Test" for title in at.title)
