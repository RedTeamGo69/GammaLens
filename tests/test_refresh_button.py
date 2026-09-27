"""The ⟳ Refresh button clears the pipeline cache the app actually uses.

Regression: it did ``from streamlit_app import fetch_all_data`` — under
``streamlit run`` that imports a second copy of the app module whose cache
(Streamlit keys caches by ``__module__``) the running app never reads.
"""
import pytest

import ui_controls


class _Rerun(Exception):
    pass


def test_refresh_clears_data_caches_and_the_injected_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(ui_controls.st, "button", lambda *a, **k: True)
    monkeypatch.setattr(ui_controls.st.cache_data, "clear", lambda: calls.append("data"))

    def rerun():
        raise _Rerun
    monkeypatch.setattr(ui_controls.st, "rerun", rerun)
    with pytest.raises(_Rerun):
        ui_controls.render_refresh_button(clear_pipeline=lambda: calls.append("pipeline"))
    assert calls == ["data", "pipeline"]


def test_ui_modules_never_import_the_app_module():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("ui_controls.py", "ui_spread_finder.py", "ui_tv_export.py"):
        assert "from streamlit_app import" not in (root / name).read_text(encoding="utf-8"), name
