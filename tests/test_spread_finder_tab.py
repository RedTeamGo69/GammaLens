"""Tab-level harness: the real Spread Finder render against fakes.

Until this existed, the only AppTest stubbed the whole tab out, so nothing
checked that the render actually produces strikes, a blocked message, or the
regime-shift warning.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FIXTURE = Path(__file__).parent / "fixtures" / "spread_finder_app.py"


def _run(monkeypatch, mode):
    monkeypatch.setenv("GL_SF_MODE", mode)
    at = AppTest.from_file(str(FIXTURE)).run(timeout=60)
    assert not at.exception, at.exception
    return at


def _html(at) -> str:
    return " ".join(str(getattr(el, "proto", el)) for el in at.get("html"))


def test_ready_week_renders_the_forecast_cards(monkeypatch):
    at = _run(monkeypatch, "ready")
    assert not at.error
    html = _html(at)
    assert "Point Estimate" in html and "Effective Range" in html
    # Mon–Thu: the reference input is seeded from the persisted Monday anchor.
    assert at.number_input(key="sf_ref_price_SPX").value == pytest.approx(6000.0)


def test_missing_week_is_blocked_before_any_forecast(monkeypatch):
    at = _run(monkeypatch, "missing")
    errors = [e.value for e in at.error]
    assert any("no feature row exists for 2026-06-15" in e for e in errors)
    assert "Point Estimate" not in _html(at)


def test_vix_spike_shows_the_regime_shift_warning(monkeypatch):
    at = _run(monkeypatch, "spike")
    errors = [e.value for e in at.error]
    assert any("VIX regime shift detected (extreme)" in e for e in errors)
    assert "Point Estimate" in _html(at)          # still forecasts, with a warning
