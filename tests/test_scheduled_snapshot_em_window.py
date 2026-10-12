"""EM snapshots are first-write-wins, so only an opening-window run may save them.

A forced rebuild on a weekend used to save next week's weekly EM from
Friday's stale closing quotes; Monday's real opening capture then hit
ON CONFLICT DO NOTHING and was discarded.
"""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import scheduled_snapshot as snapshot

NY = ZoneInfo("America/New_York")


class _FakeClient:
    def __init__(self, token):
        pass

    def clear_cache(self):
        pass

    def get_expirations(self, ticker):
        return ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-16", "2026-10-23"]


def _calculation(**_):
    return SimpleNamespace(
        target_expirations=("2026-10-12", "2026-10-13", "2026-10-14", "2026-10-16"),
        spot=6700.0, spot_info={"source": "test"}, quote={},
        gex_df=pd.DataFrame({"strike": [6700.0], "net_gex": [1.0]}), stats={},
        levels={"zero_gamma": 6650.0, "call_wall": 6800.0, "put_wall": 6600.0},
        regime_info={"regime": "Positive Gamma"}, staleness_info={}, confidence_info={},
        em_analysis={"expected_move": {"expected_move_pts": 40.0,
                                       "upper_level": 6740.0, "lower_level": 6660.0}},
    )


@pytest.fixture
def forced_run(monkeypatch):
    saved = []
    monkeypatch.setenv("TICKER", "SPX")
    monkeypatch.setenv("FORCE_WEEKLY_SETUP", "true")
    monkeypatch.setattr(snapshot.credentials, "tradier_token", lambda: "token")
    monkeypatch.setattr(snapshot.credentials, "database_url", lambda: "postgres://test")
    monkeypatch.setattr(snapshot.credentials, "fred_api_key", lambda: "")
    monkeypatch.setattr("phase1.data_client.TradierDataClient", _FakeClient)
    monkeypatch.setattr("phase1.rates.fetch_risk_free_rate",
                        lambda key: {"rate": 0.04, "source": "test"})
    monkeypatch.setattr("phase1.gamma_archive_capture.calculate_gamma_archive_observation",
                        _calculation)
    monkeypatch.setattr("phase1.gamma_level_history.archive_computed_gamma_levels",
                        lambda **kw: False)
    monkeypatch.setattr("phase1.gex_history.get_em_snapshot", lambda *a, **k: None)
    monkeypatch.setattr("phase1.gex_history.save_em_snapshot",
                        lambda em, key, ticker, em_type: saved.append((em_type, key)))
    monkeypatch.setattr("phase1.expected_move.compute_em_for_expiration",
                        lambda client, ticker, exp, spot: {"expected_move_pts": 90.0})
    monkeypatch.setattr(snapshot, "_run_weekly_spread_setup", lambda *a, **k: True)
    monkeypatch.setattr(snapshot, "_weekly_setup_artifacts_complete", lambda t, w: True)

    def run(now):
        monkeypatch.setattr("phase1.market_clock.now_ny", lambda: now)
        snapshot.capture_snapshot()
        return saved
    return run


def test_forced_weekend_run_saves_no_em_snapshot(forced_run):
    assert forced_run(datetime(2026, 10, 11, 15, 0, tzinfo=NY)) == []


def test_forced_run_inside_the_opening_window_still_saves(forced_run):
    saved = forced_run(datetime(2026, 10, 12, 9, 40, tzinfo=NY))
    assert [em_type for em_type, _ in saved] == ["daily", "weekly", "monthly"]
