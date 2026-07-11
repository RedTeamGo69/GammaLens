"""Verdict persistence (range_finder.verdict_persistence).

The load-bearing property: the snapshot hash keys on the DECISION state
(verdict, reason codes, strikes, rounded ratios) and ignores observation
noise (live spot), so steady-state reruns dedupe while real drift lands a
new row. SQL is exercised against a fake conn — placeholder/param counts
and the conflict target are what break silently in refactors.
"""
import json

from range_finder.cockpit_config import CockpitConfig
from range_finder.cockpit_engine import evaluate_cockpit
from range_finder.verdict_persistence import (
    _COCKPIT_COLS,
    _PREFLIGHT_COLS,
    build_cockpit_row,
    load_last_proposal,
    save_cockpit_verdict,
    save_preflight_log,
    snapshot_hash,
)

ANCHOR = 600.0
WEEK = "2026-07-13"


def _chain():
    q = {}
    for k, (bid, ask) in {584: (2.9, 3.1), 589: (3.5, 3.7), 594: (5.0, 5.2),
                          606: (5.0, 5.2), 611: (3.5, 3.7), 616: (2.9, 3.1)}.items():
        q[float(k)] = {
            "call_bid": bid, "call_ask": ask, "call_oi": 500.0,
            "call_iv": 0.15, "call_delta": 0.12,
            "put_bid": bid, "put_ask": ask, "put_oi": 500.0,
            "put_iv": 0.15, "put_delta": -0.13,
        }
    return q


def make_verdict(spot=601.0, straddle=7.2, k=1.0):
    return evaluate_cockpit(
        ticker="XSP", anchor=ANCHOR, spot=spot, friday_exp="2026-07-17",
        forecast={"point_pct": 0.02, "upper_pct": 0.03},
        straddle_em_pts=straddle, gex_levels=None, events=[],
        chain_quotes=_chain(), cfg=CockpitConfig(k_expected_move=k),
        strike_increment=1, wing_widths=[5],
    )


class FakeConn:
    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows or []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def commit(self):
        pass


# ── snapshot hash: dedupe identity ────────────────────────────────────────────

def test_hash_deterministic():
    assert snapshot_hash(make_verdict(), WEEK) == snapshot_hash(make_verdict(), WEEK)


def test_spot_drift_does_not_change_hash():
    # A 90-second refresh where only live spot moved is the SAME decision
    assert snapshot_hash(make_verdict(spot=601.0), WEEK) == \
        snapshot_hash(make_verdict(spot=602.7), WEEK)


def test_strike_change_changes_hash():
    assert snapshot_hash(make_verdict(k=1.0), WEEK) != \
        snapshot_hash(make_verdict(k=1.25), WEEK)


def test_verdict_flip_changes_hash():
    assert snapshot_hash(make_verdict(straddle=7.2), WEEK) != \
        snapshot_hash(make_verdict(straddle=6.0), WEEK)   # vrp_thin SKIP


def test_week_start_changes_hash():
    v = make_verdict()
    assert snapshot_hash(v, "2026-07-13") != snapshot_hash(v, "2026-07-20")


# ── row serialization ─────────────────────────────────────────────────────────

def test_build_cockpit_row_maps_fields():
    v = make_verdict()
    row = build_cockpit_row(v, ticker="XSP", week_start=WEEK,
                            model_name="M3_extended",
                            now_iso="2026-07-13T13:35:00+00:00")
    assert row["verdict"] == "TRADE"
    assert row["call_short"] == 606 and row["put_short"] == 594
    assert row["anchor"] == ANCHOR
    assert row["model_name"] == "M3_extended"
    assert row["logged_at"] == "2026-07-13T13:35:00+00:00"
    assert row["snapshot_hash"] == snapshot_hash(v, WEEK)
    assert isinstance(json.loads(row["reasons"]), list)
    assert json.loads(row["proposal_json"])["call_short"] == 606
    assert row["tripwire_ok"] == 1
    assert set(row) == set(_COCKPIT_COLS)


def test_build_cockpit_row_skip_without_proposal():
    v = make_verdict()
    v_skip = evaluate_cockpit(
        ticker="XSP", anchor=None, spot=601.0, friday_exp="2026-07-17",
        forecast={"point_pct": 0.02}, straddle_em_pts=7.2, gex_levels=None,
        events=[], chain_quotes=_chain(), cfg=CockpitConfig(),
        strike_increment=1, wing_widths=[5],
    )
    row = build_cockpit_row(v_skip, ticker="XSP", week_start=WEEK,
                            model_name="M3_extended")
    assert row["verdict"] == "SKIP"
    assert row["proposal_json"] is None and row["call_short"] is None
    assert row["snapshot_hash"] != snapshot_hash(v, WEEK)


# ── SQL plumbing against a fake conn ──────────────────────────────────────────

def test_save_cockpit_verdict_sql_shape():
    fc = FakeConn()
    v = make_verdict()
    h = save_cockpit_verdict(fc, v, ticker="XSP", week_start=WEEK,
                             model_name="M3_extended")
    assert len(fc.calls) == 1
    sql, params = fc.calls[0]
    assert sql.count("?") == len(params) == len(_COCKPIT_COLS)
    assert "ON CONFLICT(ticker, week_start, snapshot_hash) DO UPDATE" in sql
    assert "logged_at = excluded" not in sql      # first-seen stamp preserved
    assert params[_COCKPIT_COLS.index("snapshot_hash")] == h


def test_save_preflight_log_sql_shape_and_null_fill():
    fc = FakeConn()
    save_preflight_log(fc, {"checked_at": "2026-07-13T14:00:00+00:00",
                            "input_hash": "abc123", "verdict": "GREEN"})
    sql, params = fc.calls[0]
    assert sql.count("?") == len(params) == len(_PREFLIGHT_COLS)
    assert params[_PREFLIGHT_COLS.index("verdict")] == "GREEN"
    assert params[_PREFLIGHT_COLS.index("category")] is None
    assert params[_PREFLIGHT_COLS.index("updated_at")] is not None


# ── loads ─────────────────────────────────────────────────────────────────────

def test_load_last_proposal_parses_json():
    fc = FakeConn(rows=[(WEEK, "TRADE",
                         json.dumps({"call_short": 606, "put_short": 594}),
                         "2026-07-13T14:00:00+00:00")])
    out = load_last_proposal(fc, "XSP")
    assert out["week_start"] == WEEK
    assert out["proposal"]["call_short"] == 606


def test_load_last_proposal_handles_garbage_and_empty():
    assert load_last_proposal(FakeConn(), "XSP") is None
    bad = FakeConn(rows=[(WEEK, "TRADE", "not-json", "ts")])
    assert load_last_proposal(bad, "XSP") is None
