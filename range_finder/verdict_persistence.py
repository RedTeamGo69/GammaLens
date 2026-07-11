"""
Cockpit / Pre-Flight verdict persistence — the audit-log write path.

Mirrors spread_persistence.py: single-statement UPSERTs on a passed conn
(safe under the wrapper's autocommit), with all serialization and hashing
pure and separately testable.

The snapshot hash is the dedupe key and deliberately covers only the
DECISION state — verdict, reason codes, strikes, k/wing, and coarsely
rounded VRP/credit ratios. Volatile observation noise (live spot, exact
quote mids) is excluded, so a Monday morning of steady-state 90-second
refreshes collapses into a handful of rows while any change that would
alter what the user is being told (gate flip, snapped strike, slider move)
lands a fresh row. The stored row itself carries the full latest payload
for that decision state.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from range_finder.cockpit_engine import CockpitVerdict

log = logging.getLogger(__name__)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _round_or_none(value, digits: int) -> float | None:
    return round(value, digits) if isinstance(value, (int, float)) else None


def snapshot_hash(verdict: CockpitVerdict, week_start: str) -> str:
    """16-hex-char identity of the decision state (see module docstring)."""
    p = verdict.proposal
    payload = {
        "week_start": week_start,
        "ticker": verdict.inputs.get("ticker"),
        "friday_exp": verdict.inputs.get("friday_exp"),
        "anchor": verdict.inputs.get("anchor"),
        "verdict": verdict.verdict,
        "reason_codes": sorted(r.code for r in verdict.reasons),
        "vrp_ratio": _round_or_none(verdict.vrp.get("ratio"), 2),
        "k_used": verdict.inputs.get("k_used"),
        "strikes": [p.call_short, p.call_long, p.put_short, p.put_long] if p else None,
        "wing_width": p.wing_width if p else None,
        "credit_ratio": _round_or_none(p.credit_ratio, 2) if p else None,
        "config": verdict.inputs.get("config"),
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]


def build_cockpit_row(
    verdict: CockpitVerdict,
    *,
    ticker: str,
    week_start: str,
    model_name: str,
    now_iso: str | None = None,
) -> dict:
    """Pure verdict → column-dict serialization (the testable half of save)."""
    now = now_iso or datetime.now(timezone.utc).isoformat()
    p = verdict.proposal
    return {
        "ticker": ticker,
        "week_start": week_start,
        "snapshot_hash": snapshot_hash(verdict, week_start),
        "logged_at": now,
        "verdict": verdict.verdict,
        "reasons": _canonical([asdict(r) for r in verdict.reasons]),
        "model_name": model_name,
        "anchor": verdict.inputs.get("anchor"),
        "spot": verdict.inputs.get("spot"),
        "friday_exp": verdict.inputs.get("friday_exp"),
        "forecast_point_pct": verdict.inputs.get("forecast_point_pct"),
        "forecast_upper_pct": verdict.inputs.get("forecast_upper_pct"),
        "straddle_mid": verdict.vrp.get("straddle_mid"),
        "implied_range_pct": verdict.vrp.get("implied_range_pct"),
        "vrp_ratio": verdict.vrp.get("ratio"),
        "tripwire_ok": (None if verdict.vrp.get("tripwire_ok") is None
                        else int(bool(verdict.vrp.get("tripwire_ok")))),
        "call_wall": verdict.inputs.get("call_wall"),
        "put_wall": verdict.inputs.get("put_wall"),
        "zero_gamma": verdict.inputs.get("zero_gamma"),
        "events_json": _canonical(verdict.inputs.get("events") or []),
        "k_used": verdict.inputs.get("k_used"),
        "wing_width": p.wing_width if p else None,
        "call_short": p.call_short if p else None,
        "call_long": p.call_long if p else None,
        "put_short": p.put_short if p else None,
        "put_long": p.put_long if p else None,
        "credit": p.credit if p else None,
        "credit_ratio": p.credit_ratio if p else None,
        "max_loss": p.max_loss if p else None,
        "pop": p.pop if p else None,
        "proposal_json": _canonical(asdict(p)) if p else None,
        "config_json": _canonical(verdict.inputs.get("config") or {}),
        "updated_at": now,
    }


_COCKPIT_COLS = [
    "ticker", "week_start", "snapshot_hash", "logged_at", "verdict", "reasons",
    "model_name", "anchor", "spot", "friday_exp", "forecast_point_pct",
    "forecast_upper_pct", "straddle_mid", "implied_range_pct", "vrp_ratio",
    "tripwire_ok", "call_wall", "put_wall", "zero_gamma", "events_json",
    "k_used", "wing_width", "call_short", "call_long", "put_short", "put_long",
    "credit", "credit_ratio", "max_loss", "pop", "proposal_json", "config_json",
    "updated_at",
]

# Everything except the PK and first-seen stamp refreshes on conflict, so the
# row always holds the LATEST observation of this decision state while
# logged_at keeps when it first appeared.
_COCKPIT_UPDATE_COLS = [c for c in _COCKPIT_COLS
                        if c not in ("ticker", "week_start", "snapshot_hash",
                                     "logged_at")]


def save_cockpit_verdict(
    conn,
    verdict: CockpitVerdict,
    *,
    ticker: str,
    week_start: str,
    model_name: str,
) -> str:
    """UPSERT one cockpit verdict; returns its snapshot hash."""
    row = build_cockpit_row(verdict, ticker=ticker, week_start=week_start,
                            model_name=model_name)
    placeholders = ", ".join("?" for _ in _COCKPIT_COLS)
    updates = ",\n            ".join(f"{c} = excluded.{c}" for c in _COCKPIT_UPDATE_COLS)
    conn.execute(
        f"""
        INSERT INTO cockpit_verdicts ({", ".join(_COCKPIT_COLS)})
        VALUES ({placeholders})
        ON CONFLICT(ticker, week_start, snapshot_hash) DO UPDATE SET
            {updates}
        """,
        tuple(row[c] for c in _COCKPIT_COLS),
    )
    conn.commit()
    log.info(f"Cockpit verdict logged: {week_start} {ticker} "
             f"{row['verdict']} ({row['snapshot_hash']})")
    return row["snapshot_hash"]


_PREFLIGHT_COLS = [
    "checked_at", "input_hash", "ticker", "category", "expiration", "dte",
    "in_weekly_window", "week_start", "verdict", "trade_json",
    "behavioral_json", "model_json", "structure_json", "public_json",
    "reasons", "updated_at",
]


def save_preflight_log(conn, report_row: dict) -> None:
    """Append one Pre-Flight evaluation (dict keyed like _PREFLIGHT_COLS;
    missing keys land as NULL). Kept dict-shaped so this module doesn't
    import the preflight engine — the engine's to_row adapter owns the
    mapping."""
    row = {c: report_row.get(c) for c in _PREFLIGHT_COLS}
    row["updated_at"] = row["updated_at"] or datetime.now(timezone.utc).isoformat()
    placeholders = ", ".join("?" for _ in _PREFLIGHT_COLS)
    conn.execute(
        f"""
        INSERT INTO preflight_log ({", ".join(_PREFLIGHT_COLS)})
        VALUES ({placeholders})
        ON CONFLICT(checked_at, input_hash) DO UPDATE SET
            updated_at = excluded.updated_at
        """,
        tuple(row[c] for c in _PREFLIGHT_COLS),
    )
    conn.commit()


def load_recent_verdicts(conn, ticker: str, limit: int = 20) -> list[dict]:
    """Most recent cockpit verdict rows (latest decision states first)."""
    rows = conn.execute(
        f"""
        SELECT {", ".join(_COCKPIT_COLS)}
        FROM cockpit_verdicts
        WHERE ticker = ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (ticker, limit),
    ).fetchall()
    return [dict(zip(_COCKPIT_COLS, r)) for r in rows or []]


def load_last_proposal(conn, ticker: str) -> dict | None:
    """Latest persisted condor proposal for the ticker (survives a Streamlit
    Cloud session reset — the Pre-Flight tab's 'Load last cockpit proposal'
    path). Returns the parsed proposal dict plus its verdict context."""
    row = conn.execute(
        """
        SELECT week_start, verdict, proposal_json, updated_at
        FROM cockpit_verdicts
        WHERE ticker = ? AND proposal_json IS NOT NULL
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    if not row:
        return None
    week_start, verdict, proposal_json, updated_at = row
    try:
        proposal = json.loads(proposal_json)
    except (TypeError, ValueError):
        log.warning(f"Unparseable proposal_json for {ticker} {week_start}")
        return None
    return {
        "week_start": week_start,
        "verdict": verdict,
        "updated_at": updated_at,
        "proposal": proposal,
    }
