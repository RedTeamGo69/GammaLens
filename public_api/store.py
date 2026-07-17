"""
Neon persistence for grouped Public.com strategies — UPSERT-only writers on
a passed conn (DDL lives in range_finder/db.py::init_all_tables, matching
every other table in the app). Deterministic strategy ids make the full
repull sync idempotent: re-running lands on the same rows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from public_api.grouping import StrategyGroup

log = logging.getLogger(__name__)

_STRATEGY_COLS = [
    "strategy_id", "account_id", "underlying", "category", "status",
    "close_reason", "opened_at", "closed_at", "expiration", "dte_at_open",
    "n_legs", "legs_json", "realized_pnl", "fees", "opened_via",
    "rolled_from", "grouping_version", "updated_at",
]


def strategy_row(group: StrategyGroup, category: str, account_id: str,
                 now_iso: str | None = None) -> dict:
    now = now_iso or datetime.now(timezone.utc).isoformat()
    return {
        "strategy_id": group.strategy_id,
        "account_id": account_id,
        "underlying": group.underlying,
        "category": category,
        "status": group.status,
        "close_reason": group.close_reason,
        "opened_at": group.opened_at,
        "closed_at": group.closed_at,
        "expiration": group.expiration,
        "dte_at_open": group.dte_at_open,
        "n_legs": len(group.legs_open) + len(group.legs_close),
        "legs_json": json.dumps(
            {"open": [asdict(l) for l in group.legs_open],
             "close": [asdict(l) for l in group.legs_close]},
            default=str),
        "realized_pnl": group.realized_pnl,
        "fees": group.fees,
        "opened_via": group.opened_via,
        "rolled_from": group.rolled_from,
        "grouping_version": group.grouping_version,
        "updated_at": now,
    }


def upsert_strategies(conn, groups: list[StrategyGroup],
                      categories: dict[str, str], account_id: str) -> int:
    placeholders = ", ".join("?" for _ in _STRATEGY_COLS)
    updates = ",\n            ".join(
        f"{c} = excluded.{c}" for c in _STRATEGY_COLS if c != "strategy_id")
    n = 0
    for g in groups:
        row = strategy_row(g, categories.get(g.strategy_id, "other"), account_id)
        conn.execute(
            f"""
            INSERT INTO strategy_history ({", ".join(_STRATEGY_COLS)})
            VALUES ({placeholders})
            ON CONFLICT(strategy_id) DO UPDATE SET
                {updates}
            """,
            tuple(row[c] for c in _STRATEGY_COLS),
        )
        n += 1
    conn.commit()
    return n


def delete_superseded_strategies(conn, account_id: str,
                                 keep_ids: list[str]) -> int:
    """Remove strategy_history rows the current full repull did NOT regenerate.

    strategy_id is a hash of the OPENING txn-id set, so it churns whenever
    that set grows between syncs — a scale-in attaches a new opening leg, or
    the leg-in merge combines two verticals into one condor. Upsert-only
    persistence left the pre-churn row behind forever as a phantom 'open'
    strategy duplicating legs that also live inside the new row.

    Safe ONLY because get_all_history fails closed (returns None on any
    incomplete pagination): reaching persistence means the id set was
    computed from the COMPLETE account history, so a row absent from it is a
    superseded artifact, never a fill we failed to fetch. The empty-set
    guard is belt-and-braces — an account with zero true history has nothing
    to reconcile, and a pathological empty result must not wipe the table.
    """
    if not keep_ids:
        return 0
    placeholders = ", ".join("?" for _ in keep_ids)
    cur = conn.execute(
        f"""
        DELETE FROM strategy_history
        WHERE account_id = ? AND strategy_id NOT IN ({placeholders})
        """,
        (account_id, *keep_ids),
    )
    conn.commit()
    removed = getattr(cur, "rowcount", 0) or 0
    if removed:
        log.info(f"reconciled strategy_history: removed {removed} superseded row(s)")
    return removed


def resolve_stale_review_items(conn, account_id: str,
                               still_ambiguous_txn_ids: list[str]) -> int:
    """Auto-resolve queue rows whose fill the latest complete sync grouped.

    A fill lands in fill_review_queue when grouping couldn't place it at the
    time; a later fill often completes the picture and the next full repull
    groups it cleanly — but the queue row previously stayed 'in review'
    forever, so the UI counter never drained. Any unresolved row whose txn_id
    is NOT in the current sync's ambiguous set is no longer ambiguous.
    """
    now = datetime.now(timezone.utc).isoformat()
    ids = [t for t in still_ambiguous_txn_ids if t]
    if ids:
        placeholders = ", ".join("?" for _ in ids)
        cur = conn.execute(
            f"""
            UPDATE fill_review_queue SET resolved = 1, updated_at = ?
            WHERE account_id = ? AND resolved = 0 AND txn_id NOT IN ({placeholders})
            """,
            (now, account_id, *ids),
        )
    else:
        cur = conn.execute(
            """
            UPDATE fill_review_queue SET resolved = 1, updated_at = ?
            WHERE account_id = ? AND resolved = 0
            """,
            (now, account_id),
        )
    conn.commit()
    resolved = getattr(cur, "rowcount", 0) or 0
    if resolved:
        log.info(f"review queue: auto-resolved {resolved} no-longer-ambiguous row(s)")
    return resolved


def upsert_review_queue(conn, ambiguous: list[dict], account_id: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for item in ambiguous:
        conn.execute(
            """
            INSERT INTO fill_review_queue (
                txn_id, account_id, symbol, underlying, ts, reason, context,
                resolved, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(txn_id) DO UPDATE SET
                reason     = excluded.reason,
                context    = excluded.context,
                updated_at = excluded.updated_at
            """,
            (item.get("txn_id"), account_id, item.get("symbol"),
             item.get("underlying"), item.get("ts"), item.get("reason"),
             item.get("context"), now),
        )
        n += 1
    conn.commit()
    return n


def load_strategies(conn, account_id: str, limit: int = 1000) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT {", ".join(_STRATEGY_COLS)}
        FROM strategy_history
        WHERE account_id = ?
        ORDER BY opened_at DESC
        LIMIT ?
        """,
        (account_id, limit),
    ).fetchall()
    return [dict(zip(_STRATEGY_COLS, r)) for r in rows or []]


def resolve_account_id(conn, account_id: str | None = None) -> str:
    """Configured account id, else whatever single account the table holds
    (single-user app — the column exists for correctness, not tenancy)."""
    if account_id:
        return account_id
    try:
        row = conn.execute(
            "SELECT account_id FROM strategy_history LIMIT 1", ()).fetchone()
        return str(row[0]) if row else ""
    except Exception:
        return ""


def history_summary(conn, account_id: str | None = None) -> dict:
    """Small counts card for the UI: strategies / closed / in review."""
    out = {"strategies": 0, "closed": 0, "review": 0, "last_sync": None,
           "account_id": ""}
    account = resolve_account_id(conn, account_id)
    out["account_id"] = account
    if not account:
        return out
    try:
        cur = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END), "
            "MAX(updated_at) FROM strategy_history WHERE account_id = ?",
            (account,))
        row = cur.fetchone()
        if row:
            out["strategies"] = int(row[0] or 0)
            out["closed"] = int(row[1] or 0)
            out["last_sync"] = row[2]
        cur = conn.execute(
            "SELECT COUNT(*) FROM fill_review_queue "
            "WHERE account_id = ? AND resolved = 0", (account,))
        row = cur.fetchone()
        out["review"] = int(row[0] or 0) if row else 0
    except Exception as e:
        log.warning(f"history summary failed: {e}")
    return out


def load_review_queue(conn, account_id: str | None = None,
                      limit: int = 50) -> list[dict]:
    account = resolve_account_id(conn, account_id)
    if not account:
        return []
    cols = ["txn_id", "symbol", "underlying", "ts", "reason", "context"]
    try:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM fill_review_queue "
            f"WHERE account_id = ? AND resolved = 0 ORDER BY ts DESC LIMIT ?",
            (account, limit)).fetchall()
    except Exception as e:
        log.warning(f"review queue read failed: {e}")
        return []
    return [dict(zip(cols, r)) for r in rows or []]
