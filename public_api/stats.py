"""
Per-category realized stats — the numbers the behavioral gate shows verbatim
("Your last N trades in this bucket: net −$X, win rate Y%").

Computed over CLOSED strategies only (open positions have no realized P&L)
and recomputed wholesale on every sync — this table is a derived cache of
strategy_history, never a source of truth.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_STATS_COLS = [
    "account_id", "category", "n_trades", "n_wins", "win_rate", "total_pnl",
    "avg_pnl", "avg_win", "avg_loss", "worst_loss", "best_trade",
    "avg_hold_days", "first_trade_date", "last_trade_date",
    "last_computed_at", "updated_at",
]


def _hold_days(opened_at: str, closed_at: str | None) -> float | None:
    try:
        o = datetime.fromisoformat(str(opened_at))
        c = datetime.fromisoformat(str(closed_at))
        if o.tzinfo is None:
            o = o.replace(tzinfo=timezone.utc)
        if c.tzinfo is None:
            c = c.replace(tzinfo=timezone.utc)
        return max(0.0, (c - o).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return None


def compute_category_stats(strategies: list, categories: dict[str, str]) -> dict:
    """{category_key: stats dict} over closed strategies with realized P&L.
    ``strategies`` are StrategyGroup objects (or dicts with the same fields)."""
    buckets: dict[str, list] = {}
    for s in strategies:
        get = (lambda k, _s=s: getattr(_s, k, None)) if not isinstance(s, dict) \
            else (lambda k, _s=s: _s.get(k))
        if get("status") != "closed" or get("realized_pnl") is None:
            continue
        cat = categories.get(get("strategy_id"), "other")
        buckets.setdefault(cat, []).append({
            "pnl": float(get("realized_pnl")),
            "opened_at": get("opened_at"),
            "closed_at": get("closed_at"),
        })

    out: dict[str, dict] = {}
    for cat, rows in buckets.items():
        pnls = [r["pnl"] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        holds = [h for h in (_hold_days(r["opened_at"], r["closed_at"])
                             for r in rows) if h is not None]
        dates = sorted(str(r["opened_at"])[:10] for r in rows)
        out[cat] = {
            "n_trades": len(pnls),
            "n_wins": len(wins),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else None,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "worst_loss": round(min(pnls), 2) if pnls else None,
            "best_trade": round(max(pnls), 2) if pnls else None,
            "avg_hold_days": round(sum(holds) / len(holds), 2) if holds else None,
            "first_trade_date": dates[0] if dates else None,
            "last_trade_date": dates[-1] if dates else None,
        }
    return out


def upsert_category_stats(conn, account_id: str, stats: dict) -> int:
    """Replace the account's derived stats wholesale (categories that no
    longer exist after a re-group must not linger)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM trade_category_stats WHERE account_id = ?",
                 (account_id,))
    placeholders = ", ".join("?" for _ in _STATS_COLS)
    n = 0
    for cat, s in stats.items():
        row = {"account_id": account_id, "category": cat,
               "last_computed_at": now, "updated_at": now, **s}
        conn.execute(
            f"INSERT INTO trade_category_stats ({', '.join(_STATS_COLS)}) "
            f"VALUES ({placeholders})",
            tuple(row.get(c) for c in _STATS_COLS),
        )
        n += 1
    conn.commit()
    return n


def load_category_stats(conn, category: str, account_id: str | None = None) -> dict | None:
    """Stats row for one category key, or None when nothing is synced yet.
    account_id None → whatever single account the table holds (this is a
    single-user app; the column exists for correctness, not multi-tenancy)."""
    try:
        if account_id:
            cur = conn.execute(
                f"SELECT {', '.join(_STATS_COLS)} FROM trade_category_stats "
                f"WHERE account_id = ? AND category = ?", (account_id, category))
        else:
            cur = conn.execute(
                f"SELECT {', '.join(_STATS_COLS)} FROM trade_category_stats "
                f"WHERE category = ? LIMIT 1", (category,))
        row = cur.fetchone()
    except Exception as e:
        log.warning(f"category stats read failed: {e}")
        return None
    if not row:
        return None
    return dict(zip(_STATS_COLS, row))
