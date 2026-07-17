"""
Idempotent Public.com history sync: fetch → group → categorize → persist →
recompute stats. Callable from the Pre-Flight "Sync fills" button and from
the daily cron (scheduled_snapshot.py piggybacks it on the SPX matrix entry).

Full repull by design: the account's history starts 2026-04 and is small;
deterministic strategy ids make re-upserts land on the same rows, so there
is no cursor state to corrupt. The ``client`` only needs a
``get_all_history()`` method — the REST client satisfies it, and so does a
local-fixture shim for offline bootstraps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from public_api.categorize import category_key_for_group
from public_api.grouping import group_fills
from public_api.stats import compute_category_stats, upsert_category_stats
from public_api.store import (
    delete_superseded_strategies,
    resolve_stale_review_items,
    upsert_review_queue,
    upsert_strategies,
)

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    total_txns: int = 0
    trade_fills: int = 0
    strategies: int = 0
    closed: int = 0
    ambiguous: int = 0
    coverage: float = 0.0
    categories: int = 0
    superseded_removed: int = 0
    review_resolved: int = 0
    errors: list[str] = field(default_factory=list)


def sync_public_history(conn, client, account_id: str | None = None,
                        as_of: str | None = None) -> SyncResult | None:
    """Returns None when Public is unreachable/unconfigured (first page
    failed); otherwise a SyncResult summary. Never raises."""
    account = account_id or getattr(client, "account_id", "") or ""
    try:
        txns = client.get_all_history()
    except Exception as e:
        log.warning(f"public sync fetch failed: {e}")
        return None
    if txns is None:
        return None

    result = SyncResult(total_txns=len(txns))
    result.trade_fills = sum(1 for t in txns if t.get("type") == "TRADE")

    grouped = group_fills(
        txns, as_of=as_of or datetime.now(timezone.utc).date().isoformat())
    result.strategies = len(grouped.strategies)
    result.closed = sum(1 for g in grouped.strategies if g.status == "closed")
    result.ambiguous = len(grouped.ambiguous)
    result.coverage = grouped.coverage

    categories = {}
    for g in grouped.strategies:
        try:
            categories[g.strategy_id] = category_key_for_group(g)
        except Exception as e:
            categories[g.strategy_id] = "other"
            result.errors.append(f"categorize {g.strategy_id}: {e}")

    try:
        upsert_strategies(conn, grouped.strategies, categories, account)
        upsert_review_queue(conn, grouped.ambiguous, account)
        # Reconcile: this id set came from the COMPLETE history (pagination
        # fails closed), so rows absent from it are id-churn artifacts
        # (scale-ins / leg-in merges changed the opening-txn set) and queue
        # rows for fills that now grouped cleanly are no longer ambiguous.
        result.superseded_removed = delete_superseded_strategies(
            conn, account, [g.strategy_id for g in grouped.strategies])
        result.review_resolved = resolve_stale_review_items(
            conn, account, [a.get("txn_id") for a in grouped.ambiguous])
        stats = compute_category_stats(grouped.strategies, categories)
        result.categories = upsert_category_stats(conn, account, stats)
    except Exception as e:
        log.warning(f"public sync persistence failed: {e}")
        result.errors.append(f"persist: {e}")

    log.info(f"Public sync: {result.strategies} strategies "
             f"({result.closed} closed) from {result.trade_fills} fills, "
             f"coverage {result.coverage:.1%}, {result.ambiguous} in review, "
             f"{result.superseded_removed} superseded removed, "
             f"{result.review_resolved} review rows auto-resolved")
    return result
