"""Delete app-table rows for tickers that are not on the forward-test roster.

Dry run by default: prints what would go. With --apply it first writes every
affected row to CSV under --backup-dir, then deletes in one transaction and
re-counts before committing (any mismatch rolls back).

Never touches ``ft_*`` tables: the forward-test history (including retired
NDX slots that the v5 scoring-only study still settles) stays intact.

Run only after Streamlit Cloud has been rebooted onto the new roster: a stale
container still lists XSP/NDX/CAT as defaults and could write them back.

  python scripts/purge_non_roster_tickers.py
  python scripts/purge_non_roster_tickers.py --apply --backup-dir D:/gamma_lens_backups
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1.pg import connect
from phase1.config import EXTRA_CAPTURE_INDICES
from range_finder.forward_test.config import UNIVERSE

KEEP = (*UNIVERSE, *EXTRA_CAPTURE_INDICES)


def inventory(cur) -> dict[str, dict[str, int]]:
    """{table: {ticker: rows}} for off-roster tickers in every base table with a ticker column."""
    cur.execute("""
        SELECT c.table_name FROM information_schema.columns c
        JOIN information_schema.tables t ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE c.table_schema = 'public' AND c.column_name = 'ticker'
          AND t.table_type = 'BASE TABLE' AND c.table_name NOT LIKE 'ft\\_%'
        ORDER BY c.table_name""")
    found = {}
    for (table,) in cur.fetchall():
        cur.execute(f'SELECT ticker, count(*) FROM "{table}" WHERE NOT (ticker = ANY(%s)) GROUP BY ticker',
                    (list(KEEP),))
        rows = dict(cur.fetchall())
        if rows:
            found[table] = rows
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="back up, then delete")
    p.add_argument("--backup-dir", type=Path, help="required with --apply")
    args = p.parse_args()
    if args.apply and not args.backup_dir:
        sys.exit("--apply needs --backup-dir")

    conn = connect(autocommit=False)
    try:
        cur = conn.cursor()
        plan = inventory(cur)
        total = sum(sum(t.values()) for t in plan.values())
        for table, tickers in plan.items():
            print(f"{table:24s} " + ", ".join(f"{k}={n}" for k, n in sorted(tickers.items())))
        print(f"{total} rows across {len(plan)} tables (roster kept: {len(KEEP)} tickers)")
        if not args.apply or not total:
            conn.rollback()
            return

        out = args.backup_dir / datetime.now().strftime("purge-%Y%m%d-%H%M%S")
        out.mkdir(parents=True)
        for table in plan:
            with open(out / f"{table}.csv", "w", encoding="utf-8", newline="") as fh:
                cur.copy_expert(
                    cur.mogrify(f'COPY (SELECT * FROM "{table}" WHERE NOT (ticker = ANY(%s))) '
                                "TO STDOUT WITH CSV HEADER", (list(KEEP),)).decode(), fh)
        print(f"backed up to {out}")

        kept_before = {}
        for table in plan:
            cur.execute(f'SELECT count(*) FROM "{table}" WHERE ticker = ANY(%s)', (list(KEEP),))
            kept_before[table] = cur.fetchone()[0]
            cur.execute(f'DELETE FROM "{table}" WHERE NOT (ticker = ANY(%s))', (list(KEEP),))
            expected = sum(plan[table].values())
            if cur.rowcount != expected:
                raise RuntimeError(f"{table}: deleted {cur.rowcount}, expected {expected}")
        # Self-check inside the transaction: nothing off-roster left, roster rows untouched.
        assert not inventory(cur), "off-roster rows remain"
        for table, n in kept_before.items():
            cur.execute(f'SELECT count(*) FROM "{table}" WHERE ticker = ANY(%s)', (list(KEEP),))
            assert cur.fetchone()[0] == n, f"{table}: roster row count changed"
        conn.commit()
        print(f"deleted {total} rows")
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
