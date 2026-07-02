"""Event calendar integrity (range_finder.event_calendars).

The 2016-2019 backfill (FRED release calendars, verified 2026-07) unlocked
the 10y history experiment — these tests pin the properties a corrupted
calendar would break silently:

  * counts per year (8 FOMC; 12 CPI; 12 NFP — 2020 FOMC has 9 incl. the
    March emergency meetings),
  * weekday sanity: no releases on weekends; FOMC decisions never Monday
    or Friday; NFP is (almost) always Friday,
  * chronological order and no duplicates.
"""
from collections import Counter
from datetime import date

from range_finder.event_calendars import CPI_DATES, FOMC_DATES, NFP_DATES


def _by_year(dates):
    return Counter(d[:4] for d in dates)


def _weekday(d: str) -> int:
    return date.fromisoformat(d).weekday()   # Mon=0 .. Sun=6


# ── counts per year ────────────────────────────────────────────────────────────

def test_fomc_counts_per_year():
    counts = _by_year(FOMC_DATES)
    for year in ("2016", "2017", "2018", "2019", "2021", "2022",
                 "2023", "2024", "2025", "2026"):
        assert counts[year] == 8, f"{year}: {counts[year]} FOMC dates"
    assert counts["2020"] == 9   # 8 scheduled + March 2020 emergency actions


def test_cpi_counts_per_year():
    counts = _by_year(CPI_DATES)
    for year in ("2016", "2017", "2018", "2019", "2020", "2021",
                 "2022", "2023", "2024", "2025", "2026"):
        assert counts[year] == 12, f"{year}: {counts[year]} CPI dates"


def test_nfp_counts_per_year():
    counts = _by_year(NFP_DATES)
    for year in ("2016", "2017", "2018", "2019", "2020", "2021",
                 "2022", "2023", "2024", "2025", "2026"):
        assert counts[year] == 12, f"{year}: {counts[year]} NFP dates"


# ── weekday sanity ─────────────────────────────────────────────────────────────

def test_no_weekend_releases():
    # One legitimate weekend entry exists: the 2020-03-15 emergency FOMC
    # cut was announced on a Sunday.
    known_weekend = {"2020-03-15"}
    for name, dates in (("FOMC", FOMC_DATES), ("CPI", CPI_DATES),
                        ("NFP", NFP_DATES)):
        weekend = [d for d in dates
                   if _weekday(d) >= 5 and d not in known_weekend]
        assert not weekend, f"{name} on weekends: {weekend}"


def test_fomc_never_monday_or_friday():
    # Scheduled two-day meetings end Tue-Thu. The one exception is the
    # 2020-03-15 emergency cut — a Sunday announcement recorded on the
    # Sunday itself would trip the weekend test, so it's stored as dated;
    # allow only that specific exception here.
    bad = [d for d in FOMC_DATES
           if _weekday(d) in (0, 4) and d != "2020-03-15"]
    assert not bad, f"FOMC on Mon/Fri: {bad}"


def test_nfp_is_friday_with_known_exceptions():
    # NFP is first-Friday except documented shifts: Thursday prints ahead
    # of July-4th (2020-07-02, 2025-07-03, 2026-07-02) and the Wednesday
    # 2026-02-11 print after the late-2025 data disruption.
    known_non_friday = {"2020-07-02", "2025-07-03", "2026-02-11", "2026-07-02"}
    bad = [d for d in NFP_DATES
           if _weekday(d) != 4 and d not in known_non_friday]
    assert not bad, f"non-Friday NFP: {bad}"


# ── ordering / duplicates ──────────────────────────────────────────────────────

def test_sorted_and_unique():
    for name, dates in (("FOMC", FOMC_DATES), ("CPI", CPI_DATES),
                        ("NFP", NFP_DATES)):
        assert dates == sorted(dates), f"{name} not chronological"
        assert len(dates) == len(set(dates)), f"{name} has duplicates"


def test_backfill_reaches_2016():
    assert min(FOMC_DATES) == "2016-01-27"
    assert min(CPI_DATES) == "2016-01-20"
    assert min(NFP_DATES) == "2016-01-08"
