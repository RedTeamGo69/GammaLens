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

from range_finder.event_calendars import (
    CPI_DATES, FOMC_DATES, NFP_DATES, PPI_DATES, PCE_DATES,
    EVENT_TIERS, RELEASE_TIMES_ET, events_for_week,
)


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


# ── tier-2 calendars (display-only) ────────────────────────────────────────────

def test_tier2_lists_sorted_unique_no_weekends():
    for name, dates in (("PPI", PPI_DATES), ("PCE", PCE_DATES)):
        assert dates == sorted(dates), f"{name} not chronological"
        assert len(dates) == len(set(dates)), f"{name} has duplicates"
        weekend = [d for d in dates if _weekday(d) >= 5]
        assert not weekend, f"{name} on weekends: {weekend}"


def test_event_tier_and_time_maps_cover_all_names():
    assert set(EVENT_TIERS) == {"fomc", "cpi", "nfp", "opex", "ppi", "pce"}
    assert set(RELEASE_TIMES_ET) == set(EVENT_TIERS)
    # Tier 1 (can gate a weekly trade) is exactly FOMC/CPI/NFP
    assert [n for n, t in sorted(EVENT_TIERS.items()) if t == 1] == ["cpi", "fomc", "nfp"]
    assert RELEASE_TIMES_ET["fomc"] == "14:00"
    assert RELEASE_TIMES_ET["opex"] is None


# ── events_for_week ────────────────────────────────────────────────────────────

def test_events_for_week_fomc_week():
    evs = events_for_week("2026-07-27")
    fomc = [e for e in evs if e.name == "fomc"]
    assert len(fomc) == 1
    assert fomc[0].date == "2026-07-29"
    assert fomc[0].tier == 1
    assert fomc[0].release_time_et == "14:00"
    # PCE prints the Thursday of the same week — tier 2, never gates
    assert any(e.name == "pce" and e.date == "2026-07-30" and e.tier == 2
               for e in evs)


def test_events_for_week_multi_event_opex_week():
    # Week of 2026-07-13: CPI Tue 7/14 (tier 1), PPI Wed 7/15 (tier 2),
    # OpEx Fri 7/17 (tier 2, computed 3rd Friday, all-session)
    evs = events_for_week("2026-07-13")
    names = {e.name for e in evs}
    assert {"cpi", "ppi", "opex"} <= names
    assert [e.date for e in evs] == sorted(e.date for e in evs)
    opex = next(e for e in evs if e.name == "opex")
    assert opex.date == "2026-07-17"
    assert opex.tier == 2
    assert opex.release_time_et is None


def test_events_for_week_normalizes_any_day_to_monday():
    assert events_for_week("2026-07-29") == events_for_week("2026-07-27")


def test_events_for_week_empty_week():
    # Week of 2016-08-22: CPI was 8/16 and OpEx 8/19 (both the prior week),
    # NFP was 8/5, no August-2016 FOMC, and tier-2 lists don't reach 2016.
    assert events_for_week("2016-08-22") == []


# ── third-Friday holiday roll (audit E59) ────────────────────────────────────

def test_third_friday_good_friday_rolls_to_thursday():
    """April 2026: the 3rd Friday is 2026-04-17 (not Good Friday that year),
    but any 3rd Friday that IS a market holiday must roll back to the prior
    session. Use a known Good Friday OpEx: April 2003's 3rd Friday was
    2003-04-18 = Good Friday → rolls to Thursday 2003-04-17."""
    from range_finder.event_calendars import _third_friday
    assert _third_friday(2003, 4) == "2003-04-17"


def test_third_friday_normal_month_unchanged():
    from range_finder.event_calendars import _third_friday
    # 2026-01-16 is the 3rd Friday of Jan 2026, a normal trading day.
    assert _third_friday(2026, 1) == "2026-01-16"
