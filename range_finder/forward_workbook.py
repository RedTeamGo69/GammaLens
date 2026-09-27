"""The weekly forward-test xlsx workbook (Scoreboard · week tab · FTBlank).

Pure: builds bytes from already-collected rows, no Streamlit, no DB. The
Spread Finder tab collects the rows and offers the download.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Weekly forward-test workbook — ALL tickers, one week-named tab (pure: openpyxl only)
# ─────────────────────────────────────────────────────────────────────────────
# One export carries the always-present default instruments plus any tickers
# the user has added to the export list, one row each, on a sheet named after
# the planning week's Monday. The intended workflow:
#
#   • First download = your MASTER workbook. It ships with a Scoreboard and
#     an empty utility sheet (FT_BLANK_SHEET) the Scoreboard needs.
#   • Every following week: download the new export, right-click its week
#     tab → Move or Copy → into the master (any position).
#   • The Scoreboard is TICKER-KEYED, not positional: it matches every week
#     tab's rows by Instrument name (SUMIF/COUNTIFS over INDIRECT built from
#     the pre-listed Monday tab names, existence-sanitized in its hidden
#     column L — see _write_ft_scoreboard). The ticker mix and row order are
#     free to differ week to week; each instrument's score simply accumulates
#     over the weeks it appears. Typing a ticker into a blank Scoreboard slot
#     starts tracking it retroactively across every tab already in the book.
#   • Upgrading an old positional (3D-sum) master: delete its Scoreboard and
#     the WeeksStart/WeeksEnd bookends (obsolete), then Move/Copy a fresh
#     export's Scoreboard AND FTBlank sheets in (Ctrl+click both tabs) —
#     they're self-contained as a pair. Re-type any previously tracked
#     tickers into blank Instrument slots; they re-score retroactively.
#     Caveat: pre-rework week tabs carry the old unguarded Scored?/flag
#     formulas, so a band-less row whose close was filled shows a 0 flag
#     there — the Scoreboard's denominators filter those out (they require
#     the tier's Low band cell to hold a number), but old tabs' Weeks
#     Scored can still count band-less rows. Re-paste a new tab's O..S
#     formulas over an old tab's rows to fully repair it.
#
# Per-week sheet layout:
#   A Week(Mon) · B Instrument · C Class · D Ref/Open · E Weekly Close
#   (user fills after Friday) · F Prev Close · G..N four tier bands
#   (Low/High) · O..R CLOSE INSIDE? per tier (formulas) · S Scored?
#   (formula) · T Notes

# Class-label overrides for the workbook "Class" column; anything else falls
# back to the ticker_config category (Index/ETF/Stock).
FT_CLASS = {"SPX": "Index", "XSP": "Index", "SPY": "ETF",
             "QQQ": "ETF", "NDX": "Index"}
FT_FIRST_DATA_ROW = 6          # first instrument row on each week sheet
FT_MAX_TICKERS = 40            # rows the Scoreboard scans on each week tab (B6:B45)
FT_LAST_DATA_ROW = FT_FIRST_DATA_ROW + FT_MAX_TICKERS - 1
FT_SB_FIRST_ROW = 6            # first instrument slot on the Scoreboard
FT_SB_SLOTS = 40               # slots incl. blanks the user can type a ticker into
FT_SB_LAST_ROW = FT_SB_FIRST_ROW + FT_SB_SLOTS - 1
FT_SB_ALL_ROW = FT_SB_LAST_ROW + 1
FT_SB_WEEKS_BACK = 260         # Monday tab names pre-listed behind the export week
                                # (5y — a fresh Scoreboard must cover every tab an
                                # older master could have accumulated)
FT_SB_WEEKS_FWD = 156          # ...and ahead of it: the 3y runway before the
                                # Scoreboard (which warns as it nears) needs swapping
                                # for a fresh export's
FT_BLANK_SHEET = "FTBlank"     # empty utility sheet not-yet-pasted weeks resolve to
def ft_class(ticker: str) -> str:
    """Asset-class label for the workbook 'Class' column (Index/ETF/Stock)."""
    t = (ticker or "").upper()
    if t in FT_CLASS:
        return FT_CLASS[t]
    try:
        from phase1.ticker_config import get_config
        return str(get_config(t)["category"]).title()
    except Exception:
        return "Stock"


def xlsx_safe_cell(text) -> str:
    """Neutralize a leading spreadsheet formula trigger on any string cell.

    Defense in depth behind _valid_ticker_symbol: Excel/Sheets treat a cell
    beginning with = + - @ (or tab/CR) as a formula. Prefix such a value with
    an apostrophe so it renders literally. Non-strings pass through unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    return "'" + text if text[0] in ("=", "+", "-", "@", "\t", "\r") else text


def monday_tab_names(week_start: str) -> list[str]:
    """Every Monday tab name the Scoreboard pre-scans, as ISO text strings.

    Week tabs are named for their planning Monday (yyyy-mm-dd), so the whole
    horizon is enumerable up front: FT_SB_WEEKS_BACK Mondays behind the
    export week (covers tabs already accumulated in an older master) through
    FT_SB_WEEKS_FWD ahead. A listed tab that doesn't exist yet is remapped
    by the Scoreboard's hidden column-L sanitizer — IF(ISREF(INDIRECT(…)),
    name, FT_BLANK_SHEET) — to the shipped empty blank sheet, so every
    INDIRECT target always exists and missing weeks contribute nothing.
    (Deliberately no IFERROR: wrapping the SUMIF-over-INDIRECT array in
    IFERROR collapses the whole SUMPRODUCT to 0 in real Excel — see
    _write_ft_scoreboard.)
    """
    anchor = datetime.strptime(week_start, "%Y-%m-%d").date()
    anchor -= timedelta(days=anchor.weekday())      # snap to Monday
    first = anchor - timedelta(weeks=FT_SB_WEEKS_BACK)
    return [(first + timedelta(weeks=k)).isoformat()
            for k in range(FT_SB_WEEKS_BACK + FT_SB_WEEKS_FWD + 1)]


def _write_ft_scoreboard(sb, week_start: str, tickers: list[str]) -> None:
    """Write the ticker-keyed Scoreboard onto worksheet ``sb``.

    Row-agnostic by design: every formula matches instruments BY NAME
    (column A) against the Instrument column of every week tab, so a ticker
    can sit on a different row — or be absent — on any given week without
    corrupting the tallies. Aggregation is SUMPRODUCT + SUMIF/COUNTIFS over
    INDIRECT references built from the pre-listed Monday tab names (classic
    multi-sheet conditional-sum pattern; no 365-only functions, works in
    LibreOffice too). Excel refuses to array-lift that pattern once IFERROR
    wraps it (verified empirically — the whole product silently collapses
    to 0), so not-yet-pasted weeks CANNOT be skipped via error handling;
    instead hidden column L sanitizes each name with ISREF, mapping missing
    tabs to the shipped, empty FT_BLANK_SHEET so every INDIRECT target
    always exists. INDIRECT is volatile, so pasting a new week tab re-runs
    the sanitizer and the tab is picked up immediately. Nothing references
    another sheet except through INDIRECT strings, so Move/Copy of this
    sheet (together with FT_BLANK_SHEET) into an older master carries the
    whole scoring system without creating external links.

    Visible layout: A Instrument · B Weeks Scored · C..J (Wins, Hit %) per
    tier, one row per slot, ALL row at the bottom; K1 self-diagnoses (live
    tab count, duplicate-"(2)"-tab and horizon warnings). Hidden: L sanitized
    week-tab names · M raw pre-listed Monday names · N..Q per-tier
    scored-week denominators · R duplicate-tab detector. A week only enters
    a tier's denominator when that tier's flag is numeric AND its Low band
    cell holds a number — the band criterion keeps pre-rework tabs (whose
    old unguarded flags emit 0 on band-less rows) from scoring false losses.
    The sheet ships protected (no password) with only the Instrument slots
    editable, so the plumbing can't be edited or shifted by accident.
    """
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Protection, Side,
    )

    NAVY, MUTED = "1F3864", "595959"
    HDR_FILL = PatternFill("solid", start_color=NAVY)
    TIER_FILLS = [PatternFill("solid", start_color=c)
                  for c in ("FCE4D6", "FFF2CC", "E2EFDA", "DDEBF7")]
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

    names = monday_tab_names(week_start)
    names_last = FT_SB_FIRST_ROW + len(names) - 1

    def _ind(col: str) -> str:
        # e.g. INDIRECT("'"&$L$6:$L$266&"'!$B$6:$B$45") — one range per listed
        # week tab, through the sanitized names in L, so every target exists
        # (missing weeks point at the blank sheet and contribute nothing).
        # Deliberately NO IFERROR: if the blank sheet gets deleted the board
        # shows #REF! loudly instead of silently under-counting.
        return (f'INDIRECT("\'"&$L${FT_SB_FIRST_ROW}:$L${names_last}&"\'!'
                f'${col}${FT_FIRST_DATA_ROW}:${col}${FT_LAST_DATA_ROW}")')

    sb["A1"] = "HAR FORWARD TEST — SCOREBOARD"
    sb["A1"].font = Font(bold=True, size=13, color=NAVY)
    # A2/A3 deliberately NOT merged: unmerged text overflows across the empty
    # cells to its right, while a merged cell clips at its own edge.
    sb["A2"] = (
        "Ticker-keyed: week tabs are matched on their Instrument column, so the "
        "ticker mix AND row order are free to differ week to week. Each week: "
        "download the new export, right-click its week tab → Move or Copy → into "
        "this workbook, any position. One tab per Monday — re-importing a week? "
        "Delete its old tab FIRST (Excel silently renames the paste to (2), "
        "which the board ignores)."
    )
    sb["A2"].font = Font(italic=True, size=9, color=MUTED)
    sb["A3"] = (
        f"Track a new instrument any time: type its ticker into a blank "
        f"Instrument cell — every week tab in the book re-scores instantly. "
        f"Don't rename week tabs or insert rows/columns on them; keep the "
        f"'{FT_BLANK_SHEET}' sheet. Hit % counts only weeks where that tier "
        f"had a band and a filled close. Weeks after {names[-1]} need a fresh "
        f"export's Scoreboard (a warning appears here in time). Upgrading an "
        f"old master: also delete its WeeksStart/WeeksEnd tabs and re-type its "
        f"extra tickers here."
    )
    sb["A3"].font = Font(italic=True, size=9, color=MUTED)

    # Self-diagnostics: live week-tab count, plus loud warnings for the two
    # silent failure modes — a pasted duplicate that Excel renamed to "(2)"
    # (invisible to the name-keyed scan) and the pre-listed name horizon
    # running out.
    _horizon_end = date.fromisoformat(names[-1])
    _warn_from = _horizon_end - timedelta(weeks=12)
    sb["K1"] = (
        f'="Week tabs found: "'
        f'&SUMPRODUCT(--($L${FT_SB_FIRST_ROW}:$L${names_last}<>"{FT_BLANK_SHEET}"))'
        f'&IF(SUM($R${FT_SB_FIRST_ROW}:$R${names_last})>0,'
        f'"   ⚠ a pasted week tab was renamed like {week_start} (2) — the board '
        f'ignores it; delete the old tab, then rename the (2) tab to the plain '
        f'date","")'
        f'&IF(TODAY()>DATE({_warn_from.year},{_warn_from.month},{_warn_from.day}),'
        f'"   ⚠ last pre-listed week is {names[-1]} — swap in a fresh export '
        f'Scoreboard + {FT_BLANK_SHEET}","")'
    )
    sb["K1"].font = Font(italic=True, size=9, color=MUTED)

    _sb_groups = [("C", "D", "LOWER PI"), ("E", "F", "POINT EST"),
                  ("G", "H", "80% PI UPPER"), ("I", "J", "EFFECTIVE")]
    for (lo, hi, label), fill in zip(_sb_groups, TIER_FILLS):
        sb.merge_cells(f"{lo}4:{hi}4")
        c = sb[f"{lo}4"]
        c.value = label
        c.fill = fill
        c.font = Font(bold=True, size=9)
        c.alignment = CENTER
    for col, label in [("A", "Instrument"), ("B", "Weeks Scored")] + [
        (c, lbl) for grp in _sb_groups for c, lbl in ((grp[0], "Wins"), (grp[1], "Hit %"))
    ]:
        cell = sb[f"{col}5"]
        cell.value = label
        cell.fill = HDR_FILL
        cell.font = Font(bold=True, size=9, color="FFFFFF")
        cell.alignment = CENTER
        cell.border = BOX
    for col, label in [("L", "Week tabs (live)"), ("M", "Week tabs (all)"),
                       ("N", "n LPI"), ("O", "n Point"),
                       ("P", "n 80PI"), ("Q", "n Eff"), ("R", "dup?")]:
        cell = sb[f"{col}5"]
        cell.value = label
        cell.font = Font(bold=True, size=8, color=MUTED)

    # M: pre-listed Monday tab names, stored as TEXT (INDIRECT concatenates
    # them into sheet references — Excel must never coerce one to a date
    # serial). L: the sanitized live name every formula actually uses — the
    # week tab itself when it exists, the blank sheet when it doesn't.
    # R: flags a duplicate tab Excel renamed to "<Monday> (2)" (feeds K1).
    for k, nm in enumerate(names):
        r = FT_SB_FIRST_ROW + k
        c = sb[f"M{r}"]
        c.number_format = "@"
        c.value = nm
        sb[f"L{r}"] = (f'=IF(ISREF(INDIRECT("\'"&$M{r}&"\'!$A$1")),'
                       f'$M{r},"{FT_BLANK_SHEET}")')
        sb[f"R{r}"] = f'=IF(ISREF(INDIRECT("\'"&$M{r}&" (2)\'!$A$1")),1,0)'

    # Instrument slots: export tickers first, the rest blank-but-armed — every
    # slot carries the full formula set, guarded on column A being non-blank
    # (IF short-circuits, so empty slots cost nothing to recalc). Slot cells
    # are the only unlocked cells on the sheet (typing a ticker must survive
    # the sheet protection below).
    # (win col, hidden denominator col, week-tab flag col, band-Low col):
    _win_cols = [("C", "N", "O", "G"), ("E", "O", "P", "I"),
                 ("G", "P", "Q", "K"), ("I", "Q", "R", "M")]
    for idx in range(FT_SB_SLOTS):
        r = FT_SB_FIRST_ROW + idx
        tick = f"$A{r}"
        if idx < len(tickers):
            sb[f"A{r}"] = xlsx_safe_cell(tickers[idx])
            sb[f"A{r}"].font = Font(bold=True)
        sb[f"A{r}"].protection = Protection(locked=False)
        sb[f"B{r}"] = (f'=IF({tick}="","",SUMPRODUCT('
                       f'SUMIF({_ind("B")},{tick},{_ind("S")})))')
        for win_col, den_col, flag_col, lo_col in _win_cols:
            sb[f"{win_col}{r}"] = (
                f'=IF({tick}="","",SUMPRODUCT('
                f'COUNTIFS({_ind("B")},{tick},{_ind(flag_col)},1)))'
            )
            # Denominator: flags are numeric (0/1) only when the tier scored;
            # ">=0" skips the ""-blanks bandless/unfilled rows produce. The
            # extra band-Low ">0" criterion drops PRE-REWORK rows whose old
            # unguarded flags emit 0 with no band at all (false losses).
            sb[f"{den_col}{r}"] = (
                f'=IF({tick}="","",SUMPRODUCT('
                f'COUNTIFS({_ind("B")},{tick},{_ind(flag_col)},">=0",'
                f'{_ind(lo_col)},">0")))'
            )
        for hit_col, (win_col, den_col, *_cols) in zip("DFHJ", _win_cols):
            sb[f"{hit_col}{r}"] = (
                f'=IF({tick}="","",IF({den_col}{r}=0,"—",{win_col}{r}/{den_col}{r}))'
            )
            sb[f"{hit_col}{r}"].number_format = "0%"
        for col in "ABCDEFGHIJ":
            sb[f"{col}{r}"].border = BOX
            if col != "A":
                sb[f"{col}{r}"].alignment = Alignment(horizontal="center")

    # ALL row — pooled across every instrument slot.
    ar = FT_SB_ALL_ROW
    lo_r, hi_r = FT_SB_FIRST_ROW, FT_SB_LAST_ROW
    sb[f"A{ar}"] = "ALL"
    sb[f"A{ar}"].font = Font(bold=True)
    sb[f"B{ar}"] = f"=SUM(B{lo_r}:B{hi_r})"
    for win_col, den_col, *_cols in _win_cols:
        sb[f"{win_col}{ar}"] = f"=SUM({win_col}{lo_r}:{win_col}{hi_r})"
        sb[f"{den_col}{ar}"] = f"=SUM({den_col}{lo_r}:{den_col}{hi_r})"
    for hit_col, (win_col, den_col, *_cols) in zip("DFHJ", _win_cols):
        sb[f"{hit_col}{ar}"] = f'=IF({den_col}{ar}=0,"—",{win_col}{ar}/{den_col}{ar})'
        sb[f"{hit_col}{ar}"].number_format = "0%"
    for col in "ABCDEFGHIJ":
        sb[f"{col}{ar}"].border = BOX
        if col != "A":
            sb[f"{col}{ar}"].alignment = Alignment(horizontal="center")

    sb.column_dimensions["A"].width = 12
    sb.column_dimensions["B"].width = 13
    for col in "CDEFGHIJ":
        sb.column_dimensions[col].width = 9
    for col in "LMNOPQR":
        sb.column_dimensions[col].hidden = True
    sb.freeze_panes = "A6"

    # Guardrail: protect the sheet (no password) so the hidden plumbing and
    # formulas can't be edited or shifted by accident — only the Instrument
    # slots (unlocked above) accept typing. Formatting stays allowed;
    # right-click the tab → Unprotect Sheet to override deliberately.
    sb.protection.sheet = True
    sb.protection.formatCells = False
    sb.protection.formatColumns = False
    sb.protection.formatRows = False


def _write_ft_week_sheet(wb, week_start: str, rows: list[dict]):
    """Write one week tab (named for its Monday) into ``wb``; returns it.

    Self-contained on purpose: every formula references THIS sheet only, so
    Move/Copy into the master never externalizes a reference. The Scoreboard
    matches its rows by the Instrument column, so ``rows`` may carry any
    ticker mix, in any order (capped at FT_MAX_TICKERS — the Scoreboard
    scans B6:B45). Every row of that window is written: exported tickers on
    top, the rest blank-but-armed (formulas + styling in place) so a row the
    user types in by hand — the only way to log the current week for a
    freshly tracked ticker — scores exactly like an exported one. Ships
    protected (no password): data cells editable, formula cells and the
    sheet structure locked (the Scoreboard reads fixed positions).
    """
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import (
        Alignment, Border, Font, PatternFill, Protection, Side,
    )
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    NAVY, MUTED = "1F3864", "595959"
    HDR_FILL = PatternFill("solid", start_color=NAVY)
    TIER_FILLS = [PatternFill("solid", start_color=c)
                  for c in ("FCE4D6", "FFF2CC", "E2EFDA", "DDEBF7")]
    GREY_FILL = PatternFill("solid", start_color="D9D9D9")
    GREEN_FILL = PatternFill("solid", start_color="C6EFCE")
    RED_FILL = PatternFill("solid", start_color="FFC7CE")
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)

    rows = list(rows)[:FT_MAX_TICKERS]
    FIRST = FT_FIRST_DATA_ROW
    ws = wb.create_sheet(week_start)

    ws["A1"] = f"HAR MODELS — WEEKLY RANGE FORWARD TEST — week of {week_start}"
    ws["A1"].font = Font(bold=True, size=13, color=NAVY)
    ws["A2"] = (
        "One row per instrument. Each model tier gives its own Low/High band. "
        "Win = weekly CLOSE inside that tier's band (intraday wicks ignored — set-and-forget). "
        "Fill 'Weekly Close' (column E) after Friday's close; CLOSE INSIDE, Scored? and the "
        "Scoreboard update automatically. The Scoreboard matches rows by Instrument name, "
        "so row order doesn't matter — just don't rename this tab or insert rows/columns "
        "(sheet is protected, no password, to keep the scored layout fixed; all data "
        "cells stay editable — add extra instrument rows below the exported ones freely)."
    )
    ws["A2"].font = Font(italic=True, size=9, color=MUTED)
    ws.merge_cells("A2:T2")

    _tier_groups = [
        ("G", "H", "LOWER PI (tightest — richest credit)"),
        ("I", "J", "POINT ESTIMATE (HAR base)"),
        ("K", "L", "80% PI UPPER (80% interval)"),
        ("M", "N", "EFFECTIVE (PI + buffer — your condor)"),
    ]
    for (lo, hi, label), fill in zip(_tier_groups, TIER_FILLS):
        ws.merge_cells(f"{lo}4:{hi}4")
        c = ws[f"{lo}4"]
        c.value = label
        c.fill = fill
        c.font = Font(bold=True, size=8)
        c.alignment = CENTER
    ws.merge_cells("O4:R4")
    ws["O4"] = "CLOSE INSIDE? (per model)"
    ws["O4"].fill = PatternFill("solid", start_color="B4C6E7")
    ws["O4"].font = Font(bold=True, size=8)
    ws["O4"].alignment = CENTER

    _headers = [
        ("A", "Week (Mon)"), ("B", "Instrument"), ("C", "Class"),
        ("D", "Ref/Open ($)"), ("E", "Weekly Close ($)"), ("F", "Prev Close ($)"),
        ("G", "Low"), ("H", "High"), ("I", "Low"), ("J", "High"),
        ("K", "Low"), ("L", "High"), ("M", "Low"), ("N", "High"),
        ("O", "Lower PI"), ("P", "Point"), ("Q", "80% PI"), ("R", "Eff"),
        ("S", "Scored?"), ("T", "Notes"),
    ]
    for col, label in _headers:
        cell = ws[f"{col}5"]
        cell.value = label
        cell.fill = HDR_FILL
        cell.font = Font(bold=True, size=9, color="FFFFFF")
        cell.alignment = CENTER
        cell.border = BOX

    _slot_cols = [("lower_pi", "G", "H"), ("point", "I", "J"),
                  ("pi_upper", "K", "L"), ("effective", "M", "N")]
    _week_date = None
    try:
        _week_date = datetime.strptime(week_start, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        pass

    # Every scannable row (6..45) is written: exported tickers on top, the
    # rest blank-but-armed so a row the user types in by hand scores exactly
    # like an exported one (the Scoreboard reads the whole window either way).
    for r in range(FIRST, FT_LAST_DATA_ROW + 1):
        row = rows[r - FIRST] if r - FIRST < len(rows) else None
        if row is not None:
            t = row.get("ticker", "?")
            ws[f"A{r}"] = _week_date or week_start
            ws[f"A{r}"].number_format = "yyyy-mm-dd"
            ws[f"B{r}"] = xlsx_safe_cell(t)
            ws[f"B{r}"].font = Font(bold=True)
            ws[f"C{r}"] = ft_class(t)
            if row.get("ref") is not None:
                ws[f"D{r}"] = row["ref"]
            if row.get("prev_close") is not None:
                ws[f"F{r}"] = row["prev_close"]
            bands = row.get("bands") or {}
            for slot, lo_col, hi_col in _slot_cols:
                band = bands.get(slot)
                if band is not None:
                    ws[f"{lo_col}{r}"] = band[0]
                    ws[f"{hi_col}{r}"] = band[1]
            note = " · ".join(row.get("notes") or [])
            if row.get("error"):
                note = (f"⚠ {row['error']}" + (f" · {note}" if note else ""))
            ws[f"T{r}"] = note
            ws[f"T{r}"].font = Font(size=8, color=MUTED)
            ws[f"T{r}"].alignment = Alignment(wrap_text=True, vertical="center")

        # E — Weekly Close: left blank for the user, shaded as fill-me.
        ws[f"E{r}"].fill = GREY_FILL
        for (slot, lo_col, hi_col), fill in zip(_slot_cols, TIER_FILLS):
            ws[f"{lo_col}{r}"].fill = fill
            ws[f"{hi_col}{r}"].fill = fill

        # CLOSE INSIDE? flags — blank until the row has an instrument AND the
        # close AND that tier's band are all NUMBERS (COUNT counts numbers
        # only): a pasted close with a stray space or an 'n/a' band stays
        # blank instead of scoring a false loss (any text compares above any
        # number in Excel, so an unguarded compare emits 0). 1/0 once
        # scoreable — the Scoreboard's denominators count exactly these
        # numeric cells.
        for (slot, lo_col, hi_col), flag_col in zip(_slot_cols, "OPQR"):
            ws[f"{flag_col}{r}"] = (
                f'=IF(OR($B{r}="",COUNT($E{r},{lo_col}{r},{hi_col}{r})<3),"",'
                f'IF(AND($E{r}>={lo_col}{r},$E{r}<={hi_col}{r}),1,0))'
            )
            ws[f"{flag_col}{r}"].alignment = Alignment(horizontal="center")
        # Scored? = instrument present, close is a number, ≥1 numeric band.
        ws[f"S{r}"] = (f'=IF(OR($B{r}="",COUNT($E{r})=0,'
                       f'COUNT($G{r}:$N{r})=0),"",1)')
        ws[f"S{r}"].alignment = Alignment(horizontal="center")

        for col_idx in range(1, 21):
            cell = ws.cell(row=r, column=col_idx)
            cell.border = BOX
            if 4 <= col_idx <= 14 and not isinstance(cell.value, str):
                cell.number_format = "#,##0.00"

    # Green/red the CLOSE INSIDE cells once scored — the whole armed window,
    # so hand-added rows colour like exported ones.
    ws.conditional_formatting.add(
        f"O{FIRST}:R{FT_LAST_DATA_ROW}",
        CellIsRule(operator="equal", formula=["1"], fill=GREEN_FILL),
    )
    ws.conditional_formatting.add(
        f"O{FIRST}:R{FT_LAST_DATA_ROW}",
        CellIsRule(operator="equal", formula=["0"], fill=RED_FILL),
    )

    _widths = {"A": 11, "B": 11, "C": 7, "D": 10, "E": 11, "F": 10,
               "S": 8, "T": 46}
    for col_idx in range(7, 15):
        _widths[get_column_letter(col_idx)] = 9
    for col_idx in range(15, 19):
        _widths[get_column_letter(col_idx)] = 8
    for col, w in _widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A6"

    # Weekly Close must be a number — typed text silently un-scores the row
    # (the COUNT guards keep flags blank rather than emit a false loss), so
    # also reject it at entry. Paste bypasses validation; the formula guards
    # stay authoritative.
    dv = DataValidation(
        type="decimal", operator="greaterThan", formula1="0", allowBlank=True,
        showErrorMessage=True, errorTitle="Weekly Close must be a number",
        error=("Type the Friday close as a plain number — text (even a stray "
               "space) keeps the row from scoring."),
    )
    ws.add_data_validation(dv)
    dv.add(f"E{FIRST}:E{FT_LAST_DATA_ROW}")

    # Guardrail: protect the sheet (no password). Data-entry cells stay
    # unlocked (B..N instrument/class/ref/close/bands + T notes — enough to
    # hand-add a full instrument row); the flag/Scored? formulas and the
    # sheet STRUCTURE stay locked, because the Scoreboard reads fixed
    # positions (B6:B45, flags O..R, Scored? S) and a single inserted
    # row/column would silently garble every tally. Formatting stays
    # allowed; right-click the tab → Unprotect Sheet to override.
    for r in range(FIRST, FT_LAST_DATA_ROW + 1):
        for col in "BCDEFGHIJKLMNT":
            ws[f"{col}{r}"].protection = Protection(locked=False)
    ws.protection.sheet = True
    ws.protection.formatCells = False
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    return ws


def _write_ft_blank_sheet(wb) -> None:
    """Create the empty utility sheet unpasted weeks resolve to.

    The Scoreboard's sanitizer (hidden column L) points every pre-listed
    Monday whose tab isn't in the workbook yet at this sheet, so its scanned
    region (rows 6-45) must stay empty — a note in row 1 is safely outside.
    """
    from openpyxl.styles import Font

    bk = wb.create_sheet(FT_BLANK_SHEET)
    bk["A1"] = (
        f"Keep this sheet (and keep it empty) — the Scoreboard resolves "
        f"not-yet-added weeks here. If you copy the Scoreboard into another "
        f"workbook, copy '{FT_BLANK_SHEET}' along with it. The sheet is "
        f"protected on purpose: anything typed into it would be counted once "
        f"per missing week."
    )
    bk["A1"].font = Font(italic=True, size=9, color="595959")
    bk.sheet_view.showGridLines = False
    # Fully locked, no password: a stray row typed here would be multiplied
    # by every not-yet-pasted Monday (~150-200x) with no error anywhere.
    bk.protection.sheet = True


def build_forward_test_workbook(*, week_start: str, model_choice: str, rows: list[dict]) -> bytes:
    """Assemble the multi-ticker forward-test workbook.

    Sheets, in order: Scoreboard · <week_start> · FTBlank. `rows` is the
    ordered list of per-ticker dicts to write (defaults first, then
    user-added extras); each carries its own ``ticker`` (see
    _collect_week_bands_for_ticker for the shape). The Scoreboard matches
    week tabs by ticker name, so the mix and order may differ freely across
    weeks (capped at FT_MAX_TICKERS per week — the fixed range its formulas
    scan). ``model_choice`` only flavors the notes the caller baked into
    ``rows``.
    """
    from io import BytesIO

    import openpyxl

    rows = list(rows)[:FT_MAX_TICKERS]
    wb = openpyxl.Workbook()
    sb = wb.active
    sb.title = "Scoreboard"
    # Slot labels feed SUMIF/COUNTIFS criteria directly — drop blanks rather
    # than emit a placeholder ("?" is an Excel wildcard matching any 1-char
    # instrument).
    _slot_tickers = [t for t in (str(r.get("ticker") or "").strip()
                                 for r in rows) if t]
    _write_ft_scoreboard(sb, week_start, _slot_tickers)
    _write_ft_week_sheet(wb, week_start, rows)
    _write_ft_blank_sheet(wb)

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
