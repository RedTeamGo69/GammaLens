"""TradingView export — serialize GEX key levels into a compact GL1 string.

TradingView's Pine Script cannot fetch external data at runtime, so Gamma
Lens ships a static companion indicator (``PINE_INDICATOR_SOURCE``, pasted
into the Pine Editor once) plus this serializer, which packs the current
levels into a one-line string the user pastes into the indicator's settings
whenever they want fresh levels on the chart.

Pure module: no Streamlit imports. The UI glue lives in ui_tv_export.py.

Format (GL1)::

    GL1|<TICKER>|<YYYY-MM-DD>|spot=F|zg=F|zgw=1|cw=F|pw=F
       |emd=LO:HI|emw=LO:HI|emm=LO:HI|pi=LO:HI

- The 3-part header is positional (version, sanitized ticker, data date);
  every other token is ``key=value`` and is omitted when the level is
  unavailable — the Pine parser leaves missing levels ``na`` and simply
  doesn't draw them.
- ``zgw=1`` marks a fallback/estimated zero gamma (drawn dashed in Pine).
- ``parse_tv_string()`` is a line-for-line Python mirror of the Pine
  parsing algorithm; the round-trip test in tests/test_tv_export.py is what
  pins the serializer and the indicator to the same format. Any format
  change must bump TV_FORMAT_VERSION *and* the version literal inside
  PINE_INDICATOR_SOURCE (a template test cross-checks that they agree).

The Pine template's builtin calls were verified signature-by-signature
against the official Pine v6 reference manual (see the pine-docs MCP server
in Projects/Pinescraper/pine-docs-mcp). Style rule learned the hard way:
every Pine statement stays on ONE line — wrapped continuation lines are
indentation-sensitive and produce cryptic "missing closing parenthesis"
compile errors when the parser rejects the wrap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

TV_FORMAT_VERSION = "GL1"

# Order in which value tokens are emitted. The parser is order-insensitive
# and ignores unknown keys, so this only pins the canonical output shape.
_SCALAR_KEYS = ("spot", "zg", "cw", "pw")
_BAND_KEYS = ("emd", "emw", "emm", "pi")

_TICKER_KEEP_RE = re.compile(r"[^A-Z0-9.]")
_MAX_TICKER_LEN = 10


@dataclass(frozen=True)
class TVLevels:
    """Everything the GL1 string can carry. Bands are (lower, upper) prices."""

    ticker: str
    date: str
    spot: float | None = None
    zero_gamma: float | None = None
    zero_gamma_fallback: bool = False
    call_wall: float | None = None
    put_wall: float | None = None
    em_daily: tuple[float, float] | None = None
    em_weekly: tuple[float, float] | None = None
    em_monthly: tuple[float, float] | None = None
    har_pi: tuple[float, float] | None = None


def sanitize_tv_ticker(ticker: str) -> str:
    """Uppercase A-Z/0-9/dot only, max 10 chars, ``NA`` when nothing survives.

    The charset structurally cannot contain the GL1 separators (| = :), so a
    hostile or malformed ticker can never corrupt the string format. Same
    philosophy as ui_spread_finder._TICKER_RE for the xlsx export.
    """
    cleaned = _TICKER_KEEP_RE.sub("", str(ticker or "").upper())[:_MAX_TICKER_LEN]
    return cleaned or "NA"


def _is_valid_price(value) -> bool:
    """True for a finite positive number (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def _normalize_band(band) -> tuple[float, float] | None:
    """Validated (lower, upper) with lower < upper, else None.

    Accepts any 2-sequence; swaps reversed sides; rejects degenerate
    (equal) or partially-missing bands.
    """
    try:
        lo, hi = band  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    if not (_is_valid_price(lo) and _is_valid_price(hi)):
        return None
    lo_f, hi_f = float(lo), float(hi)
    if lo_f == hi_f:
        return None
    return (lo_f, hi_f) if lo_f < hi_f else (hi_f, lo_f)


def _em_band(em) -> tuple[float, float] | None:
    """(lower_level, upper_level) from an expected-move dict, else None."""
    if not isinstance(em, dict):
        return None
    return _normalize_band((em.get("lower_level"), em.get("upper_level")))


def build_tv_levels(*, ticker: str, date: str, spot,
                    levels: dict | None,
                    daily_em: dict | None = None,
                    weekly_em: dict | None = None,
                    monthly_em: dict | None = None,
                    har_pi: tuple | None = None) -> TVLevels:
    """Adapt the app's live dicts into a TVLevels snapshot.

    Total function: tolerates None/missing keys everywhere and never raises.
    ``levels`` is the find_key_levels() dict; the EM dicts are
    compute_expected_move()-shaped (upper_level/lower_level); ``har_pi`` is
    (pi_lower_px, pi_upper_px) or None.

    Zero-gamma fallback: the empty-chain path of find_key_levels returns no
    ``zero_gamma_is_true_crossing`` key at all, and the sweep sets it False
    for rescued/fallback nodes — either way the level is an estimate, not a
    true crossing, so it gets flagged and Pine draws it dashed.
    """
    lv = levels if isinstance(levels, dict) else {}

    def _price(key: str) -> float | None:
        value = lv.get(key)
        return float(value) if _is_valid_price(value) else None

    return TVLevels(
        ticker=sanitize_tv_ticker(ticker),
        date=str(date),
        spot=float(spot) if _is_valid_price(spot) else None,
        zero_gamma=_price("zero_gamma"),
        zero_gamma_fallback=not bool(lv.get("zero_gamma_is_true_crossing")),
        call_wall=_price("call_wall"),
        put_wall=_price("put_wall"),
        em_daily=_em_band(daily_em),
        em_weekly=_em_band(weekly_em),
        em_monthly=_em_band(monthly_em),
        har_pi=_normalize_band(har_pi) if har_pi is not None else None,
    )


def build_tv_string(tv: TVLevels) -> str:
    """Serialize a TVLevels snapshot into the canonical GL1 string."""
    parts = [TV_FORMAT_VERSION, sanitize_tv_ticker(tv.ticker), str(tv.date)]

    scalars = {"spot": tv.spot, "zg": tv.zero_gamma,
               "cw": tv.call_wall, "pw": tv.put_wall}
    bands = {"emd": tv.em_daily, "emw": tv.em_weekly,
             "emm": tv.em_monthly, "pi": tv.har_pi}

    for key in _SCALAR_KEYS:
        value = scalars[key]
        if _is_valid_price(value):
            parts.append(f"{key}={value:.2f}")
        # zgw rides directly behind zg so the string stays human-scannable
        if key == "zg" and tv.zero_gamma_fallback and _is_valid_price(value):
            parts.append("zgw=1")

    for key in _BAND_KEYS:
        band = _normalize_band(bands[key])
        if band is not None:
            parts.append(f"{key}={band[0]:.2f}:{band[1]:.2f}")

    return "|".join(parts)


def missing_tokens(tv: TVLevels) -> list[str]:
    """Human-readable names of levels the string will NOT carry.

    Uses the same validity rules as build_tv_string so the UI caption and
    the emitted string can never disagree.
    """
    out = []
    if not _is_valid_price(tv.zero_gamma):
        out.append("zero gamma")
    if not _is_valid_price(tv.call_wall):
        out.append("call wall")
    if not _is_valid_price(tv.put_wall):
        out.append("put wall")
    for band, name in ((tv.em_daily, "daily EM"), (tv.em_weekly, "weekly EM"),
                       (tv.em_monthly, "monthly EM"), (tv.har_pi, "HAR PI")):
        if _normalize_band(band) is None:
            out.append(name)
    return out


def _to_number(text: str) -> float | None:
    """Python mirror of Pine's str.tonumber(): float or None (Pine: na)."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_tv_string(s: str) -> dict | None:
    """Reference parser — a line-for-line mirror of the Pine algorithm.

    Exists so the round-trip test pins the Python serializer and the Pine
    indicator to the same format; keep in lockstep with the parse block in
    PINE_INDICATOR_SOURCE. Returns None when the header is unusable (Pine:
    gParsed stays false), otherwise a dict with every field present and
    unparseable values left None (Pine: na).
    """
    if not isinstance(s, str):
        return None
    toks = s.split("|")
    if len(toks) < 3 or toks[0] != TV_FORMAT_VERSION:
        return None

    out: dict = {"version": toks[0], "ticker": toks[1].upper(), "date": toks[2],
                 "spot": None, "zg": None, "zgw": False, "cw": None, "pw": None,
                 "emd": None, "emw": None, "emm": None, "pi": None}
    for tok in toks[3:]:
        kv = tok.split("=")
        if len(kv) != 2:
            continue
        key, value = kv
        if key == "spot":
            out["spot"] = _to_number(value)
        elif key == "zg":
            out["zg"] = _to_number(value)
        elif key == "zgw":
            out["zgw"] = value == "1"
        elif key == "cw":
            out["cw"] = _to_number(value)
        elif key == "pw":
            out["pw"] = _to_number(value)
        elif key in _BAND_KEYS:
            band = value.split(":")
            if len(band) == 2:
                lo, hi = _to_number(band[0]), _to_number(band[1])
                if lo is not None and hi is not None:
                    out[key] = (lo, hi)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Companion Pine Script v6 indicator (single source of truth — the UI's
# download button and code block both serve this constant verbatim).
#
# Authoring rules (enforced by tests):
#   - ONE statement per line, never wrapped: Pine's continuation-line
#     indentation rules are fragile and a rejected wrap surfaces as
#     "Missing closing parenthesis". Parentheses balance on every line.
#   - Parse + draw + status all happen on barstate.islast (no isfirst).
#   - Every builtin call verified against the official v6 reference.
# ─────────────────────────────────────────────────────────────────────────────

PINE_INDICATOR_SOURCE = '''//@version=6
//
// Gamma Lens Levels (GL1) — companion indicator for the Gamma Lens dashboard.
//
// SETUP (once)
//   TradingView → Pine Editor → paste this whole script → Add to chart.
//   An orange prompt appears in the chart's top-right corner until a level
//   string is pasted; after pasting it becomes a small gray status line.
//
// UPDATE (whenever you want fresh levels)
//   Gamma Lens → Strike GEX tab → TradingView export → copy the level
//   string → indicator Settings → Inputs → paste. That's it — the script
//   itself never changes.
//
// WHY A PASTE STRING
//   Pine indicators cannot fetch external data at runtime, and the built-in
//   hline primitive only accepts compile-time constants — so the levels
//   ride in through one input string and are drawn with line.new.
//
// STRING FORMAT (GL1)
//   GL1|SPX|2026-07-25|spot=6350.25|zg=6320.50|cw=6400.00|pw=6250.00|emd=6310.50:6390.75|emw=6280.00:6420.00|emm=6150.00:6550.00|pi=6270.20:6440.80
//   Tokens after the 3-part header are optional key=value pairs — a missing
//   token means "level unavailable" and nothing is drawn for it. zgw=1
//   flags a fallback/estimated zero gamma (drawn dashed).
//
// STYLE NOTE: every statement is deliberately on ONE line, however long —
// Pine's line-wrapping rules are indentation-sensitive and a mis-indented
// continuation produces cryptic "missing closing parenthesis" errors.
indicator("Gamma Lens Levels", shorttitle="GL Levels", overlay=true, max_lines_count=32, max_labels_count=32)

// The compiler's output check ("indicator must contain at least one plot or drawing")
// only scans the global scope and ignores drawings created inside user-defined
// functions like _lvl/_band below — this invisible, non-editable plot is
// required for the script to compile at all.
float _anchor = na
plot(_anchor, "GL anchor", display=display.none, editable=false)

// ── Inputs ──────────────────────────────────────────────────────────────
string src = input.string("", "Gamma Lens level string", tooltip="Copy from Gamma Lens → Strike GEX → TradingView export, then paste here.")
bool showZG = input.bool(true, "Zero gamma")
bool showWalls = input.bool(true, "Call / put walls")
bool showEMD = input.bool(true, "Daily EM")
bool showEMW = input.bool(true, "Weekly EM")
bool showEMM = input.bool(true, "Monthly EM")
bool showPI = input.bool(true, "HAR model PI")
bool showLabels = input.bool(true, "Level labels")
bool showSpot = input.bool(false, "Snapshot spot")
int lineLen = input.int(40, "Line length (bars)", minval=5, maxval=500, tooltip="How many bars back each level line reaches. Lines always stop at the last bar, next to their labels — they never extend right.")
int labSize = input.int(10, "Label text size", minval=7, maxval=24, tooltip="Point size: 7 ≈ tiny, 10 ≈ small, 12 ≈ normal, 18 ≈ large.")

// ── Colors (mirror the Gamma Lens palette) ──────────────────────────────
color COL_ZG = #25d8ef
color COL_CW = #2be88a
color COL_PW = #ff4d68
color COL_EMD = #6ea8ff
color COL_EMW = #a98bff
color COL_EMM = #f5c542
color COL_PI = #ffb454
color COL_SPOT = #9aa7b8

// ── Drawing containers + helpers ────────────────────────────────────────
var array<line> lns = array.new<line>()
var array<label> lbls = array.new<label>()

_wipe() =>
    while array.size(lns) > 0
        line.delete(array.pop(lns))
    while array.size(lbls) > 0
        label.delete(array.pop(lbls))

_tag(float y, string txt, color col) =>
    if showLabels and not na(y)
        array.push(lbls, label.new(bar_index, y, txt, style=label.style_label_left, color=color.new(color.black, 100), textcolor=col, size=labSize))

// Lines run lineLen bars back and STOP at the last bar (no right extension),
// so each one ends exactly where its label sits.
_lvl(float y, color col, string txt, string sty) =>
    if not na(y)
        int x1 = math.max(bar_index - lineLen, 0)
        array.push(lns, line.new(x1, y, bar_index, y, extend=extend.none, color=col, style=sty, width=1))
        _tag(y, txt + "  " + str.tostring(y, format.mintick), col)

// A range is a High/Low pair of plain lines — no fill, so wide ranges
// (monthly EM, HAR PI) don't paint over the whole chart.
_band(float lo, float hi, color col, string txt) =>
    if not na(lo) and not na(hi)
        _lvl(hi, col, txt + " High", line.style_solid)
        _lvl(lo, col, txt + " Low", line.style_solid)

var table wtab = table.new(position.top_right, 1, 3)

// Everything happens on the last bar: parse → wipe → draw → status. The
// string is re-read on every pass, so a fresh paste always takes effect;
// drawings are deleted and redrawn to keep object counts flat.
if barstate.islast
    bool gParsed = false
    string gTkr = ""
    string gDate = ""
    float gSpot = na
    float gZg = na
    bool gZgFallback = false
    float gCw = na
    float gPw = na
    float gEmdLo = na
    float gEmdHi = na
    float gEmwLo = na
    float gEmwHi = na
    float gEmmLo = na
    float gEmmHi = na
    float gPiLo = na
    float gPiHi = na

    // Strip spaces a copy/paste can smuggle in — a valid GL1 string never contains any.
    string cleaned = str.replace_all(src, " ", "")
    array<string> toks = str.split(cleaned, "|")
    if array.size(toks) >= 3 and array.get(toks, 0) == "GL1"
        gParsed := true
        gTkr := str.upper(array.get(toks, 1))
        gDate := array.get(toks, 2)
        for [i, tok] in toks
            if i >= 3
                array<string> kv = str.split(tok, "=")
                if array.size(kv) == 2
                    string k = array.get(kv, 0)
                    string v = array.get(kv, 1)
                    if k == "spot"
                        gSpot := str.tonumber(v)
                    else if k == "zg"
                        gZg := str.tonumber(v)
                    else if k == "zgw"
                        gZgFallback := v == "1"
                    else if k == "cw"
                        gCw := str.tonumber(v)
                    else if k == "pw"
                        gPw := str.tonumber(v)
                    else
                        array<string> band = str.split(v, ":")
                        if array.size(band) == 2
                            float lo = str.tonumber(array.get(band, 0))
                            float hi = str.tonumber(array.get(band, 1))
                            if k == "emd"
                                gEmdLo := lo
                                gEmdHi := hi
                            else if k == "emw"
                                gEmwLo := lo
                                gEmwHi := hi
                            else if k == "emm"
                                gEmmLo := lo
                                gEmmHi := hi
                            else if k == "pi"
                                gPiLo := lo
                                gPiHi := hi

    _wipe()

    // Ranges first, then the single key levels so ZG/walls draw on top.
    if showEMM
        _band(gEmmLo, gEmmHi, COL_EMM, "Monthly EM")
    if showEMW
        _band(gEmwLo, gEmwHi, COL_EMW, "Weekly EM")
    if showPI
        _band(gPiLo, gPiHi, COL_PI, "HAR PI")
    if showEMD
        _band(gEmdLo, gEmdHi, COL_EMD, "Daily EM")
    if showWalls
        _lvl(gCw, COL_CW, "Call wall", line.style_solid)
        _lvl(gPw, COL_PW, "Put wall", line.style_solid)
    if showZG
        string zgTxt = gZgFallback ? "Zero Γ (fallback)" : "Zero Γ"
        string zgSty = gZgFallback ? line.style_dashed : line.style_solid
        _lvl(gZg, COL_ZG, zgTxt, zgSty)
    if showSpot
        _lvl(gSpot, COL_SPOT, "GL spot", line.style_dotted)

    // Status / warnings — always at least one visible row, so the script
    // can never fail silently: no table at all means it isn't running.
    table.clear(wtab, 0, 0, 0, 2)
    int row = 0
    if not gParsed
        string msg = str.length(cleaned) == 0 ? "GL Levels: paste a GL1 string in Settings → Inputs" : "GL Levels: couldn't read the string — recopy it from Gamma Lens"
        table.cell(wtab, 0, row, msg, text_color=color.orange, text_size=size.small, text_halign=text.align_left)
        row += 1
    else
        int nLevels = (na(gSpot) ? 0 : 1) + (na(gZg) ? 0 : 1) + (na(gCw) ? 0 : 1) + (na(gPw) ? 0 : 1) + (na(gEmdLo) ? 0 : 1) + (na(gEmwLo) ? 0 : 1) + (na(gEmmLo) ? 0 : 1) + (na(gPiLo) ? 0 : 1)
        table.cell(wtab, 0, row, "GL " + gTkr + " · " + gDate + " · " + str.tostring(nLevels) + " levels", text_color=color.gray, text_size=size.small, text_halign=text.align_left)
        row += 1
        if gTkr != str.upper(syminfo.ticker)
            table.cell(wtab, 0, row, "GL Levels: levels are for " + gTkr + " — chart is " + syminfo.ticker, text_color=color.orange, text_size=size.small, text_halign=text.align_left)
            row += 1
        bool stale = false
        array<string> dp = str.split(gDate, "-")
        if array.size(dp) == 3
            float yy = str.tonumber(array.get(dp, 0))
            float mm = str.tonumber(array.get(dp, 1))
            float dd = str.tonumber(array.get(dp, 2))
            if not na(yy) and not na(mm) and not na(dd)
                stale := time >= timestamp("America/New_York", int(yy), int(mm), int(dd), 0, 0, 0) + 86400000
        if stale
            table.cell(wtab, 0, row, "GL Levels: string dated " + gDate + " — may be stale", text_color=color.yellow, text_size=size.small, text_halign=text.align_left)
            row += 1
'''
