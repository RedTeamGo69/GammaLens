"""Tests for the TradingView GL1 export format.

parse_tv_string is a line-for-line Python mirror of the Pine indicator's
parse block, so the round-trip tests here are what pin the Python
serializer and the Pine parser to the same format. If any test in this
file has to change, the GL1 version (and the Pine template) must be
re-checked for compatibility.
"""

import pytest

from phase1.tv_export import (
    PINE_INDICATOR_SOURCE,
    TV_FORMAT_VERSION,
    TVLevels,
    build_tv_levels,
    build_tv_string,
    missing_tokens,
    parse_tv_string,
    sanitize_tv_ticker,
)


def _full_levels() -> TVLevels:
    return TVLevels(
        ticker="SPX", date="2026-07-25",
        spot=6350.25, zero_gamma=6320.5, zero_gamma_fallback=False,
        call_wall=6400.0, put_wall=6250.0,
        em_daily=(6310.5, 6390.75), em_weekly=(6280.0, 6420.0),
        em_monthly=(6150.0, 6550.0), har_pi=(6270.2, 6440.8),
    )


# ── Serialization ────────────────────────────────────────────────────────

def test_full_serialization_exact_string():
    expected = ("GL1|SPX|2026-07-25|spot=6350.25|zg=6320.50|cw=6400.00"
                "|pw=6250.00|emd=6310.50:6390.75|emw=6280.00:6420.00"
                "|emm=6150.00:6550.00|pi=6270.20:6440.80")
    assert build_tv_string(_full_levels()) == expected


def test_zgw_token_rides_behind_zg_when_fallback():
    tv = TVLevels(ticker="SPX", date="2026-07-25",
                  zero_gamma=6320.5, zero_gamma_fallback=True)
    assert build_tv_string(tv) == "GL1|SPX|2026-07-25|zg=6320.50|zgw=1"


def test_zgw_omitted_when_zg_missing_even_if_flagged():
    tv = TVLevels(ticker="SPX", date="2026-07-25",
                  zero_gamma=None, zero_gamma_fallback=True, spot=6350.0)
    s = build_tv_string(tv)
    assert "zgw" not in s
    assert "zg=" not in s.replace("zgw", "")


def test_omission_rules():
    tv = TVLevels(ticker="QQQ", date="2026-07-25",
                  spot=None, zero_gamma=560.0, call_wall=570.0,
                  put_wall=None, har_pi=None, em_daily=None)
    s = build_tv_string(tv)
    assert s.startswith("GL1|QQQ|2026-07-25|")
    assert "spot=" not in s
    assert "pw=" not in s
    assert "pi=" not in s
    assert "emd=" not in s
    assert "zg=560.00" in s
    assert "cw=570.00" in s


def test_one_sided_band_is_omitted():
    tv = build_tv_levels(
        ticker="SPX", date="2026-07-25", spot=6350.0,
        levels={"zero_gamma": 6320.0, "zero_gamma_is_true_crossing": True},
        daily_em={"lower_level": None, "upper_level": 6400.0},
    )
    assert tv.em_daily is None
    assert "emd=" not in build_tv_string(tv)


def test_degenerate_band_is_omitted():
    tv = TVLevels(ticker="SPX", date="2026-07-25", em_weekly=(6300.0, 6300.0))
    assert "emw=" not in build_tv_string(tv)


def test_band_normalization_swaps_reversed_sides():
    tv = TVLevels(ticker="SPX", date="2026-07-25", har_pi=(6440.8, 6270.2))
    assert "pi=6270.20:6440.80" in build_tv_string(tv)


# ── build_tv_levels adapter ──────────────────────────────────────────────

def test_fallback_flag_from_explicit_false():
    tv = build_tv_levels(ticker="SPX", date="2026-07-25", spot=6350.0,
                         levels={"zero_gamma": 6320.0,
                                 "zero_gamma_is_true_crossing": False})
    assert tv.zero_gamma_fallback is True
    assert "zgw=1" in build_tv_string(tv)


def test_fallback_flag_from_empty_chain_shape():
    # The empty-chain path of find_key_levels returns no zero_gamma_type /
    # zero_gamma_is_true_crossing keys at all — must count as fallback.
    empty_chain = {"call_wall": 6350.0, "call_wall_gex": 0.0,
                   "put_wall": 6350.0, "put_wall_gex": 0.0,
                   "zero_gamma": 6350.0, "net_gex": 0.0}
    tv = build_tv_levels(ticker="SPX", date="2026-07-25", spot=6350.0,
                         levels=empty_chain)
    assert tv.zero_gamma_fallback is True


def test_healthy_levels_have_no_zgw():
    tv = build_tv_levels(ticker="SPX", date="2026-07-25", spot=6350.0,
                         levels={"zero_gamma": 6320.0,
                                 "zero_gamma_is_true_crossing": True})
    assert tv.zero_gamma_fallback is False
    assert "zgw" not in build_tv_string(tv)


def test_adapter_is_total_on_junk_inputs():
    tv = build_tv_levels(ticker=None, date="2026-07-25", spot="oops",
                         levels=None, daily_em="not-a-dict",
                         weekly_em=None, monthly_em={}, har_pi=("x", 5))
    assert tv.ticker == "NA"
    assert tv.spot is None
    assert tv.zero_gamma is None
    assert tv.em_daily is None
    assert tv.har_pi is None
    assert build_tv_string(tv) == "GL1|NA|2026-07-25"


# ── Ticker sanitization ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("spx", "SPX"),
    ("BRK.B", "BRK.B"),
    ("SP|X=:", "SPX"),
    ("€∆", "NA"),
    ("", "NA"),
    (None, "NA"),
    ("ABCDEFGHIJKLMNO", "ABCDEFGHIJ"),
])
def test_sanitize_tv_ticker(raw, expected):
    assert sanitize_tv_ticker(raw) == expected


def test_sanitized_ticker_cannot_carry_format_separators():
    for junk in ("A|B", "A=B", "A:B", "=cmd", "@SUM(A1)"):
        cleaned = sanitize_tv_ticker(junk)
        assert "|" not in cleaned and "=" not in cleaned and ":" not in cleaned


# ── Round-trip: serializer ↔ (mirrored) Pine parser ─────────────────────

def test_round_trip_full():
    tv = _full_levels()
    parsed = parse_tv_string(build_tv_string(tv))
    assert parsed is not None
    assert parsed["version"] == TV_FORMAT_VERSION
    assert parsed["ticker"] == "SPX"
    assert parsed["date"] == "2026-07-25"
    assert parsed["spot"] == pytest.approx(6350.25)
    assert parsed["zg"] == pytest.approx(6320.50)
    assert parsed["zgw"] is False
    assert parsed["cw"] == pytest.approx(6400.0)
    assert parsed["pw"] == pytest.approx(6250.0)
    assert parsed["emd"] == (pytest.approx(6310.5), pytest.approx(6390.75))
    assert parsed["emw"] == (pytest.approx(6280.0), pytest.approx(6420.0))
    assert parsed["emm"] == (pytest.approx(6150.0), pytest.approx(6550.0))
    assert parsed["pi"] == (pytest.approx(6270.2), pytest.approx(6440.8))


def test_round_trip_sparse():
    tv = TVLevels(ticker="XSP", date="2026-07-25",
                  zero_gamma=635.5, zero_gamma_fallback=True)
    parsed = parse_tv_string(build_tv_string(tv))
    assert parsed is not None
    assert parsed["ticker"] == "XSP"
    assert parsed["zg"] == pytest.approx(635.5)
    assert parsed["zgw"] is True
    for absent in ("spot", "cw", "pw", "emd", "emw", "emm", "pi"):
        assert parsed[absent] is None


# ── Parser robustness (mirrors Pine's na-on-garbage semantics) ──────────

def test_parser_rejects_wrong_version():
    assert parse_tv_string("GL2|SPX|2026-07-25|zg=6320.50") is None


def test_parser_rejects_short_header():
    assert parse_tv_string("GL1|SPX") is None
    assert parse_tv_string("") is None
    assert parse_tv_string(None) is None


def test_parser_ignores_unknown_tokens():
    parsed = parse_tv_string("GL1|SPX|2026-07-25|foo=1|zg=6320.50|bar=9:9:9")
    assert parsed is not None
    assert parsed["zg"] == pytest.approx(6320.5)
    assert "foo" not in parsed


def test_parser_leaves_non_numeric_fields_none_but_parses_rest():
    parsed = parse_tv_string("GL1|SPX|2026-07-25|zg=abc|cw=6400.00|emd=x:6390")
    assert parsed is not None
    assert parsed["zg"] is None
    assert parsed["cw"] == pytest.approx(6400.0)
    assert parsed["emd"] is None


# ── missing_tokens (UI caption helper) ───────────────────────────────────

def test_missing_tokens_agrees_with_serializer():
    tv = TVLevels(ticker="SPX", date="2026-07-25",
                  zero_gamma=6320.0, call_wall=6400.0,
                  em_weekly=(6280.0, 6420.0))
    missing = missing_tokens(tv)
    assert "zero gamma" not in missing
    assert "call wall" not in missing
    assert "weekly EM" not in missing
    assert {"put wall", "daily EM", "monthly EM", "HAR PI"} <= set(missing)


def test_missing_tokens_empty_when_everything_present():
    assert missing_tokens(_full_levels()) == []


# ── Pine template pinning ────────────────────────────────────────────────

def test_pine_template_structure():
    src = PINE_INDICATOR_SOURCE
    assert src.startswith("//@version=6")
    assert "indicator(" in src
    assert f'"{TV_FORMAT_VERSION}"' in src  # version literals stay in sync
    assert "input.string" in src
    assert "syminfo.ticker" in src
    assert "line.new" in src
    # Styling inputs: line length (bars back) and label text size.
    assert "input.int" in src
    # Ranges are High/Low line pairs — filled boxes made wide bands
    # (monthly EM, HAR PI) paint over the whole chart.
    assert "box.new" not in src
    # Lines stop at the last bar next to their labels — no right extension.
    assert "extend.both" not in src
    # Parsed values are series — hline() only takes constants, so the
    # template must never call it.
    assert "hline(" not in src
    # Parse must happen on the LAST bar, together with drawing: parsing on
    # barstate.isfirst compiled but silently drew nothing on real charts
    # (recalculation-timing), so isfirst must never reappear.
    assert "barstate.islast" in src
    assert "barstate.isfirst" not in src
    # Paste tolerance: smuggled spaces are stripped before parsing.
    assert "str.replace_all" in src


def test_pine_template_never_wraps_statements():
    """Every Pine statement must live on ONE line: Pine's continuation-line
    indentation rules are fragile, and a rejected wrap surfaces as the
    cryptic 'Missing closing parenthesis' (CE10015) compile error. With no
    wrapped lines, parentheses balance on every individual line."""
    for i, line in enumerate(PINE_INDICATOR_SOURCE.splitlines(), start=1):
        code = line.split("//")[0]  # comments can say anything
        assert code.count("(") == code.count(")"), (
            f"line {i} has unbalanced parens (wrapped statement?): {line!r}"
        )
        assert "\t" not in line, f"line {i} contains a tab character"


@pytest.mark.parametrize("key", ['"spot"', '"zg"', '"zgw"', '"cw"', '"pw"',
                                 '"emd"', '"emw"', '"emm"', '"pi"'])
def test_pine_template_parses_every_token_key(key):
    assert key in PINE_INDICATOR_SOURCE
