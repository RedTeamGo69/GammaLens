"""Plain-language, read-only recaps derived from the dashboard's saved rows."""
from collections import defaultdict
from datetime import date

import pandas as pd
import streamlit as st

from range_finder.forward_test.results import metrics
from range_finder.recommendations import TIER_KEYS, TIER_LABELS


MODEL_LABELS = {"M1_baseline": "M1 Baseline", "M2_vix": "M2 VIX",
                "M3_extended": "M3 Extended", "M4_full": "M4 Full"}


def ratio(hits, total):
    return f"{hits}/{total} · {hits / total:.1%}" if total else "Awaiting data"


def ticker_recap(rows):
    """Do not interpret missing path evidence as an absence of breaches."""
    m = metrics(rows)
    if not m["close_n"]:
        return "Weekly-close results are not available yet."
    failures = m["call_failures"] + m["put_failures"]
    parts = []
    if failures:
        if m["call_failures"]:
            parts.append(f"{m['call_failures']} closed above the upper boundary.")
        if m["put_failures"]:
            parts.append(f"{m['put_failures']} closed below the lower boundary.")
    elif m["close_n"] == len(rows):
        parts.append("Every range held at the weekly close.")
    else:
        parts.append("All scored closes were inside.")
    if m["recoveries"]:
        parts.append(f"{m['recoveries']} breached, then recovered by the close.")
    elif m["path_n"] == len(rows) and not m["breaches"]:
        parts.append("No breaches after capture.")
    if m["close_n"] < len(rows) or m["path_n"] < len(rows):
        parts.append("Some results are still pending or lack complete data.")
    return " ".join(parts)


def weekly_summary(rows):
    """One study/cohort/week only; never silently blend independent studies."""
    scopes = {(r["study_id"], r["cohort"], r["week_start"]) for r in rows}
    if len(scopes) > 1:
        raise ValueError("Select one study, capture cohort and week for a summary")
    m = metrics(rows)
    captured = sum(bool(r.get("forecast_id")) for r in rows)
    complete = bool(rows) and captured == len(rows) and m["close_n"] == len(rows) and m["path_n"] == len(rows)
    tickers = defaultdict(list)
    models = defaultdict(list)
    tiers = defaultdict(list)
    for r in rows:
        tickers[r["ticker"]].append(r)
        models[r["model"]].append(r)
        tiers[r["tier"]].append(r)
    ticker_rows = []
    for ticker, group in sorted(tickers.items()):
        tm = metrics(group)
        ticker_rows.append({"Ticker": ticker,
                            "Closed inside": ratio(tm["close_hits"], tm["close_n"]),
                            "Breached": f"{tm['breaches']}/{tm['path_n']}" if tm["path_n"] else "Awaiting data",
                            "What happened": ticker_recap(group)})
    model_rows = []
    for model, group in sorted(models.items()):
        mm = metrics(group)
        model_rows.append({"Model": MODEL_LABELS.get(model, model),
                           "Closed inside": ratio(mm["close_hits"], mm["close_n"]),
                           "Breached": f"{mm['breaches']}/{mm['path_n']}" if mm["path_n"] else "Awaiting data"})
    tier_rows = []
    for tier, label in zip(TIER_KEYS, TIER_LABELS):
        if tier not in tiers:
            continue
        tm = metrics(tiers[tier])
        tier_rows.append({"Range width": label,
                          "Closed inside": ratio(tm["close_hits"], tm["close_n"]),
                          "Breached": f"{tm['breaches']}/{tm['path_n']}" if tm["path_n"] else "Awaiting data"})
    spotlight = ""
    failed = [r for r in rows if r.get("close_eligible") and r.get("close_inside") is False]
    if failed:
        counts = {ticker: sum(r["ticker"] == ticker for r in failed) for ticker in tickers}
        biggest = max(counts.values())
        leaders = [ticker for ticker in sorted(tickers) if counts[ticker] == biggest]
        spotlight = (f"{', '.join(leaders)} accounted for the most weekly-close misses "
                     f"({biggest} each)." if len(leaders) > 1 else
                     f"{leaders[0]} accounted for {biggest} of the {len(failed)} weekly-close misses.")
        if len(leaders) == 1:
            group = tickers[leaders[0]]
            prices = {(r.get("reference"), r.get("final_close")) for r in group if r.get("close_eligible")}
            if len(prices) == 1:
                reference, close = prices.pop()
                if reference and close is not None:
                    spotlight += (f" Its opening reference was ${reference:,.2f}; it closed at "
                                  f"${close:,.2f} ({close / reference - 1:+.1%}).")
            gm = metrics(group)
            side = "call_short" if gm["call_failures"] and not gm["put_failures"] else (
                "put_short" if gm["put_failures"] and not gm["call_failures"] else None)
            held = [r for r in group if r.get("close_eligible") and r.get("close_inside") is True]
            if side and held:
                nearest = min(held, key=lambda r: abs(r[side] - r["final_close"]))
                margin = abs(nearest[side] - nearest["final_close"])
                spotlight += (f" The closest {'upper' if side == 'call_short' else 'lower'} boundary that held "
                              f"at the close was ${nearest[side]:,.2f} "
                              f"({MODEL_LABELS.get(nearest['model'], nearest['model'])}, "
                              f"{nearest['tier_label']}), a ${margin:,.2f} margin.")
    return {"metrics": m, "captured": captured, "complete": complete,
            "tickers": ticker_rows, "models": model_rows, "tiers": tier_rows,
            "spotlight": spotlight, "ticker_count": len(tickers),
            "model_count": len(models), "tier_count": len(tiers)}


def _choose(label, options, key, *, format_func=str):
    # Mirror into a non-widget key so switching views does not erase the choice.
    saved = f"_{key}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(saved, options[0])
    if st.session_state[key] not in options:
        st.session_state[key] = options[0]
    value = st.selectbox(label, options, key=key, format_func=format_func)
    st.session_state[saved] = value
    return value


def render_summary(rows, studies):
    """Return the explicitly scoped rows for the shared workbook download."""
    if not rows:
        for study in studies:
            st.caption(f"First eligible week: {study['start_week']}")
        return []
    scoped = list(rows)
    for field, label in (("study_id", "Study"), ("cohort", "Capture cohort")):
        options = sorted({r[field] for r in scoped})
        if len(options) > 1:
            chosen = _choose(label, options, f"ft_summary_{field}")
            scoped = [r for r in scoped if r[field] == chosen]
    weeks = sorted({r["week_start"] for r in scoped}, reverse=True)
    week = _choose("Week", weeks, "ft_summary_week",
                   format_func=lambda value: "Week of " + date.fromisoformat(value).strftime("%b %d, %Y"))
    scoped = [r for r in scoped if r["week_start"] == week]
    result = weekly_summary(scoped)
    m = result["metrics"]
    with st.container(border=True):
        st.subheader("The week at a glance")
        if m["close_n"]:
            st.markdown(f"**{m['close_hits']} of {m['close_n']} scored predictions finished inside their ranges "
                        f"({m['close_hit_rate']:.1%}).**")
        else:
            st.markdown("**Waiting for weekly-close results.**")
        if result["complete"]:
            st.caption(f"All {result['captured']} predictions were captured and scored with complete close and breach data.")
        else:
            st.warning(f"Results are not complete: {result['captured']} forecasts captured; "
                       f"{m['close_n']} have valid weekly closes and {m['path_n']} have complete breach data.")
            st.caption(f"Pending: {m['pending']} · Missed: {m['missed']} · Unavailable: {m['unavailable']} · "
                       f"Incomplete/invalid: {m['incomplete']}")
        if m["path_n"]:
            st.write(f"{m['breaches']} of {m['path_n']} predictions with complete path data broke outside "
                     "a boundary after capture.")
        if m["recovery_n"]:
            st.write(f"Of the {m['recovery_n']} breached predictions with a valid weekly close, "
                     f"{m['recoveries']} recovered by that close.")
        if m["close_n"]:
            st.caption(f"Closed above the upper boundary: {m['call_failures']} · "
                       f"Closed below the lower boundary: {m['put_failures']}")
    st.caption(f"{result['ticker_count']} tickers · {result['model_count']} models · "
               f"{result['tier_count']} range widths · {len(scoped)} comparison rows. "
               "These are overlapping predictions for one market week, not independent trading weeks.")
    st.subheader("By ticker")
    st.table(pd.DataFrame(result["tickers"]), hide_index=True)
    if result["spotlight"]:
        # Currency pairs must not become Markdown/KaTeX math delimiters.
        st.markdown(result["spotlight"].replace("$", r"\$"))
    st.subheader("Models and range widths")
    for column, key in zip(st.columns(2), ("models", "tiers")):
        with column:
            st.table(pd.DataFrame(result[key]), hide_index=True)
    st.caption("Lower PI is the tighter range; Effective (+buffer) includes the model's buffer. "
               "Wider boundaries give prices more room, so a higher hit rate alone does not mean better trading performance.")
    st.caption("This is a one-week comparison, not enough to establish a winning model. "
               "Results measure price-range accuracy, not trading profits or verified option settlement.")
    return scoped
