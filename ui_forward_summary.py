"""Plain-language, read-only recaps derived from the dashboard's saved rows."""
from collections import defaultdict
from datetime import date
from itertools import combinations

import pandas as pd
import streamlit as st

from range_finder.forward_test.results import metrics
from range_finder.forward_workbook import ft_class
from range_finder.recommendations import TIER_KEYS, TIER_LABELS
from ui_theme import TOKENS


MODEL_LABELS = {"M1_baseline": "M1 Baseline", "M2_vix": "M2 VIX",
                "M3_extended": "M3 Extended", "M4_full": "M4 Full"}
# Colour = how many ranges broke; the dot's position shows which side, and the
# hover text names it. Value = theme colour.
# Labels name one strike, never "every range", so a one-sided miss doesn't read as both sides.
LANDING_OUTCOMES = {"Inside every range": "green", "Past a tighter strike only": "amber",
                    "Past even the widest strike": "red"}


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
        parts.append("No breaches during the tracking window.")
    if m["close_n"] < len(rows) or m["path_n"] < len(rows):
        parts.append("Some results are still pending or lack complete data.")
    return " ".join(parts)


def weekly_summary(rows):
    """One study/cohort only; never silently blend independent studies.

    Several weeks of the same study may be pooled — the caller labels the
    result as a multi-week recap so it is not read as one market week.
    """
    scopes = {(r["study_id"], r["cohort"]) for r in rows}
    if len(scopes) > 1:
        raise ValueError("Select one study and capture cohort for a summary")
    week_count = len({r["week_start"] for r in rows})
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
            # Each week has its own opening reference; quote one only for a single week.
            if week_count == 1 and len(prices) == 1:
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
                # Pooled weeks have different closes, so name the week the margin came from.
                week = (", week of " + _week_label(nearest["week_start"])) if week_count > 1 else ""
                spotlight += (f" The closest {'upper' if side == 'call_short' else 'lower'} boundary that held "
                              f"at the close was ${nearest[side]:,.2f} "
                              f"({MODEL_LABELS.get(nearest['model'], nearest['model'])}, "
                              f"{nearest['tier_label']}{week}), a ${margin:,.2f} margin.")
    return {"metrics": m, "captured": captured, "complete": complete,
            "tickers": ticker_rows, "models": model_rows, "tiers": tier_rows,
            "spotlight": spotlight, "ticker_count": len(tickers), "week_count": week_count,
            "model_count": len(models), "tier_count": len(tiers)}


def _shown_model(ticker, models):
    # Mirrors ui_spread_finder._default_model_for_ticker: the model Quasimodo shows.
    preferred = "M2_vix" if ft_class(ticker) == "Stock" else "M3_extended"
    return preferred if preferred in models else min(models)


def close_landings(rows):
    """Latest week in scope: each ticker's nested ranges and its close, as moves from the opening reference.

    One model per ticker; four models' overlapping bars would be unreadable.
    """
    week = max(r["week_start"] for r in rows)
    rows = [r for r in rows if r["week_start"] == week and r.get("reference")]
    models = {r["model"] for r in rows}
    shown = {r["ticker"]: _shown_model(r["ticker"], models) for r in rows}
    ranges, closes = [], {}
    # Widest first so each tighter range is drawn on top of the one around it.
    for r in sorted(rows, key=lambda r: -TIER_KEYS.index(r["tier"])):
        if r["model"] != shown[r["ticker"]]:
            continue
        ref = r["reference"]
        ranges.append({"Ticker": r["ticker"], "Model": MODEL_LABELS.get(r["model"], r["model"]),
                       "Range": r["tier_label"], "Put short": r["put_short"] / ref - 1,
                       "Call short": r["call_short"] / ref - 1,
                       "Strikes": f"${r['put_short']:,.2f} – ${r['call_short']:,.2f}"})
        if r.get("close_eligible") and r.get("final_close") is not None:
            c = closes.setdefault(r["ticker"], {"Ticker": r["ticker"], "Close": r["final_close"] / ref - 1,
                                                "Price": f"${r['final_close']:,.2f}", "held": 0, "n": 0})
            c["n"] += 1
            c["held"] += r.get("close_inside") is True
            if r["final_close"] > r["call_short"]:
                c["above"] = True
    # Ranges that snapped to the same listed strikes draw as one bar; the hover names them all.
    same = defaultdict(list)
    for r in ranges:
        same[r["Ticker"], r["Strikes"]].append(r["Range"])
    for r in ranges:
        r["Ranges"] = " = ".join(reversed(same[r["Ticker"], r["Strikes"]]))
    for c in closes.values():
        above = c.pop("above", False)
        where, side = ("Above", "call") if above else ("Below", "put")
        c["Outcome"] = ("Inside every range" if c["held"] == c["n"] else
                        "Past a tighter strike only" if c["held"] else "Past even the widest strike")
        c["Result"] = (c["Outcome"] if c["held"] == c["n"] else
                       f"{where} the tighter {side}s" if c["held"] else f"{where} every {side} short")
    return week, sorted(set(shown.values())), ranges, list(closes.values())


def width_tradeoff(rows):
    """One point per model and range width: how often the close held against how wide the range was."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["tier"])].append(r)
    points = []
    for (model, tier), group in sorted(groups.items(), key=lambda g: (g[0][0], TIER_KEYS.index(g[0][1]))):
        m = metrics(group)
        if m["close_n"] and m["average_width_ratio"] is not None:
            points.append({"Model": MODEL_LABELS.get(model, model), "Range": TIER_LABELS[TIER_KEYS.index(tier)],
                           "order": TIER_KEYS.index(tier), "Width": m["average_width_ratio"],
                           "Closed inside": m["close_hit_rate"], "Scored": m["close_n"]})
    # Dots that land on each other answer the hover with only the top one, so every
    # dot carries the whole stack's details.
    def _join(values):
        return values[0] if len(set(values)) == 1 else " / ".join(values)
    for p in points:
        here = [q for q in points if abs(q["Width"] - p["Width"]) < .0025
                and abs(q["Closed inside"] - p["Closed inside"]) < .005]
        p.update({"Models": " / ".join(q["Model"] for q in here),
                  "Ranges": _join([q["Range"] for q in here]),
                  "Widths": _join([f'{q["Width"]:.2%}' for q in here]),
                  "Inside": _join([f'{q["Closed inside"]:.1%}' for q in here]),
                  "Counts": _join([str(q["Scored"]) for q in here])})
    return points


def hidden_lines(points, width_tol=.0025):
    """Model pairs whose every point lands on the other's, so one line covers the other."""
    by_model = defaultdict(dict)
    for p in points:
        by_model[p["Model"]][p["Range"]] = (p["Width"], p["Closed inside"])
    return [(a, b) for a, b in combinations(sorted(by_model), 2)
            if by_model[a].keys() == by_model[b].keys()
            and all(abs(by_model[a][k][0] - by_model[b][k][0]) < width_tol
                    and by_model[a][k][1] == by_model[b][k][1] for k in by_model[a])]


def _landing_chart(ranges, closes):
    import altair as alt
    # Every ticker is labelled; Vega otherwise drops alternate labels when rows are tight.
    y = alt.Y("Ticker:N", title=None, axis=alt.Axis(labelOverlap=False))
    bars = alt.Chart(pd.DataFrame(ranges)).mark_bar(size=12).encode(
        x=alt.X("Put short:Q", axis=alt.Axis(format="+.1~%"), title="Move from the opening reference"),
        x2="Call short:Q", y=y,
        color=alt.Color("Range:N", scale=alt.Scale(domain=list(TIER_LABELS),
                                                   range=["#9cc2ff", "#6c9be0", "#4a77b5", "#33578a"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Ticker", "Model", alt.Tooltip("Ranges", title="Range"), "Strikes"])
    reference = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[3, 3], opacity=.5).encode(x="x:Q")
    layers = bars + reference
    if closes:
        layers += alt.Chart(pd.DataFrame(closes)).mark_point(
            filled=True, size=80, opacity=1, stroke=TOKENS["bg_base"], strokeWidth=1.5).encode(
            x="Close:Q", y=y,
            color=alt.Color("Outcome:N", scale=alt.Scale(
                domain=list(LANDING_OUTCOMES), range=[TOKENS[c] for c in LANDING_OUTCOMES.values()]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Ticker", alt.Tooltip("Price", title="Close"),
                     alt.Tooltip("Close:Q", format="+.2%", title="Move"), "Result"])
    tickers = len({r["Ticker"] for r in ranges})
    # Streamlit's height covers both legends and the x-axis (~190px), not just the rows;
    # a flat floor squeezed few-ticker weeks into one overlapping row.
    return layers.resolve_scale(color="independent").properties(height=200 + 30 * tickers)


def _tradeoff_chart(points):
    import altair as alt
    base = alt.Chart(pd.DataFrame(points)).encode(
        # ".1~%" labels half-percent ticks as 7.5%, not a second "8%".
        x=alt.X("Width:Q", axis=alt.Axis(format=".1~%"), scale=alt.Scale(zero=False),
                title="Average range width (share of the opening price)"),
        y=alt.Y("Closed inside:Q", axis=alt.Axis(format=".1~%"), scale=alt.Scale(zero=False),
                title="Closed inside"),
        # No four hues stay distinct when every pair can touch, so the baseline is the
        # neutral and the three richer models get a CVD-validated blue/orange/aqua.
        color=alt.Color("Model:N", scale=alt.Scale(
            domain=list(MODEL_LABELS.values()),
            range=[TOKENS["text_secondary"], "#3987e5", "#d95926", "#199e70"]),
            legend=alt.Legend(orient="top", title=None, symbolOpacity=1)))
    lines = base.mark_line(strokeWidth=2, opacity=.7).encode(order="order:Q")
    dots = base.mark_point(filled=True, size=90, opacity=1, stroke=TOKENS["bg_base"], strokeWidth=1.5).encode(
        shape=alt.Shape("Range:N", sort=list(TIER_LABELS), legend=alt.Legend(
            orient="bottom", title=None, symbolFillColor=TOKENS["text_muted"], symbolStrokeColor=TOKENS["text_muted"])),
        tooltip=[alt.Tooltip("Models", title="Model"), alt.Tooltip("Ranges", title="Range"),
                 alt.Tooltip("Widths", title="Average width"), alt.Tooltip("Inside", title="Closed inside"),
                 alt.Tooltip("Counts", title="Scored")])
    return (lines + dots).properties(height=380)


def _week_label(value):
    return date.fromisoformat(value).strftime("%b %d, %Y")


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


def _choose_many(label, options, key, *, default=(), format_func=str):
    """Multiselect twin of _choose; an empty pick means no filter, like Detailed data."""
    saved = f"_{key}"
    previous = st.session_state.get(key, st.session_state.get(saved, list(default)))
    kept = [value for value in previous if value in options]
    # Picks that all fell out of scope (e.g. a study switch) restore the default
    # rather than silently widening to every week.
    kept = kept if kept or not previous else list(default)
    # The widget's built-in "Select all" chips every option, which means the
    # same as no filter; show it as the "All" placeholder instead.
    st.session_state[key] = [] if len(options) > 1 and set(kept) == set(options) else kept
    value = st.multiselect(label, options, key=key, format_func=format_func, placeholder="All")
    st.session_state[saved] = list(value)
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
    # Options come from the whole study so a ticker or model stays pickable
    # while the week selection changes; the latest week is the opening view.
    pickers = (("week_start", "Week", weeks, [weeks[0]], lambda value: "Week of " + _week_label(value)),
               ("ticker", "Ticker", sorted({r["ticker"] for r in scoped}), [], str),
               ("model", "Model", sorted({r["model"] for r in scoped}), [],
                lambda value: MODEL_LABELS.get(value, value)))
    for column, (field, label, options, default, fmt) in zip(st.columns(len(pickers)), pickers):
        with column:
            chosen = _choose_many(label, options, f"ft_summary_{field}", default=default, format_func=fmt)
        if chosen:
            scoped = [r for r in scoped if r[field] in chosen]
    if not scoped:
        st.info("No predictions match these filters.")
        return []
    result = weekly_summary(scoped)
    m = result["metrics"]
    weeks_n = result["week_count"]
    with st.container(border=True):
        st.subheader("The week at a glance" if weeks_n == 1 else f"{weeks_n} weeks at a glance")
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
                     "a boundary during the tracking window.")
        if m["recovery_n"]:
            st.write(f"Of the {m['recovery_n']} breached predictions with a valid weekly close, "
                     f"{m['recoveries']} recovered by that close.")
        if m["close_n"]:
            st.caption(f"Closed above the upper boundary: {m['call_failures']} · "
                       f"Closed below the lower boundary: {m['put_failures']}")
    st.caption(f"{result['ticker_count']} tickers · {result['model_count']} models · "
               f"{result['tier_count']} range widths · {len(scoped)} comparison rows. " +
               ("These are overlapping predictions for one market week, not independent trading weeks."
                if weeks_n == 1 else
                f"Pooled across {weeks_n} market weeks; predictions within each week overlap, "
                "so rows are not independent trials."))
    week, shown_models, ranges, closes = close_landings(scoped)
    if ranges:
        st.subheader("Where each close landed")
        st.altair_chart(_landing_chart(ranges, closes))
        models_note = (f"{MODEL_LABELS.get(shown_models[0], shown_models[0])} ranges" if len(shown_models) == 1 else
                       "each ticker shows the model Quasimodo defaults to (M3 Extended for indices and ETFs, "
                       "M2 VIX for single names); hover a bar for its model or pick one model above")
        st.caption(f"Week of {_week_label(week)}{' (latest selected)' if weeks_n > 1 else ''} · {models_note}. "
                   "Bars run from put short to call short, brightest = tightest range; the dot is the weekly close.")
    st.subheader("By ticker")
    st.table(pd.DataFrame(result["tickers"]), hide_index=True)
    if result["spotlight"]:
        # Currency pairs must not become Markdown/KaTeX math delimiters.
        st.markdown(result["spotlight"].replace("$", r"\$"))
    st.subheader("Models and range widths")
    points = width_tradeoff(scoped)
    if len(points) > 1:
        st.altair_chart(_tradeoff_chart(points))
        st.caption("Higher and further left is better: more closes inside with narrower ranges. "
                   "Lines join each model's four range widths." +
                   "".join(f" {a} and {b} gave near-identical results, so one line covers the other; "
                           "hover a dot to see both." for a, b in hidden_lines(points)))
    for column, key in zip(st.columns(2), ("models", "tiers")):
        with column:
            st.table(pd.DataFrame(result[key]), hide_index=True)
    st.caption("Lower PI is the tighter range; Effective (+buffer) includes the model's buffer. "
               "Wider boundaries give prices more room, so a higher hit rate alone does not mean better trading performance.")
    st.caption(("This is a one-week comparison" if weeks_n == 1 else f"This is a {weeks_n}-week comparison") +
               ", not enough to establish a winning model. "
               "Results measure price-range accuracy, not trading profits or verified option settlement.")
    return scoped
