"""
Pre-Flight gates — pure functions, no I/O, no Streamlit, no DB.

A proposed trade passes through four independent gates in the seconds before
the human submits it at the broker:

  * behavioral — the user's OWN realized stats for this trade's category,
    shown verbatim ("your last N trades in this bucket: net −$X"). Red is a
    confrontation, never a block — the tool confronts, the human decides.
  * model — short strikes vs the weekly HAR band, ONLY for trades whose
    expiration is the planning week's Friday. Any other horizon returns
    status "na" with reason "no HAR model at this horizon" — the weekly
    model is never extrapolated down to intraday or out to monthlies.
  * structure — credit/width floor, breakevens vs expected move, per-leg
    bid-ask width and open-interest floors.
  * public — the broker's own margin/cost preflight, when available.

Roll-up: any red → RED, else any yellow → YELLOW, else GREEN. "na" gates are
itemized but never block — "unavailable" is an honest state, not a pass.

This module also owns the trade taxonomy (category keys like
``index|3-7DTE|iron_condor|credit``) so the fill-history categorizer in
``public_api`` and this proposed-trade path can never drift apart.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from phase1.ticker_config import TICKER_CONFIG
from range_finder.cockpit_config import CockpitConfig

_KNOWN_INDEXES = {"SPX", "SPXW", "XSP", "NDX", "XND", "RUT", "VIX", "DJX"}


# =============================================================================
# TRADE TAXONOMY (shared with public_api.categorize — keep keys stable)
# =============================================================================

@dataclass(frozen=True)
class ProposedLeg:
    option_type: str      # "call" | "put" | "equity"
    direction: str        # "buy" | "sell"
    strike: float


@dataclass(frozen=True)
class ProposedTrade:
    ticker: str
    expiration: str       # ISO YYYY-MM-DD
    quantity: int
    legs: tuple[ProposedLeg, ...]
    category: str = ""    # blank → categorized from the legs


def underlying_type(ticker: str) -> str:
    """"index" | "etf" | "single_name" — static lookup, never a network call."""
    t = (ticker or "").upper()
    cfg = TICKER_CONFIG.get(t)
    if cfg:
        cat = cfg.get("category", "stock")
        return {"index": "index", "etf": "etf"}.get(cat, "single_name")
    if t in _KNOWN_INDEXES:
        return "index"
    return "single_name"


def dte_bucket(dte: int | None) -> str:
    if dte is None or dte < 0:
        return "unknown"
    if dte == 0:
        return "0DTE"
    if dte <= 2:
        return "1-2DTE"
    if dte <= 7:
        return "3-7DTE"
    if dte <= 30:
        return "8-30DTE"
    return "30+DTE"


def structure_of_legs(legs) -> tuple[str, str]:
    """Classify legs → (structure, direction). Direction is inferred from
    strike geometry (a call vertical selling the lower strike is a credit
    spread, etc.) so it works on proposed trades that carry no premiums."""
    legs = list(legs)
    if not legs:
        return "other", "unknown"
    if any(l.option_type == "equity" for l in legs):
        return "equity", "debit" if legs[0].direction == "buy" else "credit"

    calls = [l for l in legs if l.option_type == "call"]
    puts = [l for l in legs if l.option_type == "put"]
    sells = [l for l in legs if l.direction == "sell"]
    buys = [l for l in legs if l.direction == "buy"]

    if len(legs) == 1:
        l = legs[0]
        return "single_leg", "debit" if l.direction == "buy" else "credit"

    if len(legs) == 2 and (len(calls) == 2 or len(puts) == 2) \
            and len(sells) == 1 and len(buys) == 1:
        sell, buy = sells[0], buys[0]
        if sell.option_type == "call":
            direction = "credit" if sell.strike < buy.strike else "debit"
        else:
            direction = "credit" if sell.strike > buy.strike else "debit"
        return "vertical", direction

    if len(legs) == 2 and len(calls) == 1 and len(puts) == 1 \
            and len({l.direction for l in legs}) == 1:
        same_strike = abs(calls[0].strike - puts[0].strike) < 1e-9
        structure = "straddle" if same_strike else "strangle"
        return structure, "debit" if legs[0].direction == "buy" else "credit"

    if (len(legs) == 4 and len(calls) == 2 and len(puts) == 2
            and sum(1 for l in calls if l.direction == "sell") == 1
            and sum(1 for l in puts if l.direction == "sell") == 1):
        call_sell = next(l for l in calls if l.direction == "sell")
        call_buy = next(l for l in calls if l.direction == "buy")
        put_sell = next(l for l in puts if l.direction == "sell")
        put_buy = next(l for l in puts if l.direction == "buy")
        if call_sell.strike < call_buy.strike and put_sell.strike > put_buy.strike:
            return "iron_condor", "credit"

    return "other_multi", "unknown"


def category_key(trade: ProposedTrade, dte: int | None) -> str:
    """Stable 4-dim key: underlying|DTE bucket|structure|direction."""
    structure, direction = structure_of_legs(trade.legs)
    return f"{underlying_type(trade.ticker)}|{dte_bucket(dte)}|{structure}|{direction}"


# =============================================================================
# GATES
# =============================================================================

@dataclass
class GateResult:
    name: str                     # "behavioral" | "model" | "structure" | "public"
    status: str                   # "green" | "yellow" | "red" | "na"
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class PreflightReport:
    verdict: str                  # "GREEN" | "YELLOW" | "RED"
    gates: list[GateResult]
    trade: ProposedTrade
    category: str
    dte: int | None
    checked_at: str
    input_hash: str


def check_behavioral(category: str, stats: dict | None, cfg: CockpitConfig) -> GateResult:
    if stats is None:
        return GateResult("behavioral", "na",
                          ["Public fill history not configured or not synced — "
                           "behavioral gate unavailable."])
    n = int(stats.get("n_trades") or 0)
    net = float(stats.get("total_pnl") or 0.0)
    win = stats.get("win_rate")
    win_txt = f"{win * 100:.0f}%" if win is not None else "—"
    summary = (f"Your last {n} trades in `{category}`: net "
               f"{'-' if net < 0 else '+'}${abs(net):,.0f}, win rate {win_txt}.")
    details = {"category": category, **stats}
    if n == 0:
        return GateResult("behavioral", "na",
                          [f"No history yet in `{category}` — nothing to confront."],
                          details)
    if n < cfg.min_sample:
        return GateResult("behavioral", "yellow",
                          [summary,
                           f"Small sample (n={n} < {cfg.min_sample}) — shown as a "
                           f"caveat, not a signal."], details)
    if net < 0:
        return GateResult("behavioral", "red",
                          [summary,
                           "This bucket has cost you money over a meaningful sample. "
                           "A red flag, not a block — the tool confronts, you decide."],
                          details)
    return GateResult("behavioral", "green", [summary], details)


def check_model(trade: ProposedTrade, forecast: dict | None, anchor: float | None,
                week_friday_exp: str | None, cfg: CockpitConfig) -> GateResult:
    if not week_friday_exp or trade.expiration != week_friday_exp:
        return GateResult("model", "na",
                          ["no HAR model at this horizon — the weekly model is "
                           "never extrapolated to other DTEs."])
    if not forecast or not anchor or anchor <= 0:
        return GateResult("model", "na",
                          ["Weekly HAR forecast or Monday anchor unavailable for "
                           f"{trade.ticker} — model gate cannot run."])
    point_pct = float(forecast.get("point_pct") or 0)
    upper_pct = float(forecast.get("upper_pct") or 0)
    if point_pct <= 0:
        return GateResult("model", "na", ["Forecast has no usable point estimate."])

    em_pts = anchor * point_pct / 2
    point_up, point_dn = anchor * (1 + point_pct / 2), anchor * (1 - point_pct / 2)
    pi_up = anchor * (1 + upper_pct / 2) if upper_pct > 0 else None
    pi_dn = anchor * (1 - upper_pct / 2) if upper_pct > 0 else None

    shorts = [l for l in trade.legs if l.direction == "sell"
              and l.option_type in ("call", "put")]
    if not shorts:
        return GateResult("model", "na",
                          ["No short option legs — the band check applies to "
                           "short strikes only."])

    statuses, reasons, leg_details = [], [], []
    for l in shorts:
        if l.option_type == "call":
            outside_pi = pi_up is not None and l.strike >= pi_up
            outside_point = l.strike >= point_up
            dist = l.strike - anchor
        else:
            outside_pi = pi_dn is not None and l.strike <= pi_dn
            outside_point = l.strike <= point_dn
            dist = anchor - l.strike
        mult = dist / em_pts if em_pts > 0 else None
        leg_details.append({"leg": f"short {l.option_type} {l.strike:g}",
                            "distance_pts": round(dist, 2),
                            "em_multiple": round(mult, 2) if mult is not None else None})
        if outside_pi:
            statuses.append("green")
            reasons.append(f"Short {l.option_type} {l.strike:g} sits outside the "
                           f"PI band ({dist:+.1f} pts, {mult:.2f}× EM).")
        elif outside_point:
            statuses.append("yellow")
            reasons.append(f"Short {l.option_type} {l.strike:g} is between the point "
                           f"band and the PI band ({dist:+.1f} pts, {mult:.2f}× EM).")
        else:
            statuses.append("red")
            reasons.append(f"Short {l.option_type} {l.strike:g} is INSIDE the model's "
                           f"expected move ({dist:+.1f} pts, {mult:.2f}× EM).")
    worst = "red" if "red" in statuses else ("yellow" if "yellow" in statuses else "green")
    return GateResult("model", worst, reasons,
                      {"anchor": anchor, "em_pts": round(em_pts, 2),
                       "point_band": [round(point_dn, 2), round(point_up, 2)],
                       "pi_band": ([round(pi_dn, 2), round(pi_up, 2)]
                                   if pi_up is not None else None),
                       "legs": leg_details})


def _leg_quote(chain_quotes: dict, leg: ProposedLeg) -> dict | None:
    row = (chain_quotes or {}).get(float(leg.strike))
    if not row or f"{leg.option_type}_bid" not in row:
        return None
    bid = float(row.get(f"{leg.option_type}_bid", 0) or 0)
    ask = float(row.get(f"{leg.option_type}_ask", 0) or 0)
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
    return {"bid": bid, "ask": ask, "mid": mid,
            "oi": float(row.get(f"{leg.option_type}_oi", 0) or 0),
            "ba_pct": round((ask - bid) / mid * 100, 1) if mid > 0 else None}


def check_structure(trade: ProposedTrade, chain_quotes: dict | None,
                    em_pts: float | None, ref_price: float | None,
                    cfg: CockpitConfig) -> GateResult:
    structure, direction = structure_of_legs(trade.legs)
    option_legs = [l for l in trade.legs if l.option_type in ("call", "put")]
    if not option_legs:
        return GateResult("structure", "na", ["No option legs to structure-check."])
    if not chain_quotes:
        return GateResult("structure", "na",
                          ["No chain quotes for this expiration — structure "
                           "economics unchecked."])

    statuses: list[str] = []
    reasons: list[str] = []
    details: dict = {"structure": structure, "direction": direction}

    # Per-leg liquidity
    missing = []
    for l in option_legs:
        q = _leg_quote(chain_quotes, l)
        tag = f"{l.direction} {l.option_type} {l.strike:g}"
        if q is None:
            missing.append(tag)
            continue
        if q["oi"] <= 0:
            statuses.append("red")
            reasons.append(f"{tag}: open interest 0 — effectively untraded.")
        elif q["oi"] < cfg.oi_floor:
            statuses.append("yellow")
            reasons.append(f"{tag}: OI {q['oi']:.0f} below the {cfg.oi_floor} floor.")
        if q["ba_pct"] is not None:
            if q["ba_pct"] > 2 * cfg.bid_ask_flag_pct:
                statuses.append("red")
                reasons.append(f"{tag}: bid-ask {q['ba_pct']:.1f}% of mid (> "
                               f"{2 * cfg.bid_ask_flag_pct:.0f}%) — fills will bleed.")
            elif q["ba_pct"] > cfg.bid_ask_flag_pct:
                statuses.append("yellow")
                reasons.append(f"{tag}: bid-ask {q['ba_pct']:.1f}% of mid.")
    if missing:
        statuses.append("red")
        reasons.append("No quotable market for: " + ", ".join(missing) + ".")

    # Credit / width and breakevens for defined-risk credit structures
    if direction == "credit" and structure in ("vertical", "iron_condor"):
        credit = 0.0
        priceable = True
        breakevens: list[tuple[str, float]] = []
        for opt_type in ("call", "put"):
            side_legs = [l for l in option_legs if l.option_type == opt_type]
            if len(side_legs) != 2:
                continue
            sell = next((l for l in side_legs if l.direction == "sell"), None)
            buy = next((l for l in side_legs if l.direction == "buy"), None)
            if not sell or not buy:
                continue
            qs, qb = _leg_quote(chain_quotes, sell), _leg_quote(chain_quotes, buy)
            if qs is None or qb is None:
                priceable = False
                continue
            credit += max(0.0, qs["bid"] - qb["ask"])
        widths = []
        for opt_type in ("call", "put"):
            side_legs = [l for l in option_legs if l.option_type == opt_type]
            if len(side_legs) == 2:
                widths.append(abs(side_legs[0].strike - side_legs[1].strike))
        width = max(widths) if widths else 0.0
        if priceable and width > 0:
            ratio = credit / width
            details.update({"credit": round(credit, 2), "width": width,
                            "credit_ratio": round(ratio, 3)})
            if ratio < cfg.min_credit_ratio:
                statuses.append("red")
                reasons.append(f"Credit/width {ratio:.2f} below the "
                               f"{cfg.min_credit_ratio:.2f} floor.")
            else:
                statuses.append("green")
                reasons.append(f"Credit/width {ratio:.2f} clears the "
                               f"{cfg.min_credit_ratio:.2f} floor "
                               f"(credit {credit:.2f} / width {width:g}).")
            shorts = [l for l in option_legs if l.direction == "sell"]
            for l in shorts:
                be = l.strike + credit if l.option_type == "call" else l.strike - credit
                breakevens.append((l.option_type, round(be, 2)))
            details["breakevens"] = breakevens
            if em_pts and em_pts > 0 and ref_price and ref_price > 0:
                for opt_type, be in breakevens:
                    dist = (be - ref_price) if opt_type == "call" else (ref_price - be)
                    mult = dist / em_pts
                    if mult < 1.0:
                        statuses.append("red")
                        reasons.append(f"{opt_type} breakeven {be:g} is inside 1× the "
                                       f"expected move ({mult:.2f}×).")
                    elif mult < 1.25:
                        statuses.append("yellow")
                        reasons.append(f"{opt_type} breakeven {be:g} is only "
                                       f"{mult:.2f}× the expected move.")
            else:
                reasons.append("Expected move unavailable — breakeven-vs-EM "
                               "sub-check skipped.")

    if not statuses:
        return GateResult("structure", "green",
                          reasons or ["Structure checks passed."], details)
    worst = "red" if "red" in statuses else ("yellow" if "yellow" in statuses else "green")
    return GateResult("structure", worst, reasons, details)


def check_public_preflight(response: dict | None) -> GateResult:
    if response is None:
        return GateResult("public", "na",
                          ["Public preflight unavailable — page still works; "
                           "margin/cost estimate skipped."])
    status = str(response.get("status", "")).lower()
    details = {k: response.get(k) for k in
               ("estimated_cost", "estimated_credit", "buying_power_impact",
                "margin_requirement", "messages") if k in response}
    if status in ("rejected", "error"):
        msgs = response.get("messages") or [response.get("error") or "rejected"]
        return GateResult("public", "red",
                          [f"Broker preflight rejected the order: {m}" for m in msgs],
                          details)
    bits = []
    if response.get("buying_power_impact") is not None:
        bits.append(f"buying-power impact ${float(response['buying_power_impact']):,.2f}")
    if response.get("margin_requirement") is not None:
        bits.append(f"margin requirement ${float(response['margin_requirement']):,.2f}")
    if response.get("estimated_credit") is not None:
        bits.append(f"estimated credit ${float(response['estimated_credit']):,.2f}")
    if response.get("estimated_cost") is not None:
        bits.append(f"estimated cost ${float(response['estimated_cost']):,.2f}")
    return GateResult("public", "green",
                      ["Broker preflight OK" + (": " + ", ".join(bits) if bits else ".")],
                      details)


# =============================================================================
# ROLL-UP
# =============================================================================

def run_preflight(
    trade: ProposedTrade,
    *,
    stats: dict | None,
    forecast: dict | None,
    anchor: float | None,
    week_friday_exp: str | None,
    chain_quotes: dict | None,
    em_pts: float | None,
    public_response: dict | None,
    cfg: CockpitConfig,
    dte: int | None,
    checked_at: str | None = None,
) -> PreflightReport:
    category = trade.category or category_key(trade, dte)
    gates = [
        check_behavioral(category, stats, cfg),
        check_model(trade, forecast, anchor, week_friday_exp, cfg),
        check_structure(trade, chain_quotes, em_pts, anchor, cfg),
        check_public_preflight(public_response),
    ]
    statuses = [g.status for g in gates]
    verdict = ("RED" if "red" in statuses
               else "YELLOW" if "yellow" in statuses
               else "GREEN")
    canonical = json.dumps(
        {"trade": asdict(trade), "gates": [(g.name, g.status) for g in gates],
         "category": category},
        sort_keys=True, separators=(",", ":"), default=str)
    return PreflightReport(
        verdict=verdict,
        gates=gates,
        trade=trade,
        category=category,
        dte=dte,
        checked_at=checked_at or datetime.now(timezone.utc).isoformat(),
        input_hash=hashlib.sha256(canonical.encode()).hexdigest()[:12],
    )


def report_to_row(report: PreflightReport, week_start: str | None) -> dict:
    """PreflightReport → verdict_persistence.save_preflight_log row dict."""
    def _gate(name: str) -> str | None:
        g = next((g for g in report.gates if g.name == name), None)
        return json.dumps(asdict(g), default=str) if g else None

    return {
        "checked_at": report.checked_at,
        "input_hash": report.input_hash,
        "ticker": report.trade.ticker,
        "category": report.category,
        "expiration": report.trade.expiration,
        "dte": report.dte,
        "in_weekly_window": int(any(
            g.name == "model" and g.status != "na" for g in report.gates)),
        "week_start": week_start,
        "verdict": report.verdict,
        "trade_json": json.dumps(asdict(report.trade), default=str),
        "behavioral_json": _gate("behavioral"),
        "model_json": _gate("model"),
        "structure_json": _gate("structure"),
        "public_json": _gate("public"),
        "reasons": json.dumps(
            [f"{g.name}:{g.status}: {r}" for g in report.gates for r in g.reasons],
            default=str),
        "updated_at": report.checked_at,
    }
