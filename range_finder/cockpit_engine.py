"""
Monday Cockpit decision engine — pure functions, no I/O, no Streamlit, no DB.

Turns one week's assembled inputs (weekly HAR forecast, ATM-straddle implied
move, Friday-expiry GEX walls, macro events, chain quotes) into a single
TRADE/SKIP verdict with explicit reasons and, when the inputs allow, a
concrete XSP iron-condor proposal. SKIP is a first-class outcome: every SKIP
carries at least one concrete gate reason, and the proposal is still built
whenever it can be so the user sees exactly *what* was skipped.

Unit conventions (the silent-bug minefield, spelled out once):
  * The HAR forecast is a whole-week HIGH−LOW range as a fraction of the
    Monday anchor open (`point_pct`), not a variance and not annualized.
  * The ATM straddle mid ≈ E[|terminal move|] in points. Repo convention
    (feature_builder.vix_implied_range): E[range] = 2·E[|move|], so the
    implied weekly range is `2 × straddle_mid ÷ anchor` — that is what the
    VRP ratio compares against `point_pct`.
  * "Expected move" for strike placement is the point-forecast HALF-range in
    points: `anchor × point_pct / 2` (the same ±half-range convention as
    har_model's pi_upper_px/pi_lower_px price band).

Strike discipline (the whole point of the tool):
  * Short strikes base off the frozen Monday anchor, never live spot.
  * GEX wall snapping is OUTWARD-only within tolerance — a wall inside the
    model strike never pulls the strike toward spot.
  * A credit below the floor reports SKIP; strikes are never tightened to
    chase premium.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from phase1.pop import condor_pop, resolve_short_delta
from range_finder.cockpit_config import CockpitConfig
from range_finder.spread_levels import (
    _lookup_chain_price,
    _snap_long_strike,
    _snap_to_chain_strike,
    round_call_short,
    round_put_short,
)

VRP_CONVENTION = (
    "implied weekly range = 2 × ATM straddle ÷ anchor (E[range] = 2·E[|move|]); "
    "compared against the HAR point range forecast"
)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class VerdictReason:
    """One line of the reasons list. severity: "gate" forces SKIP,
    "warning" is surfaced yellow, "info" is neutral context."""
    code: str
    severity: str
    text: str


@dataclass(frozen=True)
class SnapResult:
    """Outcome of trying to snap one short strike to its GEX wall."""
    strike: float
    snapped: bool
    inward_wall: bool
    note: str | None


@dataclass
class CondorProposal:
    ticker: str
    expiration: str
    anchor: float
    call_short: float
    call_long: float
    put_short: float
    put_long: float
    wing_width: float          # requested wing; actual per-side widths in leg math
    call_width: float
    put_width: float
    credit: float              # total natural credit, points
    credit_ratio: float        # credit ÷ max(call_width, put_width)
    max_profit: float          # credit × 100, $ per 1-lot
    max_loss: float            # (max width − credit) × 100, $ per 1-lot
    breakeven_up: float
    breakeven_dn: float
    em_pts: float              # point-forecast half-range, points
    k_used: float
    call_snapped_to_wall: bool
    put_snapped_to_wall: bool
    snap_notes: list[str] = field(default_factory=list)
    pop: float | None = None
    pop_source: str = "unavailable"   # "vendor" | "bs_iv" | "mixed" | "unavailable"
    leg_quotes: dict = field(default_factory=dict)
    breakeven_vs_em: dict = field(default_factory=dict)


@dataclass
class CockpitVerdict:
    verdict: str                       # "TRADE" | "SKIP"
    reasons: list[VerdictReason]
    proposal: CondorProposal | None
    vrp: dict
    inputs: dict


# =============================================================================
# PURE HELPERS
# =============================================================================

def build_leg_quote_map(calls: list[dict], puts: list[dict]) -> dict:
    """Chain rows → {strike: {call_bid/ask/mid/oi/iv/delta, put_...}}.

    Superset of the spread finder's chain_quotes shape — spread_levels'
    _snap_to_chain_strike / _lookup_chain_price read only the `{side}_bid` /
    `{side}_ask` keys, so this map drops into those helpers unchanged.
    """
    quotes: dict = {}

    def _absorb(rows: list[dict], prefix: str) -> None:
        for opt in rows or []:
            k = opt.get("strike")
            if k is None:
                continue
            row = quotes.setdefault(float(k), {})
            bid = float(opt.get("bid", 0) or 0)
            ask = float(opt.get("ask", 0) or 0)
            row[f"{prefix}_bid"] = bid
            row[f"{prefix}_ask"] = ask
            row[f"{prefix}_mid"] = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            row[f"{prefix}_oi"] = float(opt.get("openInterest", 0) or 0)
            row[f"{prefix}_iv"] = float(opt.get("impliedVolatility", 0) or 0)
            row[f"{prefix}_delta"] = float(opt.get("vendorDelta", 0) or 0)

    _absorb(calls, "call")
    _absorb(puts, "put")
    return quotes


def compute_vrp(
    straddle_em_pts: float | None,
    anchor: float | None,
    forecast_point_pct: float | None,
    cfg: CockpitConfig,
) -> dict:
    """Implied-vs-forecast comparison in one consistent unit family (week-range pct)."""
    implied = None
    if straddle_em_pts and straddle_em_pts > 0 and anchor and anchor > 0:
        implied = 2.0 * straddle_em_pts / anchor

    ratio = tripwire_ok = passes_gate = None
    if implied is not None and forecast_point_pct and forecast_point_pct > 0:
        ratio = implied / forecast_point_pct
        tripwire_ok = cfg.tripwire_low <= ratio <= cfg.tripwire_high
        passes_gate = ratio >= cfg.vrp_min_ratio

    return {
        "straddle_mid": straddle_em_pts,
        "implied_range_pct": implied,
        "forecast_point_pct": forecast_point_pct,
        "ratio": ratio,
        "tripwire_ok": tripwire_ok,
        "passes_gate": passes_gate,
        "convention": VRP_CONVENTION,
    }


def snap_never_inward(
    base_strike: float,
    wall: float | None,
    side: str,
    anchor: float,
    tolerance_pct: float,
) -> SnapResult:
    """Snap a short strike to its GEX wall ONLY when the wall is on the
    outward side of the base strike (calls: wall ≥ base; puts: wall ≤ base)
    and within tolerance_pct of the anchor. Snapping may only maintain or
    increase distance from spot — an inward wall keeps the model strike and
    is reported so the UI can warn that the wall sits inside the shorts.
    """
    if wall is None or wall <= 0 or anchor <= 0:
        return SnapResult(base_strike, False, False, None)

    outward = wall >= base_strike if side == "call" else wall <= base_strike
    if not outward:
        return SnapResult(
            base_strike, False, True,
            f"{side} wall {wall:g} sits inside the model short {base_strike:g} — "
            f"kept the model strike (walls never pull strikes toward spot)",
        )

    tol_pts = anchor * tolerance_pct / 100.0
    if abs(wall - base_strike) <= tol_pts:
        return SnapResult(
            wall, True, False,
            f"short {side} snapped outward to the {side} wall {wall:g} "
            f"(within {tolerance_pct:g}% of anchor)",
        )
    return SnapResult(base_strike, False, False, None)


def price_condor(
    call_short: float,
    call_long: float,
    put_short: float,
    put_long: float,
    chain_quotes: dict,
) -> dict | None:
    """Natural-credit condor economics from real chain quotes.

    Credit per side = short bid − long ask (floored at 0 — a $0.00 bid is a
    valid price for a worthless option). Returns None only when a leg is
    missing from the chain entirely, which callers treat as an
    insufficient-liquidity gate rather than a zero-credit structure.
    """
    cs_bid = _lookup_chain_price(chain_quotes, call_short, "call", "bid")
    cl_ask = _lookup_chain_price(chain_quotes, call_long, "call", "ask")
    ps_bid = _lookup_chain_price(chain_quotes, put_short, "put", "bid")
    pl_ask = _lookup_chain_price(chain_quotes, put_long, "put", "ask")
    if None in (cs_bid, cl_ask, ps_bid, pl_ask):
        return None

    call_credit = max(0.0, cs_bid - cl_ask)
    put_credit = max(0.0, ps_bid - pl_ask)
    credit = round(call_credit + put_credit, 2)

    # Long legs snap outward to listed strikes, so the two wings can end up
    # unequal; margin (and worst-case loss) follows the wider wing.
    call_width = round(call_long - call_short, 2)
    put_width = round(put_short - put_long, 2)
    max_width = max(call_width, put_width)

    def _leg(strike: float, side: str, role: str) -> dict:
        row = chain_quotes.get(strike) or {}
        bid = float(row.get(f"{side}_bid", 0) or 0)
        ask = float(row.get(f"{side}_ask", 0) or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        return {
            "strike": strike,
            "side": side,
            "role": role,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "oi": float(row.get(f"{side}_oi", 0) or 0),
            "ba_pct": round((ask - bid) / mid * 100, 1) if mid > 0 else None,
        }

    return {
        "credit": credit,
        "call_credit": round(call_credit, 2),
        "put_credit": round(put_credit, 2),
        "call_width": call_width,
        "put_width": put_width,
        "credit_ratio": credit / max_width if max_width > 0 else 0.0,
        "max_profit": round(credit * 100, 2),
        "max_loss": round((max_width - credit) * 100, 2),
        "breakeven_up": round(call_short + credit, 2),
        "breakeven_dn": round(put_short - credit, 2),
        "leg_quotes": {
            "call_short": _leg(call_short, "call", "short"),
            "call_long": _leg(call_long, "call", "long"),
            "put_short": _leg(put_short, "put", "short"),
            "put_long": _leg(put_long, "put", "long"),
        },
    }


def _short_leg_row(chain_quotes: dict, strike: float, side: str) -> dict:
    """Adapt a quote-map row back to the chain-row shape phase1.pop expects."""
    row = chain_quotes.get(strike) or {}
    return {
        "strike": strike,
        "vendorDelta": row.get(f"{side}_delta", 0.0),
        "impliedVolatility": row.get(f"{side}_iv", 0.0),
    }


def propose_condor(
    *,
    anchor: float,
    em_pts: float,
    k_eff: float,
    gex_levels: dict | None,
    chain_quotes: dict,
    cfg: CockpitConfig,
    strike_increment: float,
    wing_widths: list[float],
    expiration: str,
    ticker: str,
    t_years: float | None = None,
) -> tuple[CondorProposal | None, list[VerdictReason]]:
    """Model strikes → wall snap (outward-only) → increment rounding →
    listed-strike snap → wings → chain pricing → POP. Returns the proposal
    (None when unpriceable) plus the reasons generated along the way."""
    reasons: list[VerdictReason] = []
    wing = float(cfg.wing_width or (wing_widths[0] if wing_widths else 5.0))

    base_call = anchor + k_eff * em_pts
    base_put = anchor - k_eff * em_pts

    call_wall = (gex_levels or {}).get("call_wall") or None
    put_wall = (gex_levels or {}).get("put_wall") or None

    call_snap = snap_never_inward(base_call, call_wall, "call", anchor, cfg.snap_tolerance_pct)
    put_snap = snap_never_inward(base_put, put_wall, "put", anchor, cfg.snap_tolerance_pct)
    for snap in (call_snap, put_snap):
        if snap.note:
            reasons.append(VerdictReason(
                "wall_inside_shorts" if snap.inward_wall else "gex_snap",
                "warning" if snap.inward_wall else "info",
                snap.note,
            ))

    # Round away from spot to the ticker increment, then to strikes the chain
    # actually lists (still outward-only — see _snap_to_chain_strike).
    call_short = round_call_short(call_snap.strike, increment=strike_increment)
    put_short = round_put_short(put_snap.strike, increment=strike_increment)
    call_short = _snap_to_chain_strike(call_short, chain_quotes, "call", direction="up")
    put_short = _snap_to_chain_strike(put_short, chain_quotes, "put", direction="down")

    call_long = _snap_long_strike(call_short, wing, chain_quotes, "call")
    put_long = _snap_long_strike(put_short, wing, chain_quotes, "put")
    if call_long is None or put_long is None:
        missing = "call" if call_long is None else "put"
        reasons.append(VerdictReason(
            "insufficient_liquidity", "gate",
            f"No listed {missing} wing at-or-beyond {wing:g} points from the short "
            f"strike — the chain edge is too close to structure this condor.",
        ))
        return None, reasons

    if not (put_long < put_short < anchor < call_short < call_long):
        reasons.append(VerdictReason(
            "strike_ordering", "gate",
            f"Strike ordering broke after chain snapping "
            f"({put_long:g}/{put_short:g} · anchor {anchor:g} · {call_short:g}/{call_long:g}) — "
            f"the chain is too sparse around the model strikes to trust.",
        ))
        return None, reasons

    econ = price_condor(call_short, call_long, put_short, put_long, chain_quotes)
    if econ is None:
        reasons.append(VerdictReason(
            "insufficient_liquidity", "gate",
            "One or more legs has no quotable market on the Friday chain — "
            "cannot price the condor honestly.",
        ))
        return None, reasons

    d_call, src_call = resolve_short_delta(
        _short_leg_row(chain_quotes, call_short, "call"), anchor, t_years or 0.0, "call")
    d_put, src_put = resolve_short_delta(
        _short_leg_row(chain_quotes, put_short, "put"), anchor, t_years or 0.0, "put")
    pop = condor_pop(d_call, d_put)
    if pop is None:
        pop_source = "unavailable"
    elif src_call == src_put:
        pop_source = src_call
    else:
        pop_source = "mixed"

    proposal = CondorProposal(
        ticker=ticker,
        expiration=expiration,
        anchor=anchor,
        call_short=call_short,
        call_long=call_long,
        put_short=put_short,
        put_long=put_long,
        wing_width=wing,
        call_width=econ["call_width"],
        put_width=econ["put_width"],
        credit=econ["credit"],
        credit_ratio=econ["credit_ratio"],
        max_profit=econ["max_profit"],
        max_loss=econ["max_loss"],
        breakeven_up=econ["breakeven_up"],
        breakeven_dn=econ["breakeven_dn"],
        em_pts=em_pts,
        k_used=k_eff,
        call_snapped_to_wall=call_snap.snapped,
        put_snapped_to_wall=put_snap.snapped,
        snap_notes=[s.note for s in (call_snap, put_snap) if s.note],
        pop=pop,
        pop_source=pop_source,
        leg_quotes=econ["leg_quotes"],
        breakeven_vs_em={
            "up": round((econ["breakeven_up"] - anchor) / em_pts, 2) if em_pts > 0 else None,
            "dn": round((anchor - econ["breakeven_dn"]) / em_pts, 2) if em_pts > 0 else None,
        },
    )
    return proposal, reasons


def _liquidity_annotations(
    proposal: CondorProposal,
    cfg: CockpitConfig,
) -> list[VerdictReason]:
    """Per-leg bid-ask width / OI warnings. Annotation only — hard liquidity
    gating is Pre-Flight's job."""
    out: list[VerdictReason] = []
    wide, thin = [], []
    for name, leg in proposal.leg_quotes.items():
        ba = leg.get("ba_pct")
        if ba is not None and ba > cfg.bid_ask_flag_pct:
            wide.append(f"{name} {ba:g}%")
        if leg.get("oi", 0) < cfg.oi_floor:
            thin.append(f"{name} OI {leg.get('oi', 0):g}")
    if wide:
        out.append(VerdictReason(
            "wide_spread", "warning",
            f"Bid-ask wider than {cfg.bid_ask_flag_pct:g}% of mid on: {', '.join(wide)}.",
        ))
    if thin:
        out.append(VerdictReason(
            "low_oi", "warning",
            f"Open interest below {cfg.oi_floor} on: {', '.join(thin)}.",
        ))
    return out


# =============================================================================
# THE DECISION ENGINE
# =============================================================================

def evaluate_cockpit(
    *,
    ticker: str,
    anchor: float | None,
    spot: float | None,
    friday_exp: str | None,
    forecast: dict | None,
    straddle_em_pts: float | None,
    gex_levels: dict | None,
    events: list,
    chain_quotes: dict,
    cfg: CockpitConfig,
    strike_increment: float,
    wing_widths: list[float],
    sessions_to_expiry: int | None = None,
    t_years: float | None = None,
) -> CockpitVerdict:
    """Ordered gates → TRADE/SKIP + reasons + proposal (built even on SKIP
    whenever the inputs allow, so the user sees what was skipped)."""
    reasons: list[VerdictReason] = []

    def gate(code: str, text: str) -> None:
        reasons.append(VerdictReason(code, "gate", text))

    def warn(code: str, text: str) -> None:
        reasons.append(VerdictReason(code, "warning", text))

    def info(code: str, text: str) -> None:
        reasons.append(VerdictReason(code, "info", text))

    point_pct = float((forecast or {}).get("point_pct") or 0.0)

    # 1 — data sufficiency (never guess a missing input)
    if not anchor or anchor <= 0:
        gate("no_anchor",
             "No Monday anchor open captured for this week — run Weekly Setup "
             "(or wait for Monday's open); strikes must base off the frozen anchor.")
    if point_pct <= 0:
        gate("no_forecast",
             "No weekly HAR forecast available — refit via Weekly Setup before trading.")
    if straddle_em_pts is None or straddle_em_pts <= 0:
        gate("no_straddle",
             "No usable ATM straddle on the Friday expiry — the implied move "
             "(VRP check) cannot be computed.")
    if not chain_quotes:
        gate("no_chain",
             "Friday-expiry option chain unavailable — cannot place or price strikes.")
    if not friday_exp:
        gate("no_expiration", "No Friday expiration resolved for this week.")

    # 2/3 — units tripwire, then the VRP hard gate (a VRP verdict on an
    # unreconciled unit comparison is meaningless, so tripwire wins)
    vrp = compute_vrp(straddle_em_pts, anchor, point_pct if point_pct > 0 else None, cfg)
    ratio = vrp["ratio"]
    if ratio is not None:
        if not vrp["tripwire_ok"]:
            gate("units_tripwire",
                 f"Implied/forecast ratio {ratio:.2f} is outside the "
                 f"{cfg.tripwire_low:g}–{cfg.tripwire_high:g} reconcile band — units or "
                 f"data are suspect; refusing to trade on an unreconciled comparison.")
        elif not vrp["passes_gate"]:
            gate("vrp_thin",
                 f"VRP thin: implied weekly range {vrp['implied_range_pct'] * 100:.2f}% is "
                 f"{ratio:.2f}× the HAR forecast {point_pct * 100:.2f}% "
                 f"(needs ≥ {cfg.vrp_min_ratio:.2f}×) — selling a condor priced below "
                 f"the model's own vol forecast is negative-edge.")

    # 4 — event policy (tier 1 gates or widens; tier 2 only warns)
    k_eff = cfg.k_expected_move
    tier1 = [e for e in (events or []) if getattr(e, "tier", 2) == 1]
    tier2 = [e for e in (events or []) if getattr(e, "tier", 2) == 2]
    if tier1:
        names = ", ".join(f"{e.name.upper()} {e.date}" for e in tier1)
        if cfg.event_policy == "widen":
            k_eff = cfg.k_expected_move * cfg.event_widen_factor
            info("event_widen",
                 f"Tier-1 event(s) in the window ({names}) — short strikes widened "
                 f"×{cfg.event_widen_factor:g} per event policy.")
        else:
            for e in tier1:
                at = f" at {e.release_time_et} ET" if e.release_time_et else ""
                gate(f"event_{e.name}_skip",
                     f"{e.name.upper()} on {e.date}{at} falls inside the Mon–Fri "
                     f"window — event policy is SKIP.")
    if tier2:
        names = ", ".join(f"{e.name.upper()} {e.date}" for e in tier2)
        warn("event_tier2",
             f"Tier-2 event(s) in the window: {names} — not a gate, but expect "
             f"8:30 ET prints / OpEx pinning.")

    if gex_levels is None:
        info("no_gex", "GEX walls unavailable — strikes are model-only (no wall snap).")

    # 5/6/7 — proposal, credit floor, liquidity annotations
    proposal = None
    em_pts = 0.0
    if anchor and anchor > 0 and point_pct > 0 and chain_quotes and friday_exp:
        em_pts = anchor * point_pct / 2.0
        proposal, prop_reasons = propose_condor(
            anchor=anchor, em_pts=em_pts, k_eff=k_eff, gex_levels=gex_levels,
            chain_quotes=chain_quotes, cfg=cfg, strike_increment=strike_increment,
            wing_widths=wing_widths, expiration=friday_exp, ticker=ticker,
            t_years=t_years,
        )
        reasons.extend(prop_reasons)
        if proposal is not None:
            if proposal.credit_ratio < cfg.min_credit_ratio:
                gate("credit_floor",
                     f"Credit/width {proposal.credit_ratio:.2f} is below the "
                     f"{cfg.min_credit_ratio:.2f} floor at model-approved strikes — "
                     f"insufficient premium; strikes are never tightened to chase credit.")
            reasons.extend(_liquidity_annotations(proposal, cfg))

    verdict = "SKIP" if any(r.severity == "gate" for r in reasons) else "TRADE"

    inputs = {
        "ticker": ticker,
        "anchor": anchor,
        "spot": spot,
        "friday_exp": friday_exp,
        "em_pts": em_pts,
        "k_configured": cfg.k_expected_move,
        "k_used": k_eff,
        "sessions_to_expiry": sessions_to_expiry,
        "forecast_point_pct": point_pct if point_pct > 0 else None,
        "forecast_upper_pct": (forecast or {}).get("upper_pct"),
        "forecast_lower_pct": (forecast or {}).get("lower_pct"),
        "call_wall": (gex_levels or {}).get("call_wall"),
        "put_wall": (gex_levels or {}).get("put_wall"),
        "zero_gamma": (gex_levels or {}).get("zero_gamma"),
        "events": [
            {"name": e.name, "date": e.date, "tier": e.tier,
             "release_time_et": e.release_time_et}
            for e in (events or [])
        ],
        "config": asdict(cfg),
    }

    return CockpitVerdict(
        verdict=verdict,
        reasons=reasons,
        proposal=proposal,
        vrp=vrp,
        inputs=inputs,
    )
