"""
Sidebar card renderers for the terminal redesign.

Every function RETURNS an HTML string (a ``<section class="term-card">…``) so the
caller can assemble the whole left ``<aside>`` as a single ``st.html`` block,
keeping the aside|main flex layout intact. Pure presentation — all numbers are
pulled from dicts the (untouched) data pipeline already produced.
"""
from __future__ import annotations

from theme import COLORS
from ui_theme import esc, fmt_commas


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _regime_color(regime_text: str) -> str:
    t = (regime_text or "").lower()
    if "positive" in t:
        return COLORS["positive"]
    if "negative" in t:
        return COLORS["negative"]
    return COLORS["warning"]


def _score_color(label: str) -> str:
    return (
        COLORS["positive"] if label == "High"
        else COLORS["warning"] if label == "Moderate"
        else COLORS["negative"]
    )


def _bar_color_for_score(score: float) -> str:
    if score >= 80:
        return COLORS["positive"]
    if score >= 60:
        return COLORS["warning"]
    return COLORS["negative"]


# ─────────────────────────────────────────────────────────────────────────────
# Key Levels (hero)
# ─────────────────────────────────────────────────────────────────────────────
def render_key_levels(levels, spot, regime_info, confidence_info, ticker, mode_short) -> str:
    conf_score = confidence_info.get("score", 0)
    conf_label = confidence_info.get("label", "?")
    conf_color = _score_color(conf_label)
    reg_color = _regime_color(regime_info.get("regime", ""))

    def cell(dot_color, glow, label, value, val_color):
        sh = f"box-shadow:0 0 7px {glow};" if glow else ""
        return (
            '<div class="lvl-cell">'
            f'<div class="lvl-head"><span class="lvl-dot" style="background:{dot_color};{sh}"></span>'
            f'<span class="lvl-lbl">{esc(label)}</span></div>'
            f'<div class="lvl-val" style="color:{val_color};">{value}</div></div>'
        )

    grid = (
        cell("#fff", "", "Spot", fmt_commas(spot, 2), "#fff")
        + cell(COLORS["zero_gamma"], "rgba(37,216,239,.6)", "Zero Γ",
               fmt_commas(levels["zero_gamma"], 2), COLORS["zero_gamma"])
        + cell(COLORS["call_wall"], "rgba(43,232,138,.6)", "Call Wall",
               fmt_commas(levels["call_wall"], 0), COLORS["call_wall"])
        + cell(COLORS["put_wall"], "rgba(255,77,104,.6)", "Put Wall",
               fmt_commas(levels["put_wall"], 0), COLORS["put_wall"])
    )

    reg_short = (regime_info.get("regime", "") or "").replace("Gamma", "Γ").strip() or "—"
    return f"""
<section class="term-card hero">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <span class="card-eyebrow lit" style="margin:0;">Key Levels</span>
    <span style="font-size:9px;letter-spacing:.08em;color:var(--text-dim);font-family:var(--mono);">{esc(ticker)} · {esc(mode_short)}</span>
  </div>
  <div class="lvl-grid">{grid}</div>
  <div class="lvl-pillrow">
    <div class="lvl-pill" style="flex:2;">
      <span class="lvl-lbl">Regime</span>
      <span style="font-size:11px;font-weight:700;color:{reg_color};letter-spacing:.02em;">{esc(reg_short)}</span>
    </div>
    <div class="lvl-pill" style="flex:1;">
      <span class="lvl-lbl">Conf</span>
      <span style="font-family:var(--mono);font-size:13px;font-weight:600;color:{conf_color};">{conf_score:.0f}</span>
    </div>
  </div>
</section>
"""


# ─────────────────────────────────────────────────────────────────────────────
# GEX Stream
# ─────────────────────────────────────────────────────────────────────────────
def render_gex_stream(stats, levels, spot) -> str:
    gex_ratio = stats.get("gex_ratio")
    net_gex = stats.get("net_gex", 0)
    pc_ratio = stats.get("pc_ratio", 0)
    call_iv = stats.get("call_iv", 0)
    put_iv = stats.get("put_iv", 0)
    g = COLORS["positive"]; r = COLORS["negative"]
    cw = COLORS["call_wall"]; pw = COLORS["put_wall"]; zg = COLORS["zero_gamma"]
    tw = COLORS["text_white"]; amber = COLORS["charm_line"]

    if gex_ratio is None:
        gr_display = "∞" if net_gex > 0 else ("0.00" if net_gex < 0 else "—")
        gr_sigma = ""
        gr_color = g if net_gex > 0 else r
    else:
        gr_display = f"{gex_ratio:.2f}"
        gr_sigma = f"{abs(gex_ratio - 1.0) / 0.5:.1f}σ"
        gr_color = g if gex_ratio > 1 else r
    ng_color = g if net_gex > 0 else r
    ng_fmt = stats.get("net_gex_fmt", f"{net_gex:.0f}")

    nc = stats.get("net_charm_per_hour", 0.0)
    nc_fmt = stats.get("net_charm_per_hour_fmt", f"{nc:,.0f}")
    nc_color = r if nc > 0 else g if nc < 0 else COLORS["text_muted"]

    def cell(label, value, val_color, sub="", lbl_color="var(--text-dim)", span=False):
        sub_html = f' <span class="stream-sub">{esc(sub)}</span>' if sub else ""
        if span:
            return (
                '<div class="stream-cell span">'
                f'<div class="stream-lbl" style="color:{lbl_color};margin:0;">{esc(label)}</div>'
                f'<div class="stream-val" style="color:{val_color};">{value}{sub_html}</div></div>'
            )
        return (
            '<div class="stream-cell">'
            f'<div class="stream-lbl" style="color:{lbl_color};">{esc(label)}</div>'
            f'<div class="stream-val" style="color:{val_color};">{value}{sub_html}</div></div>'
        )

    gr_val = f'{gr_display} <span class="stream-sub">{gr_sigma}</span>' if gr_sigma else gr_display
    grid = (
        cell("GEX RATIO", gr_val, gr_color)
        + cell("NET GEX", ng_fmt, ng_color)
        + cell("CALL OI", f'{stats.get("call_oi", "0")}', cw, f'@{stats.get("call_oi_strike", 0):.0f}')
        + cell("PUT OI", f'{stats.get("put_oi", "0")}', pw, f'@{stats.get("put_oi_strike", 0):.0f}')
        + cell("POS GEX", f'{stats.get("pos_gex", "0")}', cw, f'@{stats.get("pos_gex_strike", 0):.0f}')
        + cell("NEG GEX", f'{stats.get("neg_gex", "0")}', pw, f'@{stats.get("neg_gex_strike", 0):.0f}')
        + cell("ZERO GAMMA", fmt_commas(levels.get("zero_gamma", 0), 2), zg, span=True)
        + cell("CALL IV", f"{call_iv:.1f}%", tw)
        + cell("PUT IV", f"{put_iv:.1f}%", tw)
        + cell("P/C OI", f"{pc_ratio:.2f}", tw)
        + cell("NET CHARM/HR", nc_fmt, nc_color, lbl_color=amber)
    )
    return f"""
<section class="term-card">
  <div class="card-eyebrow">📡 GEX Stream</div>
  <div class="stream-grid">{grid}</div>
</section>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Expected Move
# ─────────────────────────────────────────────────────────────────────────────
def render_expected_move_panel(em_analysis, spot, ticker="SPX") -> str:
    em = em_analysis.get("expected_move", {}) or {}
    overnight = em_analysis.get("overnight_move", {}) or {}
    classification = em_analysis.get("classification", {}) or {}
    level_ctx = em_analysis.get("level_context") or {}
    market_ctx = em_analysis.get("market_context", "live")

    if em.get("expected_move_pts") is None:
        return (
            '<section class="term-card"><div class="card-eyebrow">⚡ Expected Move</div>'
            '<div style="font-size:11px;color:var(--text-dim);">Expected move data not available.</div></section>'
        )

    straddle = em.get("straddle") or {}
    dte = straddle.get("dte")
    badge = "0DTE" if (dte is None or dte == 0) else f"{dte}DTE"
    em_pts = em.get("expected_move_pts", 0) or 0
    em_pct = em.get("expected_move_pct", 0) or 0
    lo = em.get("lower_level")
    hi = em.get("upper_level")

    # range bar spot position
    if lo is not None and hi is not None and hi > lo and spot is not None:
        spot_pct = max(0.0, min(100.0, (spot - lo) / (hi - lo) * 100))
    else:
        spot_pct = 50.0

    # Today's / session move
    on_pts = overnight.get("overnight_move_pts")
    on_pct = overnight.get("overnight_move_pct")
    on_label = "Today's Move" if market_ctx == "live" else "Session Move"
    on_color = COLORS["positive"] if (on_pts or 0) >= 0 else COLORS["negative"]
    on_arrow = "▲" if (on_pts or 0) > 0 else "▼" if (on_pts or 0) < 0 else "–"
    on_html = (
        f'<span style="font-family:var(--mono);font-size:12.5px;font-weight:600;color:{on_color};">'
        f'{on_arrow} {on_pts:+.1f} <span style="color:var(--text-dim);font-size:10px;">{(on_pct or 0):+.2f}%</span></span>'
        if on_pts is not None else '<span style="color:var(--text-dim);font-size:11px;">—</span>'
    )

    # Vol budget
    ratio = classification.get("move_ratio")
    if ratio is not None:
        ratio_pct = min(ratio * 100, 100)
        ratio_color = (COLORS["positive"] if ratio < 0.40
                       else COLORS["warning"] if ratio < 0.70 else COLORS["negative"])
        vb_val = f"{ratio_pct:.0f}%"
    else:
        ratio_pct = 0
        ratio_color = COLORS["text_muted"]
        vb_val = "—"

    # Session type
    cls_name = classification.get("classification", "–")
    cls_bias = classification.get("bias", "")
    cls_signal = classification.get("signal_strength", "weak")
    cls_acc = classification.get("bucket_accuracy")
    if cls_signal == "strong":
        cls_color = (COLORS["positive"] if cls_bias in ("range-bound", "mean-revert")
                     else COLORS["negative"] if cls_bias in ("directional", "continued-trend")
                     else COLORS["warning"])
    elif cls_signal == "moderate":
        cls_color = COLORS["warning"]
    else:
        cls_color = COLORS["text_muted"]
    acc_tag = (f'<span style="color:var(--text-dim);font-weight:500;font-family:var(--mono);">hist {cls_acc*100:.0f}%</span>'
               if cls_acc is not None else "")

    # Zero Γ vs EM
    zg_within = level_ctx.get("zero_gamma_within_em")
    zg_dist = level_ctx.get("zero_gamma_distance_to_spot")
    if zg_within is not None:
        zg_icon = "✅ inside" if zg_within else "⚠ outside"
        zg_color = COLORS["positive"] if zg_within else COLORS["warning"]
        zg_dist_html = (f'<span style="color:var(--text-dim);font-weight:500;font-family:var(--mono);">{zg_dist:+.1f} pts</span>'
                        if zg_dist is not None else "")
        zg_row = (
            '<div class="em-row" style="padding-bottom:0;">'
            '<span class="lbl">Zero Γ vs EM</span>'
            f'<span style="font-size:11px;font-weight:700;color:{zg_color};">{zg_icon} {zg_dist_html}</span></div>'
        )
    else:
        zg_row = ""

    lo_txt = fmt_commas(lo, 0) if lo is not None else "—"
    hi_txt = fmt_commas(hi, 0) if hi is not None else "—"

    return f"""
<section class="term-card">
  <div class="card-eyebrow">⚡ Expected Move</div>
  <div class="em-big">
    <span class="em-num">±{em_pts:.0f}</span>
    <span style="font-size:12px;color:var(--text-muted);">pts</span>
    <span style="font-family:var(--mono);font-size:13px;color:var(--text-muted);margin-left:auto;">{em_pct:.2f}%</span>
    <span class="em-badge">{esc(badge)}</span>
  </div>
  <div style="font-size:10px;color:var(--text-dim);margin-bottom:11px;font-family:var(--mono);">ATM straddle @ {esc(straddle.get("strike", "?"))} strike</div>
  <div class="em-rangebar">
    <div class="dot" style="left:0;background:var(--red);"></div>
    <div class="dot" style="right:0;background:var(--green);"></div>
    <div class="dot spot" style="left:{spot_pct:.1f}%;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;margin-bottom:14px;">
    <span style="color:var(--red);">{lo_txt}</span><span style="color:var(--text-dim);">EM RANGE</span><span style="color:var(--green);">{hi_txt}</span>
  </div>
  <div class="em-row"><span class="lbl">{on_label}</span>{on_html}</div>
  <div class="em-row" style="display:block;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
      <span class="lbl">Vol Budget Used</span>
      <span style="font-family:var(--mono);font-size:12px;font-weight:600;color:{ratio_color};">{vb_val}</span></div>
    <div class="em-prog"><span style="width:{ratio_pct:.0f}%;"></span></div>
  </div>
  <div class="em-row"><span class="lbl">Session Type</span>
    <span style="font-size:11px;font-weight:700;color:{cls_color};">{esc(cls_name)} {acc_tag}</span></div>
  {zg_row}
</section>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Wall Credibility
# ─────────────────────────────────────────────────────────────────────────────
def render_wall_credibility(wall_cred) -> str:
    if not wall_cred:
        return ""
    rows = ""
    icon = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}
    for key, label in [("call_wall", "Call Wall"), ("put_wall", "Put Wall"), ("zero_gamma", "Zero Γ")]:
        info = wall_cred.get(key, {})
        if not info:
            continue
        score = info.get("score", 0)
        lbl = info.get("label", "?")
        color = _bar_color_for_score(score)
        rows += (
            '<div class="wc-row">'
            '<div class="wc-head">'
            f'<span style="font-size:11px;color:var(--text-secondary);font-weight:600;">{icon.get(lbl, "⚪")} {esc(label)}</span>'
            f'<span style="font-family:var(--mono);font-size:11px;color:{color};font-weight:600;">{score:.0f}'
            '<span style="color:var(--text-dim);">/100</span></span></div>'
            f'<div class="wc-track"><span style="width:{min(max(score,0),100):.0f}%;background:{color};"></span></div></div>'
        )
    return f"""
<section class="term-card">
  <div class="card-eyebrow">Wall Credibility</div>
  <div style="display:flex;flex-direction:column;gap:11px;">{rows}</div>
</section>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality
# ─────────────────────────────────────────────────────────────────────────────
def render_data_quality(stats, staleness_info) -> str:
    fresh = staleness_info.get("freshness_score", 0)
    fresh_lbl = staleness_info.get("freshness_label", "?")
    fresh_color = _score_color(fresh_lbl)
    coverage = stats.get("coverage_ratio", 0)
    cov_color = (COLORS["positive"] if coverage >= 0.95
                 else COLORS["warning"] if coverage >= 0.85 else COLORS["negative"])

    def cell(value, label, color="var(--text-primary)"):
        return (
            '<div class="dq-cell">'
            f'<div class="dq-val" style="color:{color};">{value}</div>'
            f'<div class="dq-lbl">{esc(label)}</div></div>'
        )

    grid = (
        cell(fmt_commas(stats.get("used_option_count", 0), 0), "USED OPTS")
        + cell(f"{coverage*100:.1f}%", "COVERAGE", cov_color)
        + cell(f"{fresh:.0f}", "FRESH", fresh_color)
        + cell(fmt_commas(stats.get("direct_iv_count", 0), 0), "DIRECT IV")
        + cell(fmt_commas(stats.get("synthetic_iv_count", 0), 0), "SYNTH IV")
        + cell(fmt_commas(stats.get("skipped_count", 0), 0), "SKIPPED", COLORS["warning"])
    )

    # vol amplification note
    note = ""
    vol_ratio = stats.get("vol_amplification_ratio")
    vol_pct = stats.get("vol_dominated_pct", 0.0) or 0.0
    if vol_ratio is not None and vol_ratio > 1.10:
        if vol_ratio >= 1.75:
            va_color, va_word = COLORS["negative"], "heavy"
        elif vol_ratio >= 1.30:
            va_color, va_word = COLORS["warning"], "elevated"
        else:
            va_color, va_word = COLORS["text_secondary"], "mild"
        note = (
            '<div class="dq-note">⚠ Vol amplification '
            f'<span style="color:{va_color};font-weight:600;font-family:var(--mono);">{vol_ratio:.2f}× ({va_word})</span> — '
            f'{vol_pct*100:.0f}% of strikes have today\'s volume &gt; settled OI; wall magnitudes lifted vs OI-only, '
            'locations reliable.</div>'
        )

    return f"""
<section class="term-card">
  <div class="card-eyebrow">Data Quality</div>
  <div class="dq-grid">{grid}</div>
  {note}
</section>
"""
