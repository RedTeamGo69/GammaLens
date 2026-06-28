"""Chart-building helpers extracted from streamlit_app.py.

``render_gex_html`` is the terminal-redesign Strike-GEX view: a pure HTML/CSS
mirrored bar grid + reference overlays + CSS-hover tooltips, generated from the
live ``gex_df``. ``build_gex_bar_chart`` is the original Plotly two-panel chart,
kept (auto-rethemed via theme.COLORS) as a fallback.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import COLORS
from phase1.config import TRADING_HOURS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# HTML/CSS Strike-GEX chart (default for the terminal redesign)
# ─────────────────────────────────────────────────────────────────────────────
def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def render_gex_html(gex_df, levels, spot, em_analysis,
                    weekly_em=None, monthly_em=None, show_daily_em=True,
                    ticker="SPX"):
    """Return the Strike-by-Strike GEX × Charm view as a self-contained HTML
    section (mirrored bars, reference overlays, CSS hover tooltips).

    Geometry mirrors the design prototype: rows are laid top→bottom from the
    highest strike to the lowest, so a price's vertical position is
    ``top% = (maxK − price) / (maxK − minK) × 100``.
    """
    df = gex_df.copy().sort_values("strike", ascending=False).reset_index(drop=True)
    strikes = df["strike"].tolist()
    gex = df["net_gex"].tolist()
    n = len(strikes)
    if n == 0:
        return ('<section class="gex-wrap"><div style="color:var(--text-dim);font-size:12px;">'
                'No GEX data to plot.</div></section>')

    kmax, kmin = max(strikes), min(strikes)
    krange = (kmax - kmin) or 1.0
    maxg = max((abs(x) for x in gex), default=1.0) or 1.0

    has_charm = "net_charm" in df.columns
    charm = []
    maxc = 1.0
    if has_charm:
        for c in df["net_charm"].tolist():
            cv = c / float(TRADING_HOURS_PER_YEAR)
            charm.append(cv if np.isfinite(cv) else 0.0)
        maxc = max((abs(x) for x in charm), default=1.0) or 1.0

    def nearest_idx(v):
        return min(range(n), key=lambda i: abs(strikes[i] - v))

    spot_i = nearest_idx(spot)
    cw_i = nearest_idx(levels["call_wall"])
    pw_i = nearest_idx(levels["put_wall"])
    label_every = max(1, round(n / 20))

    def toppct(price):
        """Map a price to a vertical % using the SAME per-row grid the bars and
        y-axis ticks use (row i, sorted high→low, is centered at (i+0.5)/n).
        Interpolating in this index space — not linear price space — keeps
        reference lines aligned with the strike ticks even when strikes are
        unevenly spaced."""
        if n == 1:
            return 50.0
        if price >= strikes[0]:                       # at/above the top strike
            span = (strikes[0] - strikes[1]) or 1.0
            frac = (strikes[0] - price) / span        # <= 0 → extends upward
            return (0.5 + frac) / n * 100.0
        if price <= strikes[-1]:                      # at/below the bottom strike
            span = (strikes[-2] - strikes[-1]) or 1.0
            frac = (strikes[-2] - price) / span       # >= 1 → extends downward
            return (n - 2 + 0.5 + frac) / n * 100.0
        for i in range(n - 1):                        # bracket and interpolate
            hi_s, lo_s = strikes[i], strikes[i + 1]
            if hi_s >= price >= lo_s:
                span = (hi_s - lo_s) or 1.0
                frac = (hi_s - price) / span
                return (i + 0.5 + frac) / n * 100.0
        return 50.0

    # ── rows ──
    rows = []
    for i, (k, g) in enumerate(zip(strikes, gex)):
        gw = abs(g) / maxg * 100.0
        pos = (f'<div class="gexbar pos" style="width:{gw:.1f}%;background:var(--green);'
               'box-shadow:0 0 4px rgba(43,232,138,.35);"></div>') if g > 0 else ''
        neg = (f'<div class="gexbar neg" style="width:{gw:.1f}%;background:var(--red);'
               'box-shadow:0 0 4px rgba(255,77,104,.35);"></div>') if g < 0 else ''

        charm_cell = ''
        charm_tip = ''
        if has_charm:
            c = charm[i]
            cw_pct = abs(c) / maxc * 100.0
            # Charm convention (inverted vs GEX): supportive (c<0, dealer buys)
            # → LEFT, green;  repelling (c>0, dealer sells) → RIGHT, red.
            ch_g = (f'<div class="gexbar" style="width:{cw_pct:.1f}%;background:var(--green);opacity:.8;"></div>'
                    if c < 0 else '')
            ch_r = (f'<div class="gexbar" style="width:{cw_pct:.1f}%;background:#ff6b81;opacity:.9;"></div>'
                    if c > 0 else '')
            charm_cell = (
                '<div style="width:14px;"></div>'
                '<div class="gex-cell" style="flex:1;">'
                '<div class="gex-mid" style="background:#3a2e1c;"></div>'
                f'<div class="gex-half neg">{ch_g}</div><div class="gex-half pos">{ch_r}</div></div>'
            )
            csign = "+" if c >= 0 else "−"
            ccol = "var(--green)" if c < 0 else "var(--red)"
            charm_tip = (
                '<div style="display:flex;justify-content:space-between;gap:16px;">'
                '<span style="color:var(--text-dim);">Charm/hr</span>'
                f'<span style="font-weight:600;color:{ccol};">{csign}{abs(c):,.0f}/hr</span></div>'
            )

        gcol = "var(--green)" if g >= 0 else "var(--red)"
        gsign = "+" if g >= 0 else "−"
        reg_tag = "POSITIVE Γ" if g >= 0 else "NEGATIVE Γ"
        tip = (
            '<div class="gex-tip">'
            f'<div style="font-size:12px;font-weight:700;margin-bottom:3px;">{ticker} '
            f'<span style="color:#fff;">{k:,.0f}</span></div>'
            '<div style="display:flex;justify-content:space-between;gap:16px;">'
            '<span style="color:var(--text-dim);">Net GEX</span>'
            f'<span style="font-weight:600;color:{gcol};">{gsign}{abs(g):,.0f}</span></div>'
            f'{charm_tip}'
            '<div style="margin-top:4px;padding-top:4px;border-top:1px solid var(--border);'
            f'font-size:9.5px;font-weight:700;letter-spacing:.04em;color:{gcol};">{reg_tag}</div></div>'
        )
        rows.append(
            '<div class="gexrow"><div class="gex-cell"><div class="gex-mid"></div>'
            f'<div class="gex-half neg">{neg}</div><div class="gex-half pos">{pos}</div></div>'
            f'{charm_cell}{tip}</div>'
        )
    rows_html = "".join(rows)

    # ── y-axis labels ──
    yticks = []
    for i, k in enumerate(strikes):
        show = (i % label_every == 0) or i in (spot_i, cw_i, pw_i)
        if i == spot_i:
            col, wt = "#fff", "600"
        elif i == cw_i:
            col, wt = "var(--green)", "600"
        elif i == pw_i:
            col, wt = "var(--red)", "600"
        else:
            col, wt = "var(--text-dim)", "400"
        txt = f"{k:,.0f}" if show else ""
        yticks.append(f'<div class="gex-ytick" style="font-size:9.5px;font-weight:{wt};color:{col};">{txt}</div>')
    yticks_html = "".join(yticks)

    # ── overlays (reference lines + EM bands) ──
    # Lines are drawn at their TRUE row position; labels are collected and then
    # de-collided per side so near-equal prices (e.g. Spot 7,354 vs Zero Γ 7,357)
    # don't stack on top of each other.
    ov = []          # lines + band fills
    labels = []      # {top, side, text, color, bg, border}

    def refline(price, color, dash, label, side):
        if not (kmin <= price <= kmax):
            return
        top = toppct(price)
        ov.append(f'<div class="gex-refline" style="top:{top:.2f}%;border-top:1.5px {dash} {color};opacity:.85;"></div>')
        if color in ("#fff", "#ffffff"):
            labels.append(dict(top=top, side=side, text=f"{label} {price:,.0f}",
                               color="#0a0d13", bg="#fff", border=""))
        else:
            labels.append(dict(top=top, side=side, text=f"{label} {price:,.0f}",
                               color=color, bg="var(--bg-input)", border=f"border:1px solid {color};"))

    def band(lo, hi, fill, border, lbl_color, lbl_lo, lbl_hi):
        if lo is None or hi is None or not lo or not hi:
            return
        t_hi = _clamp(toppct(hi))
        t_lo = _clamp(toppct(lo))
        top = min(t_hi, t_lo)
        height = abs(t_lo - t_hi)
        if height <= 0:
            return
        ov.append(f'<div style="position:absolute;left:0;right:0;top:{top:.2f}%;height:{height:.2f}%;'
                  f'background:{fill};{border}"></div>')
        for price, lab in ((hi, lbl_hi), (lo, lbl_lo)):
            if kmin <= price <= kmax:
                labels.append(dict(top=toppct(price), side="right", text=f"{lab} {price:,.0f}",
                                   color=lbl_color, bg="var(--bg-input)", border=f"border:1px solid {lbl_color};"))

    m_em = monthly_em or {}
    band(m_em.get("lower_level"), m_em.get("upper_level"),
         "rgba(110,168,255,.045)",
         "border-top:1px solid rgba(110,168,255,.3);border-bottom:1px solid rgba(110,168,255,.3);",
         "#6ea8ff", "OpEx−", "OpEx+")
    w_em = weekly_em or {}
    band(w_em.get("lower_level"), w_em.get("upper_level"),
         "rgba(245,197,66,.05)",
         "border-top:1px dashed rgba(245,197,66,.4);border-bottom:1px dashed rgba(245,197,66,.4);",
         "#f5c542", "wEM−", "wEM+")
    if show_daily_em:
        d_em = em_analysis.get("expected_move", {}) or {}
        band(d_em.get("lower_level"), d_em.get("upper_level"),
             "rgba(169,139,255,.12)",
             "border-top:1px solid rgba(169,139,255,.35);border-bottom:1px solid rgba(169,139,255,.35);",
             "#a98bff", "EM−", "EM+")

    refline(levels["call_wall"], COLORS["call_wall"], "dashed", "CALL WALL", "left")
    refline(spot, "#fff", "dashed", "SPOT", "left")
    refline(levels["zero_gamma"], COLORS["zero_gamma"], "dotted", "ZERO Γ", "left")
    refline(levels["put_wall"], COLORS["put_wall"], "dashed", "PUT WALL", "left")

    # De-collide labels per side. Rows are a fixed 6px (see CSS); a reference
    # label renders ~17px tall, so reserve ~20px (≈3.3 rows) of vertical space,
    # expressed as a % of the n-row (n×6px) chart height.
    min_gap = (20.0 / 6.0) / n * 100.0

    def _decollide(side):
        sub = sorted([L for L in labels if L["side"] == side], key=lambda d: d["top"])
        for L in sub:
            L["adj"] = L["top"]
        for i in range(1, len(sub)):
            if sub[i]["adj"] - sub[i - 1]["adj"] < min_gap:
                sub[i]["adj"] = sub[i - 1]["adj"] + min_gap
        if sub:  # if the cluster overflowed the bottom, slide it back up
            over = sub[-1]["adj"] - 99.0
            if over > 0:
                for L in sub:
                    L["adj"] = max(0.5, L["adj"] - over)
        return sub

    for L in _decollide("left") + _decollide("right"):
        ov.append(f'<div class="gex-reflabel" style="{L["side"]}:4px;top:{L["adj"]:.2f}%;'
                  f'color:{L["color"]};background:{L["bg"]};{L["border"]}">{L["text"]}</div>')
    overlay_html = "".join(ov)

    # ── legend ──
    legend = (
        '<span><span style="width:18px;height:7px;border-radius:2px;background:var(--green);"></span>Pos GEX</span>'
        '<span><span style="width:18px;height:7px;border-radius:2px;background:var(--red);"></span>Neg GEX</span>'
        '<span><span style="width:14px;border-top:2px dashed #fff;"></span>Spot</span>'
        '<span><span style="width:14px;border-top:2px dotted var(--cyan);"></span>Zero Γ</span>'
        '<span><span style="width:14px;height:9px;border-radius:2px;background:rgba(169,139,255,.25);"></span>Daily EM</span>'
        '<span><span style="width:14px;height:9px;border-radius:2px;background:rgba(245,197,66,.20);border:1px dashed rgba(245,197,66,.55);"></span>Weekly EM</span>'
        '<span><span style="width:14px;height:9px;border-radius:2px;background:rgba(110,168,255,.18);border:1px solid rgba(110,168,255,.5);"></span>OpEx EM</span>'
    )

    charm_hdr = ('<div style="width:14px;"></div>'
                 '<div style="flex:1;text-align:center;font-size:9.5px;letter-spacing:.08em;color:var(--amber);font-weight:700;">◀ SUPPORT · REPEL ▶</div>'
                 ) if has_charm else ''
    charm_note = (
        '<div class="gex-note">'
        '<span style="font-weight:700;color:var(--amber);letter-spacing:.04em;">CHARM/HR</span>'
        '<span><span style="width:14px;height:7px;border-radius:2px;background:var(--green);"></span>'
        '<b style="color:var(--text-secondary);">left = supportive</b> — dealer buys, pins price</span>'
        '<span><span style="width:14px;height:7px;border-radius:2px;background:#ff6b81;"></span>'
        '<b style="color:var(--text-secondary);">right = repelling</b> — dealer sells into decay</span>'
        '<span style="color:var(--text-dim);">Color logic is inverted vs GEX — read by side of zero, not sign.</span>'
        '</div>'
    ) if has_charm else ''

    return f"""
<section class="gex-wrap">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
    <div>
      <div class="gex-title">Strike-by-Strike Net GEX <span style="color:var(--text-dim);font-weight:500;">×</span> Charm/hr</div>
      <div class="gex-sub">Dealer gamma proxy per strike · charm = Θ-decay hedging drift</div>
    </div>
    <div class="gex-legend">{legend}</div>
  </div>
  <div style="display:flex;align-items:flex-end;gap:0;padding-left:50px;margin-bottom:6px;">
    <div style="flex:3;text-align:center;font-size:9.5px;letter-spacing:.1em;color:var(--text-dim);font-weight:700;">◀ NEGATIVE&nbsp;&nbsp;·&nbsp;&nbsp;NET&nbsp;GEX&nbsp;PROXY&nbsp;&nbsp;·&nbsp;&nbsp;POSITIVE ▶</div>
    {charm_hdr}
  </div>
  <div class="gex-plot">
    <div class="gex-yaxis">{yticks_html}</div>
    <div class="gex-area" style="overflow:visible;height:{n * 6}px;">
      <div class="gex-rows">{rows_html}</div>
      <div class="gex-overlay">{overlay_html}</div>
    </div>
  </div>
  {charm_note}
</section>
"""


def build_gex_bar_chart(gex_df, levels, spot, em_analysis,
                         weekly_em=None, monthly_em=None, show_daily_em=True):
    """
    Two-panel GEX + Charm chart.

    Left panel (~78% width):  strike-by-strike net GEX horizontal bars
    Right panel (~22% width): per-strike charm ($Δ drift per trading
                              hour) as a line with sign-coded fills

    Both panels share the strike Y axis, so the reader can scan
    horizontally across any strike level and read GEX (left) and charm
    (right) with a single visual sweep. Each panel owns its own X axis
    and zero, which fixes the zero-misalignment that plagued the earlier
    single-chart dual-X-axis overlay — the two zeros no longer need to
    be geometrically aligned because they're in separate plots.

    Reference lines (spot, zero-gamma, walls, EM levels) are duplicated
    across both panels so the level context carries through; annotation
    labels are only placed on the left panel to avoid double-labeling.
    """
    df = gex_df.copy().sort_values("strike").reset_index(drop=True)
    strikes = df["strike"].values
    net_gex = df["net_gex"].values
    colors = [COLORS["bar_green"] if g >= 0 else COLORS["bar_red"] for g in net_gex]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        column_widths=[0.78, 0.22],
    )

    # ── LEFT PANEL: GEX bars ──────────────────────────────────────────
    fig.add_trace(go.Bar(
        y=strikes, x=net_gex, orientation="h",
        marker_color=colors, marker_opacity=0.85,
        hovertemplate="Strike: $%{y:.0f}<br>Net GEX: %{x:,.0f}<extra></extra>",
    ), row=1, col=1)

    # ── RIGHT PANEL: Charm/hr with sign-coded fills ───────────────────
    # CEX sign under SqueezeMetrics convention:
    #   CEX > 0 → dealer book GAINS delta → dealer SELLS → repelling (red)
    #   CEX < 0 → dealer book LOSES delta → dealer BUYS  → supportive (green)
    # No competing bars on this side so fill opacity is bumped up from
    # the old overlay (0.18 → 0.30) for a crisper direction read.
    charm_range = None
    if "net_charm" in df.columns and len(df) > 0:
        charm_per_hr = df["net_charm"].values / float(TRADING_HOURS_PER_YEAR)
        charm_per_hr = np.where(np.isfinite(charm_per_hr), charm_per_hr, 0.0)

        max_abs_charm = float(np.max(np.abs(charm_per_hr)))
        if max_abs_charm > 0:
            charm_range = [-max_abs_charm * 1.05, max_abs_charm * 1.05]

        supportive = np.minimum(charm_per_hr, 0.0)
        repelling  = np.maximum(charm_per_hr, 0.0)

        fig.add_trace(go.Scatter(
            y=strikes, x=supportive, mode="lines",
            line=dict(color=COLORS["bar_green"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(43, 232, 138, 0.30)",
            name="Supportive",
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            y=strikes, x=repelling, mode="lines",
            line=dict(color=COLORS["bar_red"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(255, 77, 104, 0.30)",
            name="Repelling",
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=2)

        direction_labels = np.where(
            charm_per_hr > 0, "Repelling (dealer sells)",
            np.where(charm_per_hr < 0, "Supportive (dealer buys)", "Neutral"),
        )
        fig.add_trace(go.Scatter(
            y=strikes, x=charm_per_hr, mode="lines",
            line=dict(color=COLORS["charm_line"], width=1.5),
            name="Charm/hr",
            customdata=direction_labels,
            hovertemplate=(
                "<b>Strike: $%{y:.0f}</b>"
                "<br>Charm/hr: %{x:,.0f} $Δ"
                "<br><b>%{customdata}</b>"
                "<br><i>(left = supportive, right = repelling)</i>"
                "<extra></extra>"
            ),
            opacity=0.95,
        ), row=1, col=2)

    # ── Reference lines (duplicated across both panels) ───────────────
    # Helper: draw a horizontal line on both panels, with the label
    # annotation only on the left (col=1) to avoid double-labeling.
    def _span_hline(val, color, dash, width, label=None, font_size=9,
                    position="top left"):
        fig.add_hline(
            y=val, line_color=color, line_dash=dash, line_width=width,
            annotation_text=(f"{label} ${val:.0f}" if label else None),
            annotation_font_color=color, annotation_font_size=font_size,
            annotation_position=position,
            row=1, col=1,
        )
        fig.add_hline(
            y=val, line_color=color, line_dash=dash, line_width=width,
            row=1, col=2,
        )

    def _span_hrect(y0, y1, fillcolor, opacity):
        for col in (1, 2):
            fig.add_hrect(
                y0=y0, y1=y1,
                fillcolor=fillcolor, opacity=opacity,
                line_width=0, layer="below",
                row=1, col=col,
            )

    for val, color, dash, name in [
        (spot, COLORS["spot"], "dash", "Spot"),
        (levels["zero_gamma"], COLORS["zero_gamma"], "dot", "Zero Γ"),
        (levels["call_wall"], COLORS["call_wall"], "dashdot", "Call Wall"),
        (levels["put_wall"], COLORS["put_wall"], "dashdot", "Put Wall"),
    ]:
        _span_hline(val, color, dash, width=1.5, label=name, font_size=9,
                    position="top left")

    em = em_analysis.get("expected_move", {})
    if show_daily_em and em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            _span_hline(val, COLORS["em_level"], "dot", width=1.2,
                        label=label, font_size=8, position="bottom right")

    w_em = weekly_em or {}
    if w_em.get("upper_level") and w_em.get("lower_level"):
        _span_hrect(w_em["lower_level"], w_em["upper_level"],
                    COLORS["em_weekly"], 0.06)
        for val, label in [(w_em["upper_level"], "wEM+"), (w_em["lower_level"], "wEM−")]:
            _span_hline(val, COLORS["em_weekly"], "dash", width=1,
                        label=label, font_size=7, position="top right")

    m_em = monthly_em or {}
    if m_em.get("upper_level") and m_em.get("lower_level"):
        _span_hrect(m_em["lower_level"], m_em["upper_level"],
                    COLORS["em_monthly"], 0.04)
        for val, label in [(m_em["upper_level"], "OpEx+"), (m_em["lower_level"], "OpEx−")]:
            _span_hline(val, COLORS["em_monthly"], "longdash", width=1,
                        label=label, font_size=7, position="top right")

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=55, b=35),
        title="Strike-by-Strike Net GEX Proxy  +  Charm/hr",
        showlegend=False, height=2000, dragmode=False,
        hovermode="y unified",
        spikedistance=-1,
        hoverdistance=100,
    )

    # Left panel X axis (GEX)
    fig.update_xaxes(
        title_text="Net GEX proxy",
        gridcolor=COLORS["grid_major"],
        zerolinecolor=COLORS["zeroline"],
        row=1, col=1,
    )

    # Right panel X axis (Charm/hr) — amber-themed, symmetric around zero
    fig.update_xaxes(
        title=dict(
            text="Charm/hr — ← supportive | repelling →",
            font=dict(color=COLORS["charm_line"], size=9),
        ),
        showgrid=False,
        range=charm_range,
        zeroline=True,
        zerolinecolor=COLORS["charm_line"],
        zerolinewidth=1.2,
        tickfont=dict(color=COLORS["charm_line"], size=8),
        row=1, col=2,
    )

    # Shared Y axis (strike)
    fig.update_yaxes(
        title_text="Strike",
        gridcolor=COLORS["grid_minor"],
        tickfont_size=8,
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=COLORS["text_white"],
        spikethickness=1,
        row=1, col=1,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor=COLORS["text_white"],
        spikethickness=1,
        row=1, col=2,
    )

    return fig
