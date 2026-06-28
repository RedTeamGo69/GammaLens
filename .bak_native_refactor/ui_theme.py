"""
Terminal-aesthetic UI foundation for the GEX dashboard redesign.

Holds the global CSS (design tokens, Streamlit chrome overrides, component
classes), the query-param-driven UI-state helpers that let custom HTML
controls stay interactive (see memory: streamlit-custom-html-interactivity),
and the sticky header renderer.

NOTHING here touches data/model logic — it is pure presentation + URL state.
"""
from __future__ import annotations

import html as _html
from typing import Any

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens (mirror of theme.COLORS, kept here as CSS variables)
# ─────────────────────────────────────────────────────────────────────────────
TOKENS = {
    "bg_base": "#0a0d13",
    "bg_surface": "#11151c",
    "bg_input": "#0c1119",
    "bg_row": "#141922",
    "border": "#1b212a",
    "border_mid": "#20272f",
    "green": "#2be88a",
    "red": "#ff4d68",
    "cyan": "#25d8ef",
    "amber": "#ffb454",
    "purple": "#a98bff",
    "blue": "#6ea8ff",
    "yellow": "#f5c542",
    "text_primary": "#e7edf5",
    "text_secondary": "#cbd5e1",
    "text_muted": "#93a1b2",
    "text_dim": "#5b6878",
}

# Default UI state (README "State Management" table). Param values are the
# short tokens stored in the URL; map_* helpers translate to/from them.
_DEFAULTS = {
    "ticker": "SPX",
    "exp": "0dte",      # 0dte | tomorrow | week | opex | custom
    "tab": "gex",       # gex | spread | 0dte
    "refresh": "off",   # off | 5min | 30min
    "tier": "2",
    "cal_start": "",
    "cal_end": "",
    "cal_offset": "0",
    "recents": "",      # comma-separated recently-viewed tickers (URL-carried so
                        # they survive the full reload each anchor click triggers)
}

# Transient action params that must never be carried forward by qlink().
_TRANSIENT = {"refresh_now"}

# Curated quick-pick tickers (mirror phase1.ticker_config.all_tickers ordering).
QUICK_TICKERS = ["SPX", "XSP", "QQQ", "AMZN", "AMD"]

EXP_MODES = [
    ("0dte", "0DTE"),
    ("tomorrow", "Tomorrow"),
    ("week", "This week"),
    ("opex", "OpEx"),
    ("custom", "Custom"),
]
REFRESH_MODES = [("off", "Off"), ("5min", "5 min"), ("30min", "30 min")]
TABS = [("gex", "📊 Strike GEX"), ("spread", "🎯 Spread Finder"), ("0dte", "⚡ 0DTE Finder")]


# ─────────────────────────────────────────────────────────────────────────────
# Query-param state
# ─────────────────────────────────────────────────────────────────────────────
def get_ui_state() -> dict[str, str]:
    """Read the current UI state from the URL query params, applying defaults.

    The URL is the single source of truth for navigable state so that the
    custom-HTML anchor/form controls drive the app (spike-verified pattern).
    """
    qp = st.query_params
    state: dict[str, str] = {}
    for key, default in _DEFAULTS.items():
        val = qp.get(key, default)
        state[key] = (val or default).strip()
    return state


def qlink(state: dict[str, str] | None = None, **overrides: Any) -> str:
    """Build an href that preserves current params and applies overrides.

    Pass ``key=None`` to drop a param. Transient action params are never
    carried forward.
    """
    base = dict(state) if state is not None else dict(st.query_params)
    for k in _TRANSIENT:
        base.pop(k, None)
    for k, v in overrides.items():
        if v is None:
            base.pop(k, None)
        else:
            base[k] = str(v)
    # Drop params equal to their default to keep URLs short/clean.
    parts = []
    for k, v in base.items():
        if k in _DEFAULTS and str(v) == _DEFAULTS[k]:
            continue
        if v == "" or v is None:
            continue
        parts.append(f"{_html.escape(str(k))}={_html.escape(str(v))}")
    return "?" + "&".join(parts) if parts else "?"


def map_exp_mode(exp_token: str) -> str:
    """Translate the URL exp token to the internal mode string main() expects."""
    return {
        "0dte": "0DTE",
        "tomorrow": "Tomorrow",
        "week": "This week",
        "opex": "OpEx Cycle",
        "custom": "Custom",
    }.get(exp_token, "0DTE")


def map_refresh(token: str) -> str:
    return {"off": "Off", "5min": "Every 5 min", "30min": "Every 30 min"}.get(token, "Off")


# ─────────────────────────────────────────────────────────────────────────────
# Small formatting helpers (presentation only)
# ─────────────────────────────────────────────────────────────────────────────
def fmt_commas(n: float | int | None, decimals: int = 0) -> str:
    if n is None:
        return "—"
    try:
        return f"{float(n):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def esc(s: Any) -> str:
    return _html.escape(str(s))


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_global_css() -> None:
    """Inject the full terminal stylesheet. Call once at the top of the app."""
    t = TOKENS
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
  --bg-base:{t['bg_base']}; --bg-surface:{t['bg_surface']}; --bg-input:{t['bg_input']};
  --bg-row:{t['bg_row']}; --border:{t['border']}; --border-mid:{t['border_mid']};
  --green:{t['green']}; --red:{t['red']}; --cyan:{t['cyan']}; --amber:{t['amber']};
  --purple:{t['purple']}; --blue:{t['blue']}; --yellow:{t['yellow']};
  --text-primary:{t['text_primary']}; --text-secondary:{t['text_secondary']};
  --text-muted:{t['text_muted']}; --text-dim:{t['text_dim']};
  --mono:'IBM Plex Mono',ui-monospace,monospace; --sans:'IBM Plex Sans',system-ui,sans-serif;
}}

/* ── Streamlit chrome overrides ── */
.stApp {{ background:var(--bg-base); }}
header[data-testid="stHeader"] {{ display:none; }}
[data-testid="stToolbar"] {{ display:none; }}
#MainMenu, footer {{ display:none; }}
section[data-testid="stSidebar"] {{ display:none; }}
[data-testid="stSidebarCollapsedControl"] {{ display:none; }}
[data-testid="stAppViewBlockContainer"], .block-container {{
  padding:0 16px 16px !important; max-width:100% !important;
}}
[data-testid="stMainBlockContainer"] {{ padding-top:0 !important; max-width:100% !important; }}
.stApp, body, [data-testid="stMarkdownContainer"] {{
  font-family:var(--sans); color:var(--text-primary); -webkit-font-smoothing:antialiased;
}}
/* Tighten gap between stacked Streamlit blocks so our HTML reads as one canvas */
[data-testid="stVerticalBlock"] {{ gap:0.6rem; }}
::-webkit-scrollbar {{ width:9px; height:9px; }}
::-webkit-scrollbar-track {{ background:var(--bg-base); }}
::-webkit-scrollbar-thumb {{ background:#232c38; border-radius:5px; }}
::-webkit-scrollbar-thumb:hover {{ background:#2e3848; }}
@keyframes livepulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:.2;}} }}

/* ── Header ── */
.term-header {{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
  padding:12px 20px; border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,#0c1017,var(--bg-base));
  position:sticky; top:0; z-index:30; margin-bottom:14px;
}}
.term-logo {{
  width:30px; height:30px; border-radius:7px; background:linear-gradient(135deg,var(--green),#0f9b62);
  display:flex; align-items:center; justify-content:center; font-weight:700; color:#06140c;
  font-size:15px; font-family:var(--mono); box-shadow:0 0 14px rgba(43,232,138,.35);
}}
.term-live-dot {{
  width:7px; height:7px; border-radius:50%; background:var(--green);
  box-shadow:0 0 8px var(--green); animation:livepulse 1.8s ease-in-out infinite; display:inline-block;
}}
.hdr-refresh {{
  font-family:var(--sans); font-size:11.5px; font-weight:600; color:#0a0d13; background:var(--green);
  border:none; padding:7px 13px; border-radius:7px; cursor:pointer; text-decoration:none;
  display:inline-flex; align-items:center; gap:6px; transition:all .12s;
}}
.hdr-refresh:hover {{ background:#3df59a; box-shadow:0 0 12px rgba(43,232,138,.5); }}

/* ── Body layout (aside + main) ── */
.term-body {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:16px; }}
.term-aside {{ flex:1 1 300px; min-width:280px; display:flex; flex-direction:column; gap:14px; }}
.term-main {{ flex:999 1 480px; min-width:320px; display:flex; flex-direction:column; gap:14px; }}

/* ── Cards ── */
.term-card {{ background:var(--bg-surface); border:1px solid var(--border); border-radius:11px; padding:14px; }}
.term-card.hero {{ border-color:#243042; box-shadow:inset 0 0 0 1px rgba(43,232,138,.04); }}
.card-eyebrow {{
  font-size:10px; letter-spacing:.14em; color:var(--text-dim); font-weight:700;
  text-transform:uppercase; margin-bottom:10px;
}}
.card-eyebrow.lit {{ color:#9aa7b8; }}

/* ── Chips / segments / tabs (anchors) ── */
.chip, .seg, .tab, .tier-btn, .act-btn {{ text-decoration:none; cursor:pointer; transition:all .12s; }}
.chip {{
  font-family:var(--mono); font-size:11px; font-weight:600; letter-spacing:.02em;
  padding:6px 11px; border-radius:7px; background:var(--bg-row); color:var(--text-muted);
  border:1px solid var(--border-mid); display:inline-block;
}}
.chip.on {{ background:rgba(43,232,138,.14); color:var(--green); border-color:rgba(43,232,138,.45); }}
.chip-wrap {{ display:flex; flex-wrap:wrap; gap:6px; }}
.seg {{
  font-family:var(--sans); font-size:11px; font-weight:600; padding:6px 12px; border-radius:7px;
  flex:1; text-align:center; background:var(--bg-row); color:var(--text-muted);
  border:1px solid var(--border-mid); display:block;
}}
.seg.on {{ background:rgba(43,232,138,.14); color:var(--green); border-color:rgba(43,232,138,.45); }}
.seg-wrap {{ display:flex; gap:6px; }}
.tab-bar {{ display:flex; gap:4px; border-bottom:1px solid var(--border); }}
.tab {{
  font-family:var(--sans); font-size:12.5px; font-weight:600; padding:9px 16px; background:transparent;
  border:none; border-bottom:2px solid transparent; color:var(--text-dim); margin-bottom:-1px; display:inline-block;
}}
.tab.on {{ border-bottom:2px solid var(--green); color:var(--text-primary); }}

/* ── Search input form ── */
.search-form {{ position:relative; margin-bottom:11px; }}
.search-form .ico {{ position:absolute; left:11px; top:50%; transform:translateY(-50%); font-size:14px; color:var(--text-dim); pointer-events:none; }}
.search-form input {{
  width:100%; background:var(--bg-input); border:1px solid var(--border-mid); border-radius:8px;
  padding:9px 11px 9px 31px; color:var(--text-primary); font-family:var(--mono); font-size:11.5px;
  letter-spacing:.01em; outline:none; transition:all .12s;
}}
.search-form input:focus {{ border-color:var(--green); box-shadow:0 0 0 2px rgba(43,232,138,.12); }}

/* ── Key levels grid ── */
.lvl-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.lvl-cell {{ background:var(--bg-input); border:1px solid var(--border); border-radius:9px; padding:11px 12px; }}
.lvl-head {{ display:flex; align-items:center; gap:6px; margin-bottom:5px; }}
.lvl-dot {{ width:7px; height:7px; border-radius:2px; display:inline-block; }}
.lvl-lbl {{ font-size:10px; color:var(--text-muted); font-weight:600; }}
.lvl-val {{ font-family:var(--mono); font-size:20px; font-weight:600; }}
.lvl-pillrow {{ display:flex; gap:8px; margin-top:8px; }}
.lvl-pill {{ background:var(--bg-input); border:1px solid var(--border); border-radius:9px; padding:9px 12px; display:flex; align-items:center; justify-content:space-between; }}

/* ── GEX stream grid ── */
.stream-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border); border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
.stream-cell {{ background:var(--bg-input); padding:9px 11px; }}
.stream-cell.span {{ grid-column:1/-1; display:flex; align-items:center; justify-content:space-between; }}
.stream-lbl {{ font-size:9.5px; color:var(--text-dim); font-weight:600; margin-bottom:3px; }}
.stream-val {{ font-family:var(--mono); font-size:14px; font-weight:600; }}
.stream-sub {{ font-size:9px; color:var(--text-dim); }}

/* ── Expected move ── */
.em-big {{ display:flex; align-items:baseline; gap:10px; margin-bottom:4px; }}
.em-num {{ font-family:var(--mono); font-size:28px; font-weight:600; color:var(--purple); }}
.em-badge {{ font-size:9px; font-weight:700; color:var(--purple); background:rgba(169,139,255,.13); border:1px solid rgba(169,139,255,.3); padding:3px 7px; border-radius:5px; }}
.em-rangebar {{ position:relative; height:4px; background:var(--border); border-radius:3px; margin:0 2px 6px; }}
.em-rangebar .dot {{ position:absolute; top:50%; transform:translateY(-50%); width:8px; height:8px; border-radius:50%; }}
.em-rangebar .spot {{ width:11px; height:11px; border:2px solid var(--bg-surface); background:#fff; box-shadow:0 0 6px rgba(255,255,255,.5); transform:translate(-50%,-50%); }}
.em-row {{ display:flex; align-items:center; justify-content:space-between; padding:7px 0; border-top:1px solid var(--border); }}
.em-row .lbl {{ font-size:11px; color:var(--text-muted); }}
.em-prog {{ height:4px; background:var(--border); border-radius:3px; overflow:hidden; }}
.em-prog > span {{ display:block; height:100%; background:linear-gradient(90deg,var(--green),var(--amber)); border-radius:3px; }}

/* ── Wall credibility ── */
.wc-row {{ margin-bottom:0; }}
.wc-head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
.wc-track {{ height:4px; background:var(--border); border-radius:3px; overflow:hidden; }}
.wc-track > span {{ display:block; height:100%; border-radius:3px; }}

/* ── Data quality ── */
.dq-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }}
.dq-cell {{ text-align:center; }}
.dq-val {{ font-family:var(--mono); font-size:15px; font-weight:600; }}
.dq-lbl {{ font-size:9px; color:var(--text-dim); font-weight:600; margin-top:2px; }}
.dq-note {{ font-size:10px; color:var(--text-dim); margin-top:11px; line-height:1.5; border-top:1px solid var(--border); padding-top:9px; }}

/* ── EM strip ── */
.em-strip {{ display:flex; flex-wrap:wrap; gap:1px; background:var(--border); border:1px solid var(--border); border-radius:11px; overflow:hidden; }}
.em-strip .cell {{ flex:1 1 120px; background:var(--bg-surface); padding:11px 15px; }}
.em-strip .cl {{ font-size:9.5px; color:var(--text-dim); font-weight:700; letter-spacing:.06em; margin-bottom:4px; }}
.em-strip .cv {{ font-family:var(--mono); font-size:17px; font-weight:600; }}

/* ── GEX html chart ── */
.gex-wrap {{ background:var(--bg-surface); border:1px solid var(--border); border-radius:11px; padding:16px 16px 12px; flex:1; display:flex; flex-direction:column; }}
.gex-title {{ font-size:14px; font-weight:700; }}
.gex-sub {{ font-size:10.5px; color:var(--text-dim); margin-top:3px; }}
.gex-legend {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; font-size:10px; color:var(--text-muted); }}
.gex-legend span {{ display:flex; align-items:center; gap:5px; }}
.gex-plot {{ display:flex; }}
.gex-yaxis {{ width:50px; display:flex; flex-direction:column; padding-top:1px; }}
.gex-ytick {{ height:6px; flex:none; display:flex; align-items:center; justify-content:flex-end; padding-right:9px; font-family:var(--mono); }}
.gex-area {{ flex:1; position:relative; }}
.gex-rows {{ position:absolute; inset:0; display:flex; flex-direction:column; }}
.gexrow {{ height:6px; flex:none; display:flex; align-items:center; background:transparent; cursor:crosshair; position:relative; }}
.gexrow:hover {{ background:rgba(255,255,255,.055); }}
.gex-cell {{ flex:3; position:relative; height:100%; display:flex; align-items:center; }}
.gex-mid {{ position:absolute; left:50%; top:0; bottom:0; width:1px; background:#2a3340; }}
.gex-half {{ flex:1; display:flex; height:100%; align-items:center; }}
.gex-half.neg {{ justify-content:flex-end; }}
.gex-half.pos {{ justify-content:flex-start; }}
.gexbar {{ height:3px; border-radius:1px; }}
.gexrow:hover .gexbar.pos {{ box-shadow:0 0 8px rgba(43,232,138,.7); }}
.gexrow:hover .gexbar.neg {{ box-shadow:0 0 8px rgba(255,77,104,.7); }}
.gex-tip {{
  position:absolute; right:4px; top:50%; transform:translateY(-50%); background:#0e1620;
  border:1px solid #2a3748; border-radius:8px; padding:8px 11px; font-family:var(--mono);
  font-size:10.5px; line-height:1.7; z-index:10; pointer-events:none; box-shadow:0 4px 16px rgba(0,0,0,.6);
  min-width:160px; opacity:0; visibility:hidden;
}}
.gexrow:hover .gex-tip {{ opacity:1; visibility:visible; }}
.gex-overlay {{ position:absolute; inset:0; pointer-events:none; }}
.gex-refline {{ position:absolute; left:0; right:0; }}
.gex-reflabel {{ position:absolute; transform:translateY(-50%); font-family:var(--mono); font-size:9px; font-weight:600; padding:1px 5px; border-radius:3px; white-space:nowrap; }}
.gex-note {{ margin-top:12px; padding:9px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:8px; display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:10.5px; color:var(--text-muted); }}

/* details/summary reading guide */
details.term-details {{ background:var(--bg-input); border:1px solid var(--border); border-radius:9px; padding:0 12px; }}
details.term-details > summary {{ cursor:pointer; padding:10px 0; font-size:11px; color:var(--text-secondary); font-weight:600; list-style:none; }}
details.term-details > summary::-webkit-details-marker {{ display:none; }}
details.term-details[open] > summary {{ border-bottom:1px solid var(--border); margin-bottom:8px; }}

/* generic banners */
.term-banner {{ border-radius:0 8px 8px 0; padding:10px 14px; font-size:11.5px; line-height:1.5; margin-bottom:8px; }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
def render_header(
    *,
    ticker: str,
    spot: float,
    day_change_pts: float | None,
    day_change_pct: float | None,
    regime_label: str,
    regime_color: str,
    regime_note: str,
    clock: str,
    refresh_href: str,
    live: bool = True,
) -> str:
    """Return the sticky header HTML (left logo, center spot, right status)."""
    chg_color = TOKENS["green"] if (day_change_pts or 0) >= 0 else TOKENS["red"]
    arrow = "▲" if (day_change_pts or 0) >= 0 else "▼"
    chg_txt = f"{arrow} {day_change_pts:+.1f}" if day_change_pts is not None else "—"
    # badge tint follows the regime color (green / red / amber)
    if regime_color == TOKENS["red"]:
        badge_bg, badge_bd = "rgba(255,77,104,.12)", "rgba(255,77,104,.32)"
    elif regime_color == TOKENS["amber"]:
        badge_bg, badge_bd = "rgba(255,180,84,.12)", "rgba(255,180,84,.32)"
    else:
        badge_bg, badge_bd = "rgba(43,232,138,.12)", "rgba(43,232,138,.32)"
    live_html = (
        f'<span style="display:flex;align-items:center;gap:6px;font-size:10px;font-weight:700;'
        f'letter-spacing:.12em;color:var(--text-dim);"><span class="term-live-dot"></span>LIVE</span>'
        if live else ""
    )
    return f"""
<div class="term-header">
  <div style="display:flex;align-items:center;gap:11px;">
    <div class="term-logo">Γ</div>
    <div style="display:flex;flex-direction:column;line-height:1.15;">
      <span style="font-size:13px;font-weight:700;letter-spacing:.02em;">GAMMA EXPOSURE</span>
      <span style="font-size:9.5px;letter-spacing:.26em;color:var(--text-dim);font-weight:600;">TERMINAL · v5</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
    <div style="display:flex;align-items:baseline;gap:8px;">
      <span style="font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text-muted);">{esc(ticker)}</span>
      <span style="font-family:var(--mono);font-size:26px;font-weight:600;letter-spacing:-.01em;">${fmt_commas(spot, 2)}</span>
      <span style="font-family:var(--mono);font-size:13px;font-weight:600;color:{chg_color};">{esc(chg_txt)}</span>
    </div>
    <div style="display:flex;align-items:center;gap:9px;">
      <span style="font-size:11px;font-weight:700;letter-spacing:.05em;color:{regime_color};background:{badge_bg};border:1px solid {badge_bd};padding:5px 10px;border-radius:6px;">{esc(regime_label)}</span>
      <span style="font-family:var(--mono);font-size:11px;color:var(--text-dim);">{esc(regime_note)}</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;">
    {live_html}
    <span style="font-family:var(--mono);font-size:10.5px;color:var(--text-dim);">{esc(clock)}</span>
    <a class="hdr-refresh" target="_self" href="{refresh_href}">⟳ Refresh</a>
  </div>
</div>
"""
