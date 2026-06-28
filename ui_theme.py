"""
Terminal-aesthetic UI foundation for the GEX dashboard redesign.

Holds the global CSS (design tokens, Streamlit chrome overrides, component
classes), the query-param-driven UI-state helpers that let custom HTML
controls stay interactive (see memory: streamlit-custom-html-interactivity),
and the sticky header renderer.

NOTHING here touches data/model logic — it is pure presentation + URL state.
"""
from __future__ import annotations

import base64
import html as _html
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Brand assets
# ─────────────────────────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent / "assets"
#: Full-resolution logo (transparent corners) — used for the browser favicon.
LOGO_PATH = ASSETS_DIR / "gamma_lens_logo.png"
#: Small mark — base64-embedded inline in the sticky header.
_LOGO_MARK_PATH = ASSETS_DIR / "gamma_lens_mark_64.png"


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """Return the Gamma Lens mark as a base64 PNG data URI for an inline ``<img>``.

    Read once per process and cached. Returns ``""`` if the asset is missing,
    in which case the header falls back to the text wordmark + ``Γ`` glyph.
    """
    try:
        raw = _LOGO_MARK_PATH.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


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

# Curated quick-pick (featured) tickers — the always-present Excel-export
# defaults, so the featured row and the workbook track each other.
QUICK_TICKERS = ["SPX", "XSP", "SPY", "QQQ", "NDX", "XND"]

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
# Expiration / refresh token → internal-mode mapping
# ─────────────────────────────────────────────────────────────────────────────
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
[data-testid="stAppViewBlockContainer"], .block-container {{
  padding:0 16px 16px !important; max-width:100% !important;
}}
[data-testid="stMainBlockContainer"] {{ padding-top:0 !important; max-width:100% !important; }}

/* ── Hide the native sidebar — controls live in an in-page column instead ── */
section[data-testid="stSidebar"] {{ display:none !important; }}
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display:none !important; }}

/* ── Body layout: aside | main columns (the terminal "term-body", all on one
   page). Scoped to the horizontal block that holds the settings card so the
   finders' own columns are unaffected. ── */
[data-testid="stHorizontalBlock"]:has(.st-key-settings_card) {{ gap:16px !important; align-items:stretch; }}
[data-testid="stHorizontalBlock"]:has(.st-key-settings_card) > [data-testid="stColumn"]:first-child {{
  flex:0 0 332px !important; width:332px !important; min-width:300px;
}}
[data-testid="stHorizontalBlock"]:has(.st-key-settings_card) > [data-testid="stColumn"]:last-child {{
  flex:1 1 auto !important; min-width:0;
}}
[data-testid="stHorizontalBlock"]:has(.st-key-settings_card) > [data-testid="stColumn"] [data-testid="stVerticalBlock"] {{ gap:0.7rem; }}
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
  width:32px; height:32px; border-radius:8px; object-fit:contain; display:block;
  filter:drop-shadow(0 0 10px rgba(43,232,138,.40));
}}
.term-logo-fallback {{
  background:linear-gradient(135deg,var(--green),#0f9b62);
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

/* ─────────────────────────────────────────────────────────────────────────
   Native widget skins — make st.pills / st.segmented_control / st.text_input /
   st.button look like the custom chips/segments/tabs. Drives the smooth
   (websocket) controls that replaced the anchor/query-param navigation.
   ───────────────────────────────────────────────────────────────────────── */
/* custom eyebrows / active line rendered via st.markdown */
.ctl-eyebrow {{
  font-size:10px; letter-spacing:.14em; color:var(--text-dim); font-weight:700;
  text-transform:uppercase; margin:13px 0 4px;
}}
.ctl-active {{ font-size:10px; color:var(--text-dim); margin:6px 0 2px; }}
.ctl-active .t {{ font-family:var(--mono); color:var(--text-secondary); font-weight:600; }}
.ctl-active .ty {{ color:var(--text-dim); }}

/* settings card wrapper (st.container(border=True, key="settings_card")) */
.st-key-settings_card {{
  background:var(--bg-surface); border:1px solid var(--border) !important;
  border-radius:11px !important; padding:12px 13px 14px !important;
}}
.st-key-settings_card [data-testid="stVerticalBlock"] {{ gap:0.35rem; }}

/* kill the little hover toolbars on markdown/elements in the settings card */
.st-key-settings_card [data-testid="stElementToolbar"] {{ display:none !important; }}

/* search text input */
.st-key-settings_card [data-baseweb="input"],
.st-key-settings_card [data-baseweb="base-input"] {{
  background:var(--bg-input) !important; border:1px solid var(--border-mid) !important;
  border-radius:8px !important;
}}
.st-key-settings_card [data-testid="stTextInput"] input {{
  background:transparent !important; color:var(--text-primary) !important;
  font-family:var(--mono) !important; font-size:11.5px !important; padding:8px 10px !important;
}}
.st-key-settings_card [data-testid="stTextInput"]:focus-within [data-baseweb="input"] {{
  border-color:var(--green) !important; box-shadow:0 0 0 2px rgba(43,232,138,.12);
}}

/* pills + segmented control → chip look */
[data-testid="stPills"] [data-testid="stButtonGroup"],
[data-testid="stSegmentedControl"] [data-testid="stButtonGroup"] {{ gap:6px; flex-wrap:wrap; }}
[data-testid^="stBaseButton-pills"],
[data-testid^="stBaseButton-segmented_control"] {{
  font-family:var(--mono) !important; font-size:11px !important; font-weight:600 !important;
  background:var(--bg-row) !important; color:var(--text-muted) !important;
  border:1px solid var(--border-mid) !important; border-radius:7px !important;
  padding:5px 11px !important; min-height:0 !important; line-height:1.45 !important;
}}
[data-testid^="stBaseButton-pills"]:hover,
[data-testid^="stBaseButton-segmented_control"]:hover {{
  color:var(--text-secondary) !important; border-color:#2e3a48 !important;
}}
[data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_controlActive"] {{
  background:rgba(43,232,138,.14) !important; color:var(--green) !important;
  border-color:rgba(43,232,138,.45) !important;
}}

/* refresh-now button */
.st-key-refresh_now_btn button {{
  background:var(--bg-row) !important; color:var(--text-secondary) !important;
  border:1px solid var(--border-mid) !important; border-radius:8px !important;
  font-family:var(--sans) !important; font-size:11px !important; font-weight:600 !important;
  padding:7px !important; min-height:0 !important;
}}
.st-key-refresh_now_btn button:hover {{ border-color:var(--green) !important; color:var(--green) !important; }}

/* date-range input */
.st-key-settings_card [data-testid="stDateInput"] [data-baseweb="input"] {{
  background:var(--bg-input) !important; border:1px solid var(--border-mid) !important;
}}
.st-key-settings_card [data-testid="stDateInput"] input {{
  background:transparent !important; color:var(--text-primary) !important;
  font-family:var(--mono) !important; font-size:11px !important;
}}

/* main tab selector → underline tab-bar look (overrides the chip skin above) */
.st-key-tab_seg [data-testid="stButtonGroup"] {{ gap:2px !important; border-bottom:1px solid var(--border); }}
.st-key-tab_seg [data-testid^="stBaseButton-segmented_control"] {{
  background:transparent !important; border:none !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
  color:var(--text-dim) !important; font-family:var(--sans) !important;
  font-size:12.5px !important; font-weight:600 !important; padding:9px 16px !important;
  margin-bottom:-1px;
}}
.st-key-tab_seg [data-testid^="stBaseButton-segmented_control"]:hover {{
  color:var(--text-secondary) !important; border-color:transparent !important;
}}
.st-key-tab_seg [data-testid="stBaseButton-segmented_controlActive"] {{
  color:var(--text-primary) !important; background:transparent !important;
  border-bottom:2px solid var(--green) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────
   Finder tabs (Phase 2) — Spread Finder + 0DTE Finder terminal styling.
   Pure presentation; all numbers come from the (untouched) spread-plan logic.
   ───────────────────────────────────────────────────────────────────────── */
.sf-section-title {{ font-size:14px; font-weight:700; color:var(--text-primary); }}
.sf-section-sub {{ font-size:10.5px; color:var(--text-dim); margin-top:3px; }}
.sf-eyebrow {{
  font-size:9.5px; letter-spacing:.08em; text-transform:uppercase; font-weight:700;
  color:var(--text-dim); margin:2px 0 7px;
}}

/* risk-tier selector: 4 equal-width buttons spanning the full row (key is
   sf_risk_tier_<ticker>, so match the st-key-* class by prefix) */
[class*="st-key-sf_risk_tier_"] [data-testid="stButtonGroup"] {{
  display:flex !important; width:100% !important; flex-wrap:nowrap !important; gap:6px;
}}
[class*="st-key-sf_risk_tier_"] [data-testid^="stBaseButton-segmented_control"] {{
  flex:1 1 0 !important; justify-content:center !important; text-align:center;
}}

/* metric-card grid (controls + forecast metrics) */
.sf-cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--border); border:1px solid var(--border); border-radius:11px; overflow:hidden; }}
.sf-card {{ background:var(--bg-surface); padding:11px 13px; }}
.sf-card .cl {{ font-size:9.5px; font-weight:700; letter-spacing:.05em; color:var(--text-dim); text-transform:uppercase; margin-bottom:5px; }}
.sf-card .cv {{ font-family:var(--mono); font-size:18px; font-weight:600; color:var(--text-primary); }}
.sf-card .cs {{ font-size:10px; color:var(--text-muted); margin-top:3px; }}

/* 8-column spread tables */
.sf-table-wrap {{ border:1px solid var(--border); border-radius:9px; overflow:hidden; margin-top:4px; }}
.sf-table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11px; }}
.sf-table th {{
  font-family:var(--sans); font-size:8.5px; font-weight:700; letter-spacing:.04em;
  color:var(--text-dim); background:var(--bg-input); padding:6px 8px; text-align:right;
  text-transform:uppercase; white-space:nowrap;
}}
.sf-table th:first-child, .sf-table td:first-child {{ text-align:left; }}
.sf-table td {{ padding:5px 8px; text-align:right; color:var(--text-muted); border-top:1px solid var(--border); white-space:nowrap; }}
.sf-table tbody tr.best {{ background:rgba(43,232,138,.09); }}
.sf-table td.short {{ color:var(--text-primary); font-weight:600; }}
.sf-table td.long {{ color:var(--text-muted); }}
.sf-table td.cr {{ color:var(--green); }}
.sf-table td.oky {{ color:var(--green); font-weight:700; }}
.sf-table td.okn {{ color:var(--red); font-weight:700; }}
.sf-cap {{ font-size:10px; color:var(--text-dim); margin-top:6px; line-height:1.5; }}

/* GEX context panel */
.sf-gex {{ background:var(--bg-input); border:1px solid var(--border); border-radius:9px; padding:12px 13px; }}
.sf-gex .reg {{ font-family:var(--sans); font-size:18px; font-weight:700; margin:2px 0 9px; }}
.sf-gex .row {{ display:flex; justify-content:space-between; align-items:baseline; padding:4px 0; font-size:11.5px; }}
.sf-gex .row .k {{ color:var(--text-muted); }}
.sf-gex .row .v {{ font-family:var(--mono); font-weight:600; color:var(--text-secondary); }}
.sf-gex .row .sub {{ font-size:9.5px; color:var(--text-dim); }}

/* VRP banner + generic finder note cards */
.sf-vrp {{ border-radius:0 8px 8px 0; padding:10px 14px; font-size:11.5px; line-height:1.5; color:var(--text-secondary); margin:2px 0 6px; }}
.sf-note {{ background:var(--bg-input); border:1px solid var(--border); border-radius:9px; padding:11px 13px; font-size:11px; color:var(--text-muted); line-height:1.55; }}
.sf-note.warn {{ border-left:3px solid var(--amber); }}
.sf-note.info {{ border-left:3px solid var(--blue); }}
.sf-note.ok {{ border-left:3px solid var(--green); }}

/* forecast range gauge */
.sf-gauge {{ padding:6px 2px 2px; }}
.sf-gauge .track {{ position:relative; height:6px; background:var(--border); border-radius:3px; margin:22px 0 6px; }}
.sf-gauge .band {{ position:absolute; top:0; bottom:0; }}
.sf-gauge .dot {{ position:absolute; top:50%; width:12px; height:12px; border-radius:50%; transform:translate(-50%,-50%);
  background:var(--green); border:2px solid var(--bg-input); box-shadow:0 0 8px rgba(43,232,138,.6); }}
.sf-gauge .vix {{ position:absolute; top:-6px; bottom:-6px; border-left:2px dashed #fff; }}
.sf-gauge .caret {{ position:absolute; top:-16px; transform:translateX(-50%); color:#fff; font-size:9px; }}
.sf-gauge .axis {{ display:flex; justify-content:space-between; font-family:var(--mono); font-size:9px; color:var(--text-dim); margin-top:4px; }}

/* swim-lane strike map */
.sf-map {{ display:flex; font-family:var(--mono); }}
.sf-map .yax {{ width:52px; flex:none; display:flex; flex-direction:column; padding:16px 0; }}
.sf-map .yax .lane {{ flex:1; display:flex; align-items:center; font-size:9px; font-weight:700; color:var(--text-dim); letter-spacing:.06em; }}
.sf-map .area {{ flex:1; position:relative; border:1px solid var(--border); border-radius:8px; overflow:hidden; background:var(--bg-input); padding:16px 0; }}
.sf-map .lane-row {{ position:relative; height:34px; border-bottom:1px solid #1b2332; }}
.sf-map .lane-row:last-child {{ border-bottom:none; }}
.sf-map .grid {{ position:absolute; top:0; bottom:0; border-left:1px solid #1b2332; }}
.sf-map .safe {{ position:absolute; top:0; bottom:0; }}
.sf-map .mk {{ position:absolute; top:50%; transform:translate(-50%,-50%); font-size:11px; white-space:nowrap; }}
.sf-map .mk .lbl {{ font-size:8px; color:var(--text-dim); }}
.sf-map-x {{ display:flex; justify-content:space-between; font-family:var(--mono); font-size:8.5px; color:var(--text-dim); margin:5px 0 0 52px; }}
.sf-map-foot {{ text-align:center; font-size:9px; color:var(--text-dim); margin-top:4px; }}

/* main-area native control skins (finder inputs / selectbox / buttons) — scoped
   to stMain so the sidebar's own skins are untouched. */
section[data-testid="stMain"] [data-testid="stNumberInput"] input,
section[data-testid="stMain"] [data-testid="stTextInput"] input {{
  background:var(--bg-input) !important; color:var(--text-primary) !important;
  font-family:var(--mono) !important; font-size:12px !important;
}}
section[data-testid="stMain"] [data-baseweb="input"],
section[data-testid="stMain"] [data-baseweb="select"] > div {{
  background:var(--bg-input) !important; border-color:var(--border-mid) !important;
  border-radius:8px !important;
}}
section[data-testid="stMain"] [data-testid="stNumberInputStepDown"],
section[data-testid="stMain"] [data-testid="stNumberInputStepUp"] {{
  background:var(--bg-row) !important; color:var(--text-muted) !important;
}}
section[data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
  background:var(--bg-input) !important; border-color:var(--border-mid) !important;
  font-family:var(--mono) !important; font-size:12px !important; color:var(--text-primary) !important;
}}
section[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
  font-size:9.5px !important; letter-spacing:.04em; color:var(--text-dim) !important;
  font-weight:700 !important; text-transform:uppercase;
}}
section[data-testid="stMain"] [data-testid="stBaseButton-secondary"],
section[data-testid="stMain"] [data-testid="stDownloadButton"] button {{
  background:var(--bg-row) !important; color:var(--text-secondary) !important;
  border:1px solid var(--border-mid) !important; border-radius:8px !important;
  font-family:var(--sans) !important; font-weight:600 !important; font-size:12px !important;
}}
section[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover,
section[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover {{
  border-color:var(--green) !important; color:var(--green) !important;
}}
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] {{
  background:var(--green) !important; border:none !important;
  border-radius:8px !important; font-family:var(--sans) !important;
  font-weight:700 !important; font-size:12px !important;
}}
section[data-testid="stMain"] [data-testid="stBaseButton-primary"],
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] p,
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] div,
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] span,
section[data-testid="stMain"] [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] {{
  color:#06140c !important;
}}
section[data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover {{
  background:#3df59a !important; box-shadow:0 0 12px rgba(43,232,138,.45) !important;
}}

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
    refresh_href: str | None = None,
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
    # Refresh is a native sidebar button now (so it reruns over the websocket,
    # not a full reload); only render a header refresh link if one is passed.
    refresh_html = (
        f'<a class="hdr-refresh" target="_self" href="{refresh_href}">⟳ Refresh</a>'
        if refresh_href else ""
    )
    _logo_uri = logo_data_uri()
    logo_html = (
        f'<img class="term-logo" src="{_logo_uri}" alt="Gamma Lens" />'
        if _logo_uri else '<div class="term-logo term-logo-fallback">Γ</div>'
    )
    return f"""
<div class="term-header">
  <div style="display:flex;align-items:center;gap:11px;">
    {logo_html}
    <div style="display:flex;flex-direction:column;line-height:1.15;">
      <span style="font-size:13px;font-weight:700;letter-spacing:.02em;">GAMMA LENS</span>
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
    {refresh_html}
  </div>
</div>
"""
