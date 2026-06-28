"""
Shared color palette and style constants used across all UI modules.

Palette = the "pro terminal" design tokens (Bloomberg/ThinkorSwim aesthetic).
Keys are kept stable so existing chart/HTML code picks up the new colors with
no call-site changes; only the hex values changed plus a few additive keys.
See ui_theme.TOKENS for the CSS-variable mirror of these values.
"""

COLORS = {
    # backgrounds / surfaces
    "bg_primary": "#0a0d13",     # bg-base (page)
    "bg_sidebar": "#11151c",     # bg-surface (cards)
    "bg_card": "#0c1119",        # bg-input (inner cells)
    "bg_row": "#141922",         # secondary buttons / alt rows
    # levels
    "spot": "#ffffff",
    "zero_gamma": "#25d8ef",     # accent-cyan
    "call_wall": "#2be88a",      # accent-green
    "put_wall": "#ff4d68",       # accent-red
    # GEX bars / regime
    "positive": "#2be88a",
    "negative": "#ff4d68",
    "bar_green": "#2be88a",
    "bar_red": "#ff4d68",
    # expected-move bands
    "em_level": "#a98bff",       # daily EM (purple)
    "em_weekly": "#f5c542",      # weekly EM (yellow)
    "em_monthly": "#6ea8ff",     # OpEx EM (blue)
    "profile_line": "#a98bff",
    "warning": "#ffb454",
    # text
    "text_white": "#e7edf5",     # text-primary
    "text_secondary": "#cbd5e1",
    "text_light": "#cbd5e1",
    "text_muted": "#93a1b2",
    "text_dim": "#5b6878",
    # borders / grid
    "border": "#1b212a",
    "border_mid": "#20272f",
    "grid_major": "#1b212a",
    "grid_minor": "#141922",
    "zeroline": "#2a3340",
    # extra accents
    "accent_blue": "#6ea8ff",
    "accent_purple": "#a98bff",
}

# Spread Finder color constants
SF_BG   = "#0a0d13"
SF_BULL = "#2be88a"
SF_BEAR = "#ff4d68"
SF_NEUT = "#93a1b2"
SF_WARN = "#ffb454"
SF_CARD = "#0c1119"
