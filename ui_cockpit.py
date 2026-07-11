"""
Monday Cockpit tab — weekly XSP iron-condor decision surface.

Scaffold stub: registers the tab end-to-end (nav token, dispatch, render
signature) ahead of the real implementation. The full surface — weekly HAR
expected move vs straddle-implied move (VRP), Friday-expiry GEX walls, macro
events, condor proposal, and a single TRADE/SKIP verdict — lands with the
cockpit engine.
"""
from __future__ import annotations

import streamlit as st

from models import GEXData


def _render_cockpit_tab(
    spot: float,
    levels: dict,
    regime: dict,
    data: GEXData,
    ticker: str = "SPX",
    weekly_em: dict | None = None,
) -> None:
    st.info(
        "Monday Cockpit is under construction — the weekly XSP condor "
        "decision surface (HAR band vs implied move, GEX walls, events, "
        "TRADE/SKIP verdict) will render here."
    )
