"""
Pre-Flight tab — last-look gate for a proposed options trade.

Scaffold stub: registers the tab end-to-end (nav token, dispatch, render
signature) ahead of the real implementation. The full surface — behavioral
stats from the user's own fill history, weekly-HAR model gate, structure
gate, and Public.com margin preflight, rolled into a GREEN/YELLOW/RED
verdict — lands with the pre-flight engine.
"""
from __future__ import annotations

import streamlit as st

from models import GEXData


def _render_preflight_tab(
    spot: float,
    levels: dict,
    regime: dict,
    data: GEXData,
    ticker: str = "SPX",
    weekly_em: dict | None = None,
) -> None:
    st.info(
        "Pre-Flight is under construction — the behavioral / model / "
        "structure / margin gate for proposed trades will render here."
    )
