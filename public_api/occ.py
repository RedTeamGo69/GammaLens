"""
OCC option-symbol parsing (pure).

Public.com history rows carry the leg detail inside the OCC symbol —
``SOFI260417P00016000`` → root SOFI, expiry 2026-04-17, put, strike 16.000 —
and provide no separate strike/expiry fields, so this parser is the
foundation of the whole strategy-grouping pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_OCC_RE = re.compile(r"^([A-Z][A-Z0-9]{0,5}?)(\d{6})([CP])(\d{8})$")

# Weekly/AM-settled index roots → the underlying the trader thinks in.
_ROOT_TO_UNDERLYING = {"SPXW": "SPX", "NDXP": "NDX", "RUTW": "RUT"}


@dataclass(frozen=True)
class OccOption:
    root: str
    expiration: str        # ISO YYYY-MM-DD
    option_type: str       # "call" | "put"
    strike: float


def parse_occ_symbol(symbol: str | None) -> OccOption | None:
    """Parse an OCC-format option symbol; None for anything else (equities)."""
    if not symbol:
        return None
    m = _OCC_RE.match(symbol.strip().upper())
    if not m:
        return None
    root, yymmdd, cp, strike_raw = m.groups()
    try:
        year = 2000 + int(yymmdd[0:2])
        month, day = int(yymmdd[2:4]), int(yymmdd[4:6])
        expiration = f"{year:04d}-{month:02d}-{day:02d}"
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31 and strike > 0):
        return None
    return OccOption(
        root=root,
        expiration=expiration,
        option_type="call" if cp == "C" else "put",
        strike=strike,
    )


def underlying_of(occ: OccOption) -> str:
    """Trading-symbol underlying for an OCC root (SPXW → SPX, etc.)."""
    return _ROOT_TO_UNDERLYING.get(occ.root, occ.root)
