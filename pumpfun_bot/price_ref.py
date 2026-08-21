"""
Shared helper for pulling a rough price/marketcap proxy out of PumpPortal
events. There's no single confirmed field name across event types (see the
README's warning that PumpPortal field names aren't guaranteed) so this tries
the candidates seen in new-token and trade events, in order of preference.
Used consistently for both the entry snapshot and later outcome checkpoints
so at least the two sides of a comparison are extracted the same way.
"""
from __future__ import annotations


def extract_price_ref(event: dict) -> float | None:
    for key in ("marketCapSol", "vSolInBondingCurve", "price", "initialBuy"):
        value = event.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None
