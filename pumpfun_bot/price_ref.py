"""
Shared helper for pulling a rough price/marketcap proxy out of PumpPortal
events. There's no single confirmed field name across event types (see the
README's warning that PumpPortal field names aren't guaranteed) so this tries
the candidates seen in new-token and trade events, in order of preference.

CONFIRMED LIVE (a real position bought via social_watch, exited 1.7s later
at exactly -100.0%, and a second one 4.9s later, same pattern): these fields
are NOT interchangeable measurements of the same thing. marketCapSol is
price-per-token times total supply; vSolInBondingCurve is the bonding
curve's virtual SOL reserve; price/initialBuy are yet other scales again.
For a fixed-supply pump.fun token, marketCapSol and a raw per-token price
differ by a factor of ~1e9 (the supply) - if an entry snapshot happens to
land on marketCapSol but a LATER trade event for the same mint is missing
that field and this function falls through to price instead, the resulting
"ratio" is comparing two values ~9 orders of magnitude apart, producing a
bogus near-(-100%) crash that never actually happened. Confirmed by the
observed pct_change: -99.99999426...%, not a clean -100.0 - consistent with
a genuine leftover fraction from dividing across mismatched scales, not a
real price actually hitting zero.

Fix: extract_price_ref() (used only for the ENTRY snapshot, when there's no
established field yet to require) still tries fields in this priority
order. Once a position is open, outcome_tracker.py records which field its
entry_ref actually came from and re-extracts ONLY that same field from
every later event via extract_price_ref_for_field() - never substituting a
different, scale-incompatible field mid-position. A later event missing
that exact field is treated as "no update this tick", not "try a fallback".
"""
from __future__ import annotations

PRICE_REF_FIELDS = ("marketCapSol", "vSolInBondingCurve", "price", "initialBuy")


def extract_price_ref(event: dict) -> float | None:
    value, _field = extract_price_ref_with_field(event)
    return value


def extract_price_ref_with_field(event: dict) -> tuple[float | None, str | None]:
    """Same field-priority search as extract_price_ref(), but also returns
    WHICH field supplied the value - callers that open a new position
    (there's no established field to require yet) should record this and
    use extract_price_ref_for_field() for every subsequent update on that
    same position, see this module's docstring for why."""
    for key in PRICE_REF_FIELDS:
        value = event.get(key)
        if value:
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue
    return None, None


def extract_price_ref_for_field(event: dict, field: str) -> float | None:
    """Extracts ONLY the given field - no fallback to a different field,
    even if one of the others is present. Use this for every price update
    on a position AFTER its entry snapshot, with the field recorded at
    entry time (see this module's docstring for why substituting fields
    mid-position produces a bogus reading)."""
    value = event.get(field)
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
