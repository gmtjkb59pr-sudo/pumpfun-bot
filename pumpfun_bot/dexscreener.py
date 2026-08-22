"""
Looks up a token's recent price momentum via DexScreener's public API, for
use as a buy filter (user-requested: only buy candidates already showing
real upward momentum - a "movers" style signal - rather than just good
fundamentals at launch).

Investigated replicating pump.fun's own "Movers" tab directly, but their
Terms of Service explicitly prohibit bots/scripts/crawlers accessing the
Pump Platform to obtain information "in any manner not purposely provided"
- same conclusion as the luminos.capital investigation (see
holder_concentration.py). DexScreener is a separate, third-party
aggregator with a public, documented API explicitly built for
programmatic access (api.dexscreener.com, 60 req/min, no key required) -
using their API as intended, not scraping a UI.

Uses /token-pairs/v1/solana/{mint} (a per-mint lookup - DexScreener
doesn't expose a free bulk "top movers" listing, only per-meta-category
trending which isn't useful per-token) - returns the highest-liquidity
pair's price change across every window DexScreener reports (m5/h1/h6/h24
- there's no 1m/2m/15m granularity available from this API, contrary to
what was originally asked for). A brand-new token may have no pair indexed
yet; treat that as "unknown", never as "0% change".

User-requested (after asking to A/B-test 1m/2m/15m windows that don't
exist in this API): only m5 gates any buy decision (see social_watch.py),
but all four windows get logged with every trade so the actual best
window can be picked from real outcome data later - the same
log-first-then-decide pattern already used for holder_count.py.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.dexscreener")

DEXSCREENER_TOKEN_PAIRS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/{mint}"
PRICE_CHANGE_WINDOWS = ("m5", "h1", "h6", "h24")


async def fetch_price_changes_pct(mint: str, timeout_sec: float = 5.0) -> dict[str, float | None] | None:
    """Returns a dict of {window: price change %} for every window
    DexScreener reports on the mint's highest-liquidity pair (individual
    values may be None if that window is missing), or None entirely if
    unavailable (lookup failure or no pair indexed yet)."""
    url = DEXSCREENER_TOKEN_PAIRS_URL.format(mint=mint)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                if resp.status != 200:
                    logger.debug("DexScreener gaf status %d voor %s", resp.status, mint)
                    return None
                data = await resp.json()
                if not data:
                    # empty list - no pair indexed yet, not "0% change"
                    return None
                pairs = sorted(
                    data, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True,
                )
                price_change = pairs[0].get("priceChange") or {}
                return {
                    window: (float(price_change[window]) if price_change.get(window) is not None else None)
                    for window in PRICE_CHANGE_WINDOWS
                }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon prijsverandering niet ophalen voor %s: %s", mint, exc)
        return None
