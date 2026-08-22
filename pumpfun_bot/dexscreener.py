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
pair's 5m price change. User-requested window (originally 1h, shortened to
5m to better match how young these candidates actually are - most are
minutes old, not hours). A brand-new token may have no pair indexed yet;
treat that as "unknown", never as "0% change".
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.dexscreener")

DEXSCREENER_TOKEN_PAIRS_URL = "https://api.dexscreener.com/token-pairs/v1/solana/{mint}"


async def fetch_price_change_5m_pct(mint: str, timeout_sec: float = 5.0) -> float | None:
    """Returns the 5m price change percentage for the mint's highest-
    liquidity pair, or None if unavailable (lookup failure, no pair
    indexed yet, or a missing field)."""
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
                price_change = (pairs[0].get("priceChange") or {}).get("m5")
                return float(price_change) if price_change is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon 5m prijsverandering niet ophalen voor %s: %s", mint, exc)
        return None
