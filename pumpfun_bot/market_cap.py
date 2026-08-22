"""
Looks up a token's current USD market cap via pump.fun's own frontend API,
for use as a buy filter (user-requested: skip anything under a minimum
market cap, alongside the existing socials/holder-count filters).

Unlike holder count (an on-chain, RPC-verifiable number), this comes from
pump.fun's own API - a third-party, unofficial source (see pumpportal_
client.py's module docstring for the same caveat about this project's other
external dependencies). Treat a lookup failure as "unknown", never as
"market cap is zero" - a broken/slow API response must not silently look
like a token that fails the filter.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.market_cap")

PUMPFUN_COIN_API_URL = "https://frontend-api-v3.pump.fun/coins/{mint}"


async def fetch_market_cap_usd(mint: str, timeout_sec: float = 5.0) -> float | None:
    """Returns the token's current USD market cap, or None if the lookup
    itself failed."""
    url = PUMPFUN_COIN_API_URL.format(mint=mint)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=timeout_sec, headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                if resp.status != 200:
                    logger.debug("pump.fun coin API gaf status %d voor %s", resp.status, mint)
                    return None
                data = await resp.json()
                market_cap = data.get("usd_market_cap")
                return float(market_cap) if market_cap is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon market cap niet ophalen voor %s: %s", mint, exc)
        return None
