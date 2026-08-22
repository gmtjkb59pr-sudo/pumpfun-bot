"""
Looks up trending Solana tokens via Birdeye's public token_trending API -
the discovery mechanism social_watch.py structurally cannot provide (it
only ever sees BRAND NEW token launches via PumpPortal's subscribeNewToken;
there is no way for it to notice an existing, days/weeks-old token
suddenly spiking in volume). Birdeye is a legitimate, official third-party
API - not scraping pump.fun's own site, which its Terms of Service
explicitly prohibit (see holder_concentration.py/dexscreener.py for the
earlier investigations that ruled that out).

Requires a free Birdeye account + API key (BIRDEYE_API_KEY, user must
create this themselves - see birdeye_movers.py for the strategy that uses
it). The free tier is 30,000 compute units/month at 1 request/second, and
this endpoint costs 30 CU/call - roughly 1,000 calls/month available.
Polling much more often than every ~45 minutes will exhaust that budget
before the month is out (see birdeye_movers.py's poll_interval_sec).
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.birdeye")

BIRDEYE_TRENDING_URL = "https://public-api.birdeye.so/defi/token_trending"


async def fetch_trending_tokens(
    api_key: str,
    chain: str = "solana",
    sort_by: str = "volumeUSD",
    sort_type: str = "desc",
    limit: int = 20,
    timeout_sec: float = 10.0,
) -> list[dict] | None:
    """Returns the raw list of trending token dicts from Birdeye (each with
    address/name/symbol/price/price24hChangePercent/volume24hUSD/...), or
    None if the lookup itself failed - never an empty list standing in for
    a real failure, since every call costs real, budget-limited CU (see
    module docstring). sort_by only accepts Birdeye's own enum: "rank",
    "volumeUSD", or "liquidity" - price-change filtering happens client
    side in birdeye_movers.py instead."""
    headers = {"X-API-KEY": api_key, "x-chain": chain}
    params = {"sort_by": sort_by, "sort_type": sort_type, "limit": limit}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                BIRDEYE_TRENDING_URL, headers=headers, params=params, timeout=timeout_sec,
            ) as resp:
                if resp.status != 200:
                    logger.debug("Birdeye gaf status %d", resp.status)
                    return None
                data = await resp.json()
                if not data.get("success"):
                    logger.debug("Birdeye antwoordde met success=false: %s", data)
                    return None
                return data.get("data", {}).get("tokens", [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon Birdeye trending tokens niet ophalen: %s", exc)
        return None
