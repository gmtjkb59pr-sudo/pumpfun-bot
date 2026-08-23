"""
Looks up trending Solana pools via CoinGecko's free "Demo" API plan (the
/onchain endpoints, which mirror GeckoTerminal's on-chain DEX data) - a
second, much higher-budget discovery source than Birdeye's trending
endpoint (see birdeye.py), used by coingecko_movers.py to react to
already-existing tokens spiking much faster than birdeye_movers' 45-minute
cadence allows.

Requires a free CoinGecko account + Demo API key (COINGECKO_API_KEY, user
must create this themselves at coingecko.com - see coingecko_movers.py for
the strategy that uses it). The free Demo plan is 100 calls/min, 10,000
calls/month - polling every 5 minutes (8,640 calls/month) stays comfortably
within that. Confirmed live: the fully keyless tier also exists with no
signup, but CoinGecko's own docs explicitly say it's "not suitable for
production workloads, scheduled polling" - the Demo key is the right one
for a running bot, not the keyless convenience layer.

Returns raw "pool" dicts (id/attributes/relationships) - deliberately not
normalized here, since coingecko_movers.py needs several nested fields
(price_change_percentage across multiple windows, the base/quote token
relationship to figure out which side is the actual candidate mint) that
would be lossy to flatten prematurely.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.coingecko")

COINGECKO_TRENDING_POOLS_URL = "https://api.coingecko.com/api/v3/onchain/networks/{network}/trending_pools"


async def fetch_trending_pools(
    api_key: str,
    network: str = "solana",
    limit: int = 20,
    timeout_sec: float = 10.0,
) -> list[dict] | None:
    """Returns the raw list of trending pool dicts from CoinGecko's onchain
    API, or None if the lookup itself failed - never an empty list standing
    in for a real failure, since every call costs real, budget-limited
    quota (see module docstring)."""
    headers = {"x-cg-demo-api-key": api_key, "accept": "application/json"}
    params = {"limit": limit}
    url = COINGECKO_TRENDING_POOLS_URL.format(network=network)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    logger.debug("CoinGecko trending_pools gaf status %d", resp.status)
                    return None
                data = await resp.json()
                return data.get("data", [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon CoinGecko trending pools niet ophalen: %s", exc)
        return None


# the two quote currencies CoinGecko's Solana pools are almost always paired
# against - used to figure out which side of a pool is the actual candidate
# token, not the pairing currency itself
_KNOWN_QUOTE_MINTS = frozenset({
    "So11111111111111111111111111111111111111112",  # SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
})


def _strip_network_prefix(token_id: str | None) -> str | None:
    if not token_id:
        return None
    return token_id.split("_", 1)[1] if "_" in token_id else token_id


def parse_pool_candidate(pool: dict) -> dict | None:
    """Extracts {mint, pair_name, price_change_pct (dict of m5/m15/m30/h1/
    h6/h24 -> float|None), market_cap_usd, volume_24h_usd} from one raw
    CoinGecko pool dict - or None if the pool is missing the data needed to
    even identify a candidate mint. Picks whichever side of the pool isn't
    a known quote currency (SOL/USDC) as the real candidate - never guesses
    when neither or both sides look like a quote currency."""
    try:
        attrs = pool["attributes"]
        relationships = pool["relationships"]
    except (KeyError, TypeError):
        return None

    base_mint = _strip_network_prefix((relationships.get("base_token") or {}).get("data", {}).get("id"))
    quote_mint = _strip_network_prefix((relationships.get("quote_token") or {}).get("data", {}).get("id"))
    if not base_mint or not quote_mint:
        return None

    base_is_quote_currency = base_mint in _KNOWN_QUOTE_MINTS
    quote_is_quote_currency = quote_mint in _KNOWN_QUOTE_MINTS
    if base_is_quote_currency == quote_is_quote_currency:
        # both or neither look like a quote currency - can't safely tell
        # which side is the real candidate, don't guess
        return None
    mint = quote_mint if base_is_quote_currency else base_mint
    # price_usd must come from the SAME side as the candidate mint - a pool
    # exposes both base_token_price_usd and quote_token_price_usd, and
    # picking the wrong one silently records the pairing currency's price
    # (e.g. SOL's ~$95) as the candidate token's entry price
    price_raw = attrs.get("quote_token_price_usd" if base_is_quote_currency else "base_token_price_usd")
    try:
        price_usd = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        price_usd = None

    price_change_raw = attrs.get("price_change_percentage") or {}
    price_change_pct = {}
    for window in ("m5", "m15", "m30", "h1", "h6", "h24"):
        value = price_change_raw.get(window)
        try:
            price_change_pct[window] = float(value) if value is not None else None
        except (TypeError, ValueError):
            price_change_pct[window] = None

    market_cap_raw = attrs.get("market_cap_usd")
    try:
        market_cap_usd = float(market_cap_raw) if market_cap_raw is not None else None
    except (TypeError, ValueError):
        market_cap_usd = None

    volume_raw = (attrs.get("volume_usd") or {}).get("h24")
    try:
        volume_24h_usd = float(volume_raw) if volume_raw is not None else None
    except (TypeError, ValueError):
        volume_24h_usd = None

    return {
        "mint": mint,
        "pair_name": attrs.get("name", "?"),
        "price_usd": price_usd,
        "price_change_pct": price_change_pct,
        "market_cap_usd": market_cap_usd,
        "volume_24h_usd": volume_24h_usd,
    }
