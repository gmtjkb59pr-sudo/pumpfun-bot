"""
Checks a token's off-chain metadata (the `uri` field on its launch event) for
social links.

The subscribeNewToken event itself never carries twitter/telegram/website
directly - confirmed by sampling live launches and inspecting the raw event
schema. That data only exists in the metadata JSON the `uri` points to
(IPFS or an HTTP host chosen by whatever tool the creator used), which has
to be fetched separately. Field names inside that JSON, once fetched,
DO match what the sniper's require_socials filter already checks for
(twitter/telegram/website) - confirmed against real metadata samples.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.social_metadata")

SOCIAL_FIELDS = ("twitter", "telegram", "website")


async def fetch_has_socials(uri: str, timeout_sec: float = 5.0) -> bool:
    """Returns whether the token's metadata declares any social link.
    Any failure (timeout, 404, invalid JSON, missing uri) is treated as "no
    socials found" rather than raised - a slow/broken metadata host
    shouldn't crash the watch loop, it should just mean this candidate
    doesn't qualify."""
    if not uri:
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(uri, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon metadata niet ophalen van %s: %s", uri, exc)
        return False

    if not isinstance(data, dict):
        return False
    return any(data.get(field) for field in SOCIAL_FIELDS)
