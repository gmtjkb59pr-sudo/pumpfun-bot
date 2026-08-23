"""
Looks up a token's recent price momentum (and, for outcome_tracker.py's
REST fallback, absolute USD price) via DexScreener's public API. Momentum
was user-requested: only buy candidates already showing real upward
momentum - a "movers" style signal - rather than just good fundamentals at
launch. The absolute-price lookup exists for a different reason: confirmed
live that PumpPortal's subscribeTokenTrade WS feed only covers trades
routed through PumpPortal's own indexed venues (bonding curve / PumpSwap),
NOT every venue a mint might actually trade on - a real, high-volume token
can receive zero WS trade events despite being genuinely, heavily traded
elsewhere. Without this fallback, such a position gets silently abandoned
after STALE_PRICE_TIMEOUT_SEC with no exit ever attempted, even though a
real market price was available the whole time via a different source.

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


async def _fetch_best_pair(mint: str, timeout_sec: float) -> dict | None:
    """Returns the mint's highest-liquidity pair dict from DexScreener, or
    None if unavailable (lookup failure or no pair indexed yet) - shared by
    both public fetch functions below so there's only one place that picks
    "which pair is authoritative" for a mint."""
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
                    # empty list - no pair indexed yet, not "0% change"/"$0"
                    return None
                pairs = sorted(
                    data, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True,
                )
                return pairs[0]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon DexScreener pair niet ophalen voor %s: %s", mint, exc)
        return None


async def fetch_price_changes_pct(mint: str, timeout_sec: float = 5.0) -> dict[str, float | None] | None:
    """Returns a dict of {window: price change %} for every window
    DexScreener reports on the mint's highest-liquidity pair (individual
    values may be None if that window is missing), or None entirely if
    unavailable (lookup failure or no pair indexed yet)."""
    pair = await _fetch_best_pair(mint, timeout_sec)
    if pair is None:
        return None
    price_change = pair.get("priceChange") or {}
    return {
        window: (float(price_change[window]) if price_change.get(window) is not None else None)
        for window in PRICE_CHANGE_WINDOWS
    }


async def fetch_price_usd(mint: str, timeout_sec: float = 5.0) -> float | None:
    """Returns the mint's current USD price from DexScreener's highest-
    liquidity pair, or None if unavailable. Used as a REST fallback price
    reference (see module docstring) when PumpPortal's WS feed has no data
    for a mint - never guesses at a missing/unparseable price."""
    pair = await _fetch_best_pair(mint, timeout_sec)
    if pair is None:
        return None
    price_raw = pair.get("priceUsd")
    try:
        return float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        return None
