"""
Looks up a token's actual current holder count on-chain, for logging
alongside each buy so it can eventually be correlated against real outcomes
(same pattern as has_socials/liquidity_sol/creator).

Unlike bonding-curve liquidity (a near-constant set by the protocol at
launch, confirmed to not differentiate anything) or socials (mostly just
metadata-file propagation delay, confirmed to pass ~87% of tokens
regardless), holder count measures real organic demand - how many distinct
wallets actually hold the token right now.

Deliberately NOT wired in as a buy filter for social_watch yet - there's no
evidence yet for what holder count actually predicts there, and guessing a
threshold would repeat the same mistake as the untested liquidity/socials
filters. Log it first, set a real threshold once there's evidence. (It IS
used as a real filter for birdeye_movers, a different population of
already-established tokens - see below for why that population needed a
second code path here.)

A mint can be owned by either of two SPL token programs, and they need
completely different queries:

- Token-2022 (confirmed all pump.fun bonding-curve launches use this):
  a getProgramAccounts scan filtered by mint works fine - a small enough
  dataset for an RPC provider to scan directly.
- Classic SPL Token: the SAME query is REJECTED outright by Helius
  ("Too many accounts requested") - this program spans every classic-SPL
  token account on all of Solana, not just this mint's, and a memcmp
  filter by mint still requires scanning that whole universe server-side
  before applying it. Confirmed live: birdeye_movers (which discovers
  already-established tokens, not just fresh pump.fun launches) hit a
  100% "holder count unknown" rate before this was fixed, because those
  tokens are commonly classic-SPL, not Token-2022.

  Falls back to getTokenLargestAccounts for these - a real, working query
  for any mint regardless of program, but capped at the top 20 accounts.
  That's a real undercount for a widely-held token, but it's a safe
  direction to be wrong in: it can only make min_holder_count harder to
  clear, never falsely claim a low-holder token looks fine. Good enough
  for a >= threshold check; would need real pagination
  (getProgramAccountsV2) to also report an accurate count past 20.

Uses getProgramAccounts (Token-2022 path) rather than
getTokenLargestAccounts unconditionally, because the latter's 20-account
cap would undercount any token with more real holders than that - this
counts token ACCOUNTS, a close proxy for holder count since a wallet
normally holds at most one associated token account per mint, but isn't a
strictly guaranteed 1:1 mapping.

Earlier tried adding a dataSize:170 filter (matching one real pump.fun
token-account layout observed) to narrow the getProgramAccounts scan, but
found - by comparing filtered vs unfiltered results on a real mint - that
it silently excluded about half the real holder accounts (170 bytes isn't
the only valid layout; accounts with a different extension set land on a
different size). Undercounting by ~50% is worse than the extra RPC-side
scan cost, so this only filters on the mint itself now.
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from .activity_log import append_jsonl

logger = logging.getLogger("pumpfun_bot.holder_count")

TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
SPL_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
# getProgramAccounts' index lags behind the live chain state - confirmed
# directly by re-checking a mint minutes after logging 0 holders right after
# our own buy and finding 7 real holders. Checking immediately after a buy
# reliably reads as 0 (it hasn't even indexed our own just-executed
# transaction yet), which would make every logged count worthless noise -
# wait before checking instead of recording a number known to be wrong.
INDEXING_DELAY_SEC = 20


async def _fetch_mint_owner_program(mint: str, rpc_http_url: str, timeout_sec: float) -> str | None:
    """Returns the program ID that owns this mint account, or None if the
    lookup itself failed - determines which of the two queries below is
    even possible for this mint."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
        "params": [mint, {"encoding": "base64"}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getAccountInfo fout voor %s: %s", mint, data["error"])
                    return None
                value = (data.get("result") or {}).get("value")
                if value is None:
                    return None
                return value.get("owner")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon mint-owner niet ophalen voor %s: %s", mint, exc)
        return None


async def _fetch_holder_count_token2022(mint: str, rpc_http_url: str, timeout_sec: float) -> int | None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getProgramAccounts",
        "params": [
            TOKEN_2022_PROGRAM_ID,
            {
                "encoding": "base64",
                "dataSlice": {"offset": 0, "length": 0},
                "filters": [
                    {"memcmp": {"offset": 0, "bytes": mint}},
                ],
            },
        ],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getProgramAccounts fout voor %s: %s", mint, data["error"])
                    return None
                return len(data.get("result") or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon holder count niet ophalen voor %s: %s", mint, exc)
        return None


async def _fetch_holder_count_classic_spl(mint: str, rpc_http_url: str, timeout_sec: float) -> int | None:
    """getProgramAccounts on the classic SPL Token program filtered by mint
    is rejected outright by Helius (spans every classic-SPL token account
    on Solana) - getTokenLargestAccounts is the fallback that actually
    works, at the cost of capping at the top 20 (see module docstring)."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTokenLargestAccounts", "params": [mint],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getTokenLargestAccounts fout voor %s: %s", mint, data["error"])
                    return None
                return len((data.get("result") or {}).get("value") or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon holder count (classic SPL) niet ophalen voor %s: %s", mint, exc)
        return None


async def fetch_holder_count(mint: str, rpc_http_url: str, timeout_sec: float = 5.0) -> int | None:
    """Returns the number of token accounts holding this mint right now
    (or, for a classic-SPL mint, the top-20-account count as a floor - see
    module docstring), or None if the lookup itself failed - a slow/broken
    RPC response shouldn't be treated as "zero holders", it means
    "unknown"."""
    owner = await _fetch_mint_owner_program(mint, rpc_http_url, timeout_sec)
    if owner == TOKEN_2022_PROGRAM_ID:
        return await _fetch_holder_count_token2022(mint, rpc_http_url, timeout_sec)
    if owner == SPL_TOKEN_PROGRAM_ID:
        return await _fetch_holder_count_classic_spl(mint, rpc_http_url, timeout_sec)
    return None


async def record_holder_count(
    mint: str, rpc_http_url: str, delay_sec: float = INDEXING_DELAY_SEC
) -> None:
    """Fire-and-forget: waits for the indexer to catch up, then fetches and
    logs the holder count, without blocking the caller's event loop - meant
    to run as a background task right after a buy, so neither the delay nor
    the ~100-200ms lookup itself ever delays processing the next launch
    event (matters most for sniper, where that's the whole point)."""
    await asyncio.sleep(delay_sec)
    count = await fetch_holder_count(mint, rpc_http_url)
    if count is None:
        return
    append_jsonl({"type": "holder_count", "ts": time.time(), "mint": mint, "count": count})
