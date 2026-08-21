"""
Looks up a token's actual current holder count on-chain, for logging
alongside each buy so it can eventually be correlated against real outcomes
(same pattern as has_socials/liquidity_sol/creator).

Unlike bonding-curve liquidity (a near-constant set by the protocol at
launch, confirmed to not differentiate anything) or socials (mostly just
metadata-file propagation delay, confirmed to pass ~87% of tokens
regardless), holder count measures real organic demand - how many distinct
wallets actually hold the token right now.

Deliberately NOT wired in as a buy filter yet - there's no evidence yet for
what holder count actually predicts here, and guessing a threshold would
repeat the same mistake as the untested liquidity/socials filters. Log it
first, set a real threshold once there's evidence.

Fetched via a filtered getProgramAccounts call on the Token-2022 program
(confirmed all pump.fun token accounts observed today use it, not the
legacy SPL Token program) rather than getTokenLargestAccounts, which caps
at 20 and undercounts any token with more holders than that. The dataSize
filter (170 bytes) matches the token-account layout observed on real
pump.fun tokens today (base 165 bytes + the immutableOwner extension) -
this counts token ACCOUNTS, a close proxy for holder count since a wallet
normally holds at most one associated token account per mint, but isn't a
strictly guaranteed 1:1 mapping.
"""
from __future__ import annotations

import logging
import time

import aiohttp

from .activity_log import append_jsonl

logger = logging.getLogger("pumpfun_bot.holder_count")

TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
PUMPFUN_TOKEN_ACCOUNT_DATA_SIZE = 170


async def fetch_holder_count(mint: str, rpc_http_url: str, timeout_sec: float = 5.0) -> int | None:
    """Returns the number of token accounts holding this mint right now, or
    None if the lookup itself failed - a slow/broken RPC response shouldn't
    be treated as "zero holders", it means "unknown"."""
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
                    {"dataSize": PUMPFUN_TOKEN_ACCOUNT_DATA_SIZE},
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


async def record_holder_count(mint: str, rpc_http_url: str) -> None:
    """Fire-and-forget: fetches and logs the holder count without blocking
    the caller's event loop - meant to run as a background task right after
    a buy, so the ~100-200ms lookup never delays processing the next launch
    event (matters most for sniper, where that's the whole point)."""
    count = await fetch_holder_count(mint, rpc_http_url)
    if count is None:
        return
    append_jsonl({"type": "holder_count", "ts": time.time(), "mint": mint, "count": count})
