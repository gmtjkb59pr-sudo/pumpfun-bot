"""
Compares what the bot actually tracks (OutcomeTracker._pending, loaded from
disk at startup) against what the wallet really holds on-chain, so leftover
tokens from an earlier session or a bug never get silently confused with
what THIS session is managing.

Real gap found today: a live session's max_open_positions/exposure
accounting only ever knew about what its OWN process had bought and tracked
- it had no way to know the wallet also held other tokens from earlier
sessions (before position persistence existed, or abandoned across a
restart). The bot's own bookkeeping was internally consistent, but blind to
the wallet actually holding more than that.

Read-only and purely informational - this doesn't change what the bot
trades or manages, it just surfaces the difference (via a log line and an
alert) instead of leaving it invisible. Only meaningful in live mode -
dry-run never touches the real wallet.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.wallet_reconciliation")

TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


async def fetch_wallet_token_mints(
    wallet_pubkey: str, rpc_http_url: str, timeout_sec: float = 15.0
) -> set[str] | None:
    """Returns the set of mints with a nonzero balance currently held by the
    wallet, or None if the lookup itself failed - a broken RPC response
    shouldn't be read as "wallet holds nothing", it means "unknown"."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_pubkey,
            {"programId": TOKEN_2022_PROGRAM_ID},
            {"encoding": "jsonParsed"},
        ],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getTokenAccountsByOwner fout: %s", data["error"])
                    return None
                accounts = (data.get("result") or {}).get("value") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon wallet holdings niet ophalen: %s", exc)
        return None

    mints = set()
    for acc in accounts:
        info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
        amount = (info.get("tokenAmount") or {}).get("uiAmount") or 0
        mint = info.get("mint")
        if mint and amount > 0:
            mints.add(mint)
    return mints


async def find_untracked_wallet_holdings(
    wallet_pubkey: str, rpc_http_url: str, tracked_mints: set[str]
) -> set[str] | None:
    """Mints the wallet actually holds that the bot isn't tracking -
    leftovers from an earlier session, a bug, or a manual purchase. Returns
    None (not an empty set) if the on-chain lookup itself failed, so callers
    can tell "confirmed nothing untracked" apart from "couldn't check"."""
    held = await fetch_wallet_token_mints(wallet_pubkey, rpc_http_url)
    if held is None:
        return None
    return held - tracked_mints
