"""
Checks whether a fresh pump.fun launch's earliest transactions cluster into
a suspiciously small number of slots - a cheap, on-chain proxy for the
"bundled launch" pattern Axiom's Pulse flags as a UI column (multiple buyer
wallets prepared to buy the instant a mint became visible).

Investigated true Jito-bundle membership first: standard Solana RPC has no
"was this tx part of a bundle" field - that information only exists on the
Jito searcher/relay side, not on-chain. Same "no way to get the real thing
for free" situation already documented in holder_concentration.py for
luminos.capital's funding-source tracing. Same-slot clustering is the cheap
on-chain-only proxy instead: an organic buyer reacting to seeing a brand new
launch can't realistically land in the EXACT same slot (~400ms) as several
other buyers without either being bundled or running a dedicated sniping
bot - both are the "manufactured, not organic" pattern this exists to flag.

Only meaningful when called on a genuinely fresh mint. social_watch.py's
watch_window_sec caps how old a mint can be at its buy decision (60s by
default) - at that age, a single getSignaturesForAddress call at the
RPC-allowed limit (1000) reliably captures a launch's ENTIRE history, no
pagination needed. Would need real pagination (see holder_count.py's own
documented gaps for a similar limit) to stay accurate on an older/
higher-volume mint - not attempted here, since it isn't needed for this
use case: coingecko_movers' candidates are hours-to-days old, and querying
those would just measure current ambient trading volume instead of
launch-time bundling (confirmed live 2026-08-24: querying an old, actively-
traded mint returned its 15 most recent transactions ALL in one single
current slot - real, but meaningless for this purpose, and the exact
failure mode this module is not used for that population).
"""
from __future__ import annotations

import logging
from collections import Counter

import aiohttp

logger = logging.getLogger("pumpfun_bot.bundle_detection")


async def fetch_launch_slot_clustering(
    mint: str, rpc_http_url: str, timeout_sec: float = 5.0
) -> dict | None:
    """Returns {"total_txs", "distinct_slots", "max_txs_in_one_slot"} for
    this mint's earliest transactions, or None if the lookup itself failed.
    max_txs_in_one_slot is the real signal: how many distinct transactions
    landed in the SAME slot as each other - a high count on a freshly-
    launched mint means many wallets were prepared to buy before the launch
    was even organically discoverable, not that they raced each other in
    fair, independent reaction time."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
        "params": [mint, {"limit": 1000}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getSignaturesForAddress fout voor %s: %s", mint, data["error"])
                    return None
                result = data.get("result") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon slot-clustering niet ophalen voor %s: %s", mint, exc)
        return None

    if not result:
        return None
    slot_counts = Counter(item["slot"] for item in result if "slot" in item)
    if not slot_counts:
        return None
    return {
        "total_txs": len(result),
        "distinct_slots": len(slot_counts),
        "max_txs_in_one_slot": max(slot_counts.values()),
    }
