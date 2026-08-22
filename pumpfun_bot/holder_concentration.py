"""
Measures what share of a token's supply sits in its top 10 holder accounts,
as a cheap on-chain proxy for "manufactured" distribution - a bundled/rug-
style launch concentrates most of the float in a handful of wallets so it
can be dumped together, while an organic launch spreads out.

Investigated luminos.capital as a ready-made version of this signal (they
run a much deeper analysis: gas fingerprints, funding-source tracing, wallet
age clustering). Their site has no public API, and their Terms of Service
explicitly prohibit "scrape, bulk-harvest, or systematically extract data
from the service beyond ordinary interactive use" - wiring the bot to query
them per-candidate would violate that. Top-10 concentration is the one
signal cheap enough to compute ourselves directly from Solana RPC data
(public on-chain data, not scraped from a third party): Luminos's own
methodology page bands it as <30% healthy, 30-50% caution, >50% flag - this
module reproduces just that one check, not their full engine.

Uses getTokenLargestAccounts (caps at the top 20 by design) rather than
getProgramAccounts (see holder_count.py) - unlike total holder count, where
that 20-account cap would undercount, concentration only ever needs the top
10 anyway, so the cap is irrelevant here.
"""
from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.holder_concentration")


async def fetch_top10_concentration_pct(
    mint: str, rpc_http_url: str, timeout_sec: float = 5.0
) -> float | None:
    """Returns the percentage of total supply held by the top 10 accounts,
    or None if the lookup itself failed or supply is reported as zero -
    never treat a failed/degenerate lookup as "0% concentration"."""
    supply_payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [mint],
    }
    largest_payload = {
        "jsonrpc": "2.0", "id": 2, "method": "getTokenLargestAccounts", "params": [mint],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=supply_payload, timeout=timeout_sec) as resp:
                supply_data = await resp.json()
            async with session.post(rpc_http_url, json=largest_payload, timeout=timeout_sec) as resp:
                largest_data = await resp.json()

            if "error" in supply_data or "error" in largest_data:
                logger.debug(
                    "RPC fout bij concentratie-check voor %s: %s / %s",
                    mint, supply_data.get("error"), largest_data.get("error"),
                )
                return None

            total_supply = float(supply_data["result"]["value"]["amount"])
            if total_supply <= 0:
                return None

            top_accounts = largest_data["result"]["value"]
            top10_amount = sum(float(acc["amount"]) for acc in top_accounts[:10])
            return (top10_amount / total_supply) * 100
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon top-10 concentratie niet ophalen voor %s: %s", mint, exc)
        return None
