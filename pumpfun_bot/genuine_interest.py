"""
Free, per-candidate proxy for "is this real organic interest, or a handful
of wallets manufacturing volume" - user-requested 2026-08-25 ("okay how can
i make the bot profitable" -> "build") after the session converged on: this
bot's real problem isn't execution speed (already addressed), it's that no
entry-signal edge has been proven, and platform-wide data shows ~90%+ of
top pump.fun wallets are bots, some of it wash-trading.

This is the free, scoped-down version of the "own on-chain infrastructure"
plan explored earlier in this session (see
/Users/sidanepontjodikromo/.claude/plans/mutable-crafting-kitten.md) - that
plan needed a paid Geyser streaming subscription ($49-999/month) to watch
the whole chain continuously. This instead reuses the RPC budget already
being paid for (Helius), doing the SAME wash-trading check but only for a
specific candidate already discovered by an existing strategy, at the
moment it's being evaluated - no continuous streaming needed, matching the
exact reasoning that already worked once this session (CoinGecko's free
pool_created_at data substantially replaced the need for the expensive
plan's "revival detection" half - see coingecko.py's module docstring).

Confirmed live 2026-08-25: a real transaction's logMessages reliably
contain a plain "Program log: Instruction: Buy" or "...Sell" line (the
pump.fun bonding-curve program's own log, not decoded from any
undocumented binary struct), and the trader's own wallet is the
transaction's fee payer (message.accountKeys[0]) - both simple, safe to
rely on, unlike the raw instruction-construction work deliberately NOT
attempted elsewhere in this codebase (see pumpportal_client.py's
build_and_send_trade_via_jito_bundle's docstring for why that one WAS
avoided - undocumented account layouts are a different, much riskier
kind of parsing than a human-readable log line).

Real, meaningfully bigger RPC cost than any other per-candidate check in
this codebase (holder_count/holder_concentration/bundle_detection are
each 1-2 calls; this is 1 + WINDOW calls, fetched in parallel to keep
wall-clock latency reasonable) - bounded by DEFAULT_WINDOW, and log-only
for now (see social_watch.py's use of this), same "log first, gate once
there's evidence" precedent as every other new signal this session.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger("pumpfun_bot.genuine_interest")

DEFAULT_WINDOW = 30


async def _fetch_recent_signatures(
    mint: str, rpc_http_url: str, limit: int, timeout_sec: float,
) -> list[str] | None:
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
        "params": [mint, {"limit": limit}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getSignaturesForAddress fout voor %s: %s", mint, data["error"])
                    return None
                result = data.get("result") or []
                return [item["signature"] for item in result if "signature" in item]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon signatures niet ophalen voor %s: %s", mint, exc)
        return None


async def _classify_transaction(
    signature: str, rpc_http_url: str, timeout_sec: float,
) -> tuple[str, str] | None:
    """Returns (wallet, "buy"|"sell") for this transaction, or None if it
    couldn't be fetched or doesn't contain a recognizable pump.fun Buy/Sell
    instruction (e.g. a create, or an unrelated program touching the same
    mint account) - excluded from the ratio calculations below, not
    treated as either direction."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
        "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    return None
                result = data.get("result")
                if not result:
                    return None
                logs = (result.get("meta") or {}).get("logMessages") or []
                account_keys = (result.get("transaction") or {}).get("message", {}).get("accountKeys") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon transactie %s niet ophalen: %s", signature, exc)
        return None

    if not account_keys:
        return None
    fee_payer = account_keys[0]
    wallet = fee_payer.get("pubkey") if isinstance(fee_payer, dict) else fee_payer
    if not wallet:
        return None

    is_buy = any("Instruction: Buy" in line for line in logs)
    is_sell = any("Instruction: Sell" in line for line in logs)
    if is_buy and not is_sell:
        return wallet, "buy"
    if is_sell and not is_buy:
        return wallet, "sell"
    return None


async def fetch_genuine_interest_stats(
    mint: str, rpc_http_url: str, window: int = DEFAULT_WINDOW, timeout_sec: float = 5.0,
) -> dict | None:
    """Returns {"total_classified", "unique_buyer_ratio", "wash_ratio"} for
    this mint's most recent `window` transactions, or None if the
    signature lookup itself failed. unique_buyer_ratio is
    distinct-buying-wallets / total-buy-transactions in the window - 1.0
    means every buy came from a different wallet (organic-looking), well
    below 1.0 means a small set of wallets are buying repeatedly.
    wash_ratio is the fraction of all distinct wallets seen that appear as
    BOTH a buyer and a seller within this same short window - the
    round-trip wash-trading signature. Both are None (not the whole
    result) if there weren't enough classified transactions to compute
    them meaningfully - never silently reported as a clean 0%/100%."""
    signatures = await _fetch_recent_signatures(mint, rpc_http_url, window, timeout_sec)
    if signatures is None:
        return None
    if not signatures:
        return {"total_classified": 0, "unique_buyer_ratio": None, "wash_ratio": None}

    results = await asyncio.gather(
        *(_classify_transaction(sig, rpc_http_url, timeout_sec) for sig in signatures)
    )
    classified = [r for r in results if r is not None]

    buy_wallets = [wallet for wallet, action in classified if action == "buy"]
    sell_wallets = [wallet for wallet, action in classified if action == "sell"]
    unique_buyers = set(buy_wallets)
    unique_sellers = set(sell_wallets)

    unique_buyer_ratio = (len(unique_buyers) / len(buy_wallets)) if buy_wallets else None

    all_wallets = unique_buyers | unique_sellers
    wash_wallets = unique_buyers & unique_sellers
    wash_ratio = (len(wash_wallets) / len(all_wallets)) if all_wallets else None

    return {
        "total_classified": len(classified),
        "unique_buyer_ratio": unique_buyer_ratio,
        "wash_ratio": wash_ratio,
    }
