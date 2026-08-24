"""
Detects external SOL transfers (deposits/withdrawals) to/from the wallet
that are NOT the bot's own trades, so the dashboard's real_pnl_sol
(wallet-balance-delta based) isn't silently distorted by a manual top-up
or withdrawal landing mid-session.

Real finding 2026-08-24, user-reported ("the actual profit is not right
because i topped up the bot with 10 dollar it is actually down 3"):
real_pnl_sol was computed as current_balance - session_start_balance with
no way to know a deposit had landed in between - a $10 top-up 23 seconds
after session start showed up entirely as "trading profit" (+$8.48
believed vs. a real -$3.88 once the deposit and buy-side slippage were
both accounted for).

Distinguishes "ours" from "external" by checking each new signature
against the bot's own logged trades (every buy and exit gets a
tx_signature written to data/activity_log.jsonl) - anything touching the
wallet's balance that ISN'T in that set is external.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aiohttp

from .activity_log import DATA_LOG_PATH
from .state import bot_state

logger = logging.getLogger("pumpfun_bot.external_transfer_watch")

POLL_INTERVAL_SEC = 60.0


def _load_own_signatures(activity_log_path: Path, since_ts: float) -> set[str]:
    """Every tx_signature the bot itself submitted this session (buys and
    exits both get one logged) - resolved fresh on every poll rather than
    cached, so a signature logged moments ago (after the last poll) is
    already excluded correctly."""
    signatures: set[str] = set()
    try:
        with activity_log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("ts", 0) < since_ts:
                    continue
                sig = record.get("tx_signature")
                if sig:
                    signatures.add(sig)
    except FileNotFoundError:
        pass
    return signatures


async def _fetch_signatures_for_address(
    session: aiohttp.ClientSession, rpc_http_url: str, wallet_pubkey: str,
    until: str | None, limit: int = 25,
) -> list[dict]:
    params: dict = {"limit": limit}
    if until:
        params["until"] = until
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
        "params": [wallet_pubkey, params],
    }
    try:
        async with session.post(
            rpc_http_url, json=payload, timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            data = await resp.json()
    except Exception:  # noqa: BLE001
        return []
    return data.get("result") or []


async def _fetch_wallet_delta(
    session: aiohttp.ClientSession, rpc_http_url: str, wallet_pubkey: str, signature: str,
) -> float | None:
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
        "params": [
            signature,
            {"encoding": "json", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"},
        ],
    }
    try:
        async with session.post(
            rpc_http_url, json=payload, timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            data = await resp.json()
    except Exception:  # noqa: BLE001
        return None
    result = data.get("result")
    if not result:
        return None
    try:
        keys = result["transaction"]["message"]["accountKeys"]
        idx = keys.index(wallet_pubkey)
        meta = result["meta"]
        return (meta["postBalances"][idx] - meta["preBalances"][idx]) / 1_000_000_000
    except (KeyError, ValueError, IndexError):
        return None


async def check_for_external_transfers(
    wallet_pubkey: str, rpc_http_url: str, session_start_ts: float, last_seen_signature: str | None,
    session: aiohttp.ClientSession,
) -> str | None:
    """One poll cycle - pure enough to unit test without a real loop.
    Returns the newest signature seen (the next call's `last_seen_signature`
    cursor), or the same one passed in in unchanged if nothing new landed."""
    entries = await _fetch_signatures_for_address(
        session, rpc_http_url, wallet_pubkey, until=last_seen_signature,
    )
    if not entries:
        return last_seen_signature

    own_signatures = _load_own_signatures(DATA_LOG_PATH, session_start_ts)
    # oldest-first, so a later re-run's `until` cursor ends on the newest
    for entry in reversed(entries):
        sig = entry.get("signature")
        block_time = entry.get("blockTime")
        if not sig or entry.get("err") is not None:
            continue
        if block_time is not None and block_time < session_start_ts:
            continue
        if sig in own_signatures:
            continue
        delta = await _fetch_wallet_delta(session, rpc_http_url, wallet_pubkey, sig)
        if delta is None or delta == 0:
            continue
        bot_state.add_external_transfer(delta)
        logger.warning(
            "Externe SOL-overboeking gedetecteerd (niet van de bot): %+.6f SOL (tx: %s) - "
            "uitgesloten van real_pnl_sol.", delta, sig,
        )

    return entries[0].get("signature") or last_seen_signature


async def watch_external_transfers(
    wallet_pubkey: str, rpc_http_url: str, session_start_ts: float,
    poll_interval_sec: float = POLL_INTERVAL_SEC,
) -> None:
    """Background loop: periodically scans the wallet's recent on-chain
    signatures, and for any signature that ISN'T one of the bot's own
    logged trades, treats its balance delta as an external transfer."""
    last_seen_signature: str | None = None
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.sleep(poll_interval_sec)
            try:
                last_seen_signature = await check_for_external_transfers(
                    wallet_pubkey, rpc_http_url, session_start_ts, last_seen_signature, session,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Kon externe overboekingen niet checken.")
