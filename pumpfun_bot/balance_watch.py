"""
Hard kill-switch: if the wallet's real SOL balance (converted to USD via a
live price) drops below a user-set floor, stop the whole bot immediately -
not a per-trade risk limit like RiskConfig's other fields, a "we're nearly
out of runway, stop before it gets to zero" safety net. User-requested
directly, after a live session saw the wallet's usable balance repeatedly
run down to near-zero.

Fails safe: if either the balance lookup or the price lookup itself fails,
that check is skipped, not treated as "below floor" - a broken RPC or price
feed must never be read as a reason to shut down an otherwise-working bot.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from .state import bot_state

logger = logging.getLogger("pumpfun_bot.balance_watch")

SOL_USD_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"


class BalanceFloorReached(Exception):
    """Raised to stop the bot once the wallet balance is confirmed below
    the configured USD floor - propagates up through main()'s
    asyncio.gather to end the whole process."""


class MaxRealLossReached(Exception):
    """Raised to stop the bot once the REAL (ground-truth wallet, not the
    bot's own fee-model estimate) session loss has reached the configured
    USD limit - propagates up through main()'s asyncio.gather the same way
    as BalanceFloorReached, but answers a different question: "stop after
    losing $X" rather than "stop once down to $Y"."""


async def fetch_sol_balance(wallet_pubkey: str, rpc_http_url: str, timeout_sec: float = 10.0) -> float | None:
    """Returns the wallet's current SOL balance, or None if the lookup
    itself failed."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_pubkey]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_http_url, json=payload, timeout=timeout_sec) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.debug("getBalance fout: %s", data["error"])
                    return None
                lamports = (data.get("result") or {}).get("value")
                if lamports is None:
                    return None
                return lamports / 1_000_000_000
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon SOL balance niet ophalen: %s", exc)
        return None


async def fetch_sol_usd_price(timeout_sec: float = 10.0) -> float | None:
    """Returns the current SOL/USD price, or None if the lookup itself
    failed."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(SOL_USD_PRICE_URL, timeout=timeout_sec) as resp:
                data = await resp.json()
                price = (data.get("solana") or {}).get("usd")
                return float(price) if price is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon SOL/USD prijs niet ophalen: %s", exc)
        return None


async def watch_balance_floor(
    wallet_pubkey: str,
    rpc_http_url: str,
    min_balance_usd: float,
    alerter=None,
    poll_interval_sec: float = 60.0,
) -> None:
    """Runs forever, checking the wallet's USD-converted balance every
    poll_interval_sec. Raises BalanceFloorReached the first time it's
    confirmed below min_balance_usd - a single failed lookup (balance or
    price) is skipped for that cycle, not treated as a trigger."""
    while True:
        await asyncio.sleep(poll_interval_sec)
        balance_sol = await fetch_sol_balance(wallet_pubkey, rpc_http_url)
        if balance_sol is None:
            continue
        sol_usd_price = await fetch_sol_usd_price()
        if sol_usd_price is None:
            continue
        balance_usd = balance_sol * sol_usd_price
        if balance_usd < min_balance_usd:
            message = (
                f"🛑 Wallet balance (${balance_usd:.2f}, {balance_sol:.4f} SOL @ "
                f"${sol_usd_price:.2f}/SOL) onder de ingestelde ondergrens van "
                f"${min_balance_usd:.2f} - bot stopt nu."
            )
            logger.error(message)
            if alerter is not None:
                await alerter.send(message)
            raise BalanceFloorReached(message)


async def watch_max_real_loss(
    max_loss_usd: float, alerter=None, poll_interval_sec: float = 30.0,
) -> None:
    """Runs forever, checking bot_state.real_pnl_usd (kept up to date
    elsewhere by main.py's periodic real-balance tracker) every
    poll_interval_sec. Raises MaxRealLossReached the first time the REAL
    loss reaches max_loss_usd. A missing real_pnl_usd (no successful
    balance/price check has landed yet) is skipped, not a trigger - same
    fail-safe stance as the rest of this module."""
    while True:
        await asyncio.sleep(poll_interval_sec)
        real_pnl_usd = bot_state.real_pnl_usd
        if real_pnl_usd is None:
            continue
        if real_pnl_usd <= -abs(max_loss_usd):
            message = (
                f"🛑 Echt verlies deze sessie (${real_pnl_usd:.2f}) heeft de ingestelde "
                f"limiet van -${max_loss_usd:.2f} bereikt - bot stopt nu."
            )
            logger.error(message)
            if alerter is not None:
                await alerter.send(message)
            raise MaxRealLossReached(message)
