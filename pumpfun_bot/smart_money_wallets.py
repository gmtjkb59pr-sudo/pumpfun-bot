"""
Extracts a clean list of "smart money" wallet addresses from OKX's
address-tracker trades feed (see okx_client.py), for use as
copytrade.watched_wallets - the existing CopyTradeStrategy already mirrors
buy/sell events from watched wallets in real time via PumpPortal's
subscribeAccountTrade WS stream, so this doesn't need its own live polling
loop or trading logic, just a periodic wallet-LIST refresh.

OKX's smart-money trades endpoint returns individual trades, not a clean
wallet list - unique wallet addresses are extracted here, filtered to
wallets whose most recent trade in this pull had a positive realizedPnlUsd
(a coarse, cheap sanity check - not the same rigor as wallet_discovery.py's
cross-token repeat-performance analysis, since OKX already curates this
list itself; this just avoids copytrading a wallet whose latest move
happened to be a loss).
"""
from __future__ import annotations

import logging

from .okx_client import TRACKER_TYPE_SMART_MONEY, fetch_address_tracker_trades

logger = logging.getLogger("pumpfun_bot.smart_money_wallets")


def extract_profitable_wallets(trades: list[dict]) -> list[str]:
    """Returns unique wallet addresses from `trades` whose realizedPnlUsd on
    that trade was positive - never guesses at a missing/unparseable PnL
    value being "good", only counts a real positive number."""
    seen: set[str] = set()
    wallets: list[str] = []
    for trade in trades:
        wallet = trade.get("walletAddress")
        if not wallet or wallet in seen:
            continue
        pnl_raw = trade.get("realizedPnlUsd")
        try:
            pnl = float(pnl_raw)
        except (TypeError, ValueError):
            continue
        if pnl > 0:
            seen.add(wallet)
            wallets.append(wallet)
    return wallets


async def fetch_smart_money_wallets(
    api_key: str, secret_key: str, passphrase: str, chain_index: str = "501",
) -> list[str] | None:
    """Returns a list of unique, currently-profitable smart-money wallet
    addresses on `chain_index` (Solana by default), or None if the
    underlying lookup failed."""
    trades = await fetch_address_tracker_trades(
        api_key, secret_key, passphrase,
        tracker_type=TRACKER_TYPE_SMART_MONEY, chain_index=chain_index,
    )
    if trades is None:
        return None
    return extract_profitable_wallets(trades)
