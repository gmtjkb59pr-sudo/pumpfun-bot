"""
Tracks OKX's "smart money" wallet feed over time, instead of judging a
wallet from a single snapshot's coarse "positive PnL on whichever trade
happened to be in this pull" check (see smart_money_wallets.py - that's
what list_smart_money_wallets.py uses today, a fine quick-start filter
but not a real track record: a wallet could show up with one lucky trade
and nine losers that just weren't in that particular pull).

Each run appends whatever new trades OKX's feed returns to a local
cache, deduped by txHash (the feed returns overlapping recent trades on
every poll, not a fixed page), so real per-wallet performance - win
rate, total realized PnL, sample size - accumulates across many runs
instead of being judged from one. Meant to be run periodically (see
track_okx_wallets.py) - the longer this runs, the more trades accumulate
per wallet, and the more meaningful "best wallets" becomes. Mirrors
wallet_discovery.py's cache-and-aggregate shape.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .okx_client import TRACKER_TYPE_SMART_MONEY, fetch_address_tracker_trades

logger = logging.getLogger("pumpfun_bot.okx_wallet_tracker")

OKX_WALLET_TRACKER_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "okx_wallet_tracker_cache.jsonl"

# a wallet's win rate/PnL is only meaningful once enough scored trades have
# accumulated for it - one or two trades could easily just be luck
MIN_TRADES_FOR_RANKING = 3


def load_cached_trades() -> dict[str, dict]:
    """Returns {txHash: trade_dict} for every trade ever recorded across
    all past runs - deduped by txHash, since OKX's feed returns
    overlapping recent trades on every poll, not a fixed page."""
    trades: dict[str, dict] = {}
    if not OKX_WALLET_TRACKER_CACHE_PATH.exists():
        return trades
    with open(OKX_WALLET_TRACKER_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tx_hash = record.get("txHash")
            if tx_hash:
                trades[tx_hash] = record
    return trades


def _append_trades(trades: list[dict]) -> None:
    OKX_WALLET_TRACKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OKX_WALLET_TRACKER_CACHE_PATH, "a", encoding="utf-8") as f:
        for trade in trades:
            f.write(json.dumps(trade) + "\n")


async def record_snapshot(
    api_key: str, secret_key: str, passphrase: str, chain_index: str = "501",
) -> int:
    """Pulls OKX's current smart-money trades feed and appends whatever
    isn't already cached (matched by txHash). Returns how many trades
    were genuinely new this run, or 0 if the fetch itself failed - never
    guesses at "new" for a trade without a txHash, since that can't be
    deduped and would otherwise risk being double-counted forever."""
    trades = await fetch_address_tracker_trades(
        api_key, secret_key, passphrase,
        tracker_type=TRACKER_TYPE_SMART_MONEY, chain_index=chain_index,
    )
    if trades is None:
        logger.debug("Kon geen OKX smart-money trades ophalen voor deze snapshot.")
        return 0
    cached = load_cached_trades()
    new_trades = [t for t in trades if t.get("txHash") and t["txHash"] not in cached]
    if new_trades:
        _append_trades(new_trades)
    return len(new_trades)


def aggregate_wallet_stats(trades: dict[str, dict]) -> dict[str, dict]:
    """Combines every cached (deduped) trade into a per-wallet real track
    record: trade_count (all recorded trades), scored_count (those with a
    parseable realizedPnlUsd), win_count, win_rate_pct, and total realized
    PnL. Never guesses at a missing/unparseable PnL being a win or a loss
    - it's excluded from scored_count/win_count/total_pnl_usd entirely,
    same discipline as smart_money_wallets.extract_profitable_wallets."""
    by_wallet: dict[str, dict] = {}
    for trade in trades.values():
        wallet = trade.get("walletAddress")
        if not wallet:
            continue
        entry = by_wallet.setdefault(wallet, {
            "trade_count": 0, "scored_count": 0, "win_count": 0, "total_pnl_usd": 0.0,
        })
        entry["trade_count"] += 1
        pnl_raw = trade.get("realizedPnlUsd")
        try:
            pnl = float(pnl_raw)
        except (TypeError, ValueError):
            continue
        entry["scored_count"] += 1
        entry["total_pnl_usd"] += pnl
        if pnl > 0:
            entry["win_count"] += 1
    for entry in by_wallet.values():
        entry["win_rate_pct"] = (
            round(100 * entry["win_count"] / entry["scored_count"], 1)
            if entry["scored_count"] else None
        )
    return by_wallet


def top_wallets(
    by_wallet: dict[str, dict], min_trades: int = MIN_TRADES_FOR_RANKING,
) -> list[tuple[str, dict]]:
    """Wallets with at least min_trades SCORED trades, ranked by win rate
    then total PnL descending - the actual "best of OKX's list" output,
    grounded in accumulated real track record instead of one snapshot's
    coarse positive/negative check. A wallet below min_trades is excluded
    entirely rather than ranked on too small a sample to mean anything.
    A wallet with zero scored trades is never ranked regardless of
    min_trades - there's no win_rate_pct to sort it by at all."""
    candidates = [
        (wallet, info) for wallet, info in by_wallet.items()
        if info["scored_count"] > 0 and info["scored_count"] >= min_trades
    ]
    candidates.sort(key=lambda item: (-(item[1]["win_rate_pct"] or 0), -item[1]["total_pnl_usd"]))
    return candidates
