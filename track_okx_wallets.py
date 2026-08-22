#!/usr/bin/env python3
"""
Records a new snapshot of OKX's smart-money trades feed and prints the
wallets with the best REAL accumulated track record so far - win rate
and total PnL across every scored trade recorded for that wallet across
ALL runs of this script, not just whatever happened to be positive in
this one pull (see pumpfun_bot/okx_wallet_tracker.py for the full
reasoning). Run this periodically - the more runs, the more trades
accumulate per wallet, the more meaningful the ranking becomes.

Print-only, like list_smart_money_wallets.py - copy the addresses into
config.yaml's copytrade.watched_wallets yourself.

Usage:
    python track_okx_wallets.py [--min-trades N]
"""
from __future__ import annotations

import argparse
import asyncio

from pumpfun_bot.config import load_config
from pumpfun_bot.okx_wallet_tracker import (
    MIN_TRADES_FOR_RANKING,
    aggregate_wallet_stats,
    load_cached_trades,
    record_snapshot,
    top_wallets,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-trades", type=int, default=MIN_TRADES_FOR_RANKING,
        help="minimum scored trades a wallet needs before it's ranked (default: %(default)s)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    if not (cfg.okx_api_key and cfg.okx_secret_key and cfg.okx_passphrase):
        raise SystemExit(
            "OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE zijn niet allemaal gezet in .env."
        )

    new_count = await record_snapshot(cfg.okx_api_key, cfg.okx_secret_key, cfg.okx_passphrase)
    print(f"Nieuwe trades deze run: {new_count}")

    trades = load_cached_trades()
    print(f"Totaal gecachte (gededupliceerde) trades: {len(trades)}")

    by_wallet = aggregate_wallet_stats(trades)
    ranked = top_wallets(by_wallet, min_trades=args.min_trades)

    print(f"\nWallets met >= {args.min_trades} gescoorde trades, gerangschikt op win rate dan PnL:")
    for wallet, info in ranked[:25]:
        print(
            f"  {wallet}  win_rate={info['win_rate_pct']}%  "
            f"total_pnl=${info['total_pnl_usd']:,.0f}  n={info['scored_count']}/{info['trade_count']}"
        )
    if not ranked:
        print("  (nog geen wallets met genoeg gescoorde trades - draai dit script later nog eens.)")


if __name__ == "__main__":
    asyncio.run(main())
