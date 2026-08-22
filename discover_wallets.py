#!/usr/bin/env python3
"""
Finds real, evidence-based copytrade candidate wallets by looking at who
actually profited on the tokens this bot has bought (see
pumpfun_bot/wallet_discovery.py for the full reasoning - short version:
excludes sniper/bundler/dev-tagged wallets, looks for repeat performance
across multiple of our tokens, not a single lucky number).

Incremental: each run only queries mints not already cached, up to a
small budget-respecting cap (Birdeye's free tier is tight - see
pumpfun_bot/birdeye.py's module docstring). Run this again later to grow
the dataset further as the bot buys more tokens; it never re-queries a
mint already in data/wallet_discovery_cache.jsonl.

This is an on-demand analysis tool, not part of the live bot loop - the
free tier's budget doesn't support querying this on every buy (see
wallet_discovery.py).

Usage:
    python discover_wallets.py [--calls N]
"""
from __future__ import annotations

import argparse
import asyncio
import json

from pumpfun_bot.activity_log import DATA_LOG_PATH
from pumpfun_bot.config import load_config
from pumpfun_bot.wallet_discovery import (
    CALLS_PER_RUN_DEFAULT,
    aggregate_wallet_performance,
    fetch_and_cache_new_mints,
    load_cached_snapshots,
    repeat_candidates,
)


def _bought_mints_from_activity_log() -> list[str]:
    mints: list[str] = []
    seen: set[str] = set()
    if not DATA_LOG_PATH.exists():
        return mints
    with open(DATA_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "trade" and d.get("action") == "buy":
                mint = d.get("mint")
                if mint and mint not in seen:
                    seen.add(mint)
                    mints.append(mint)
    return mints


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls", type=int, default=CALLS_PER_RUN_DEFAULT,
        help="max new Birdeye lookups this run (each costs real, budget-limited CU)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    api_key = cfg.birdeye_movers.api_key
    if not api_key:
        raise SystemExit("BIRDEYE_API_KEY is niet gezet in .env - kan geen wallets opzoeken.")

    mints = _bought_mints_from_activity_log()
    print(f"Bekende gekochte mints (uniek): {len(mints)}")

    fetched = await fetch_and_cache_new_mints(mints, api_key, calls_per_run=args.calls)
    print(f"Nieuw opgehaald deze run: {fetched}")

    snapshots = load_cached_snapshots()
    print(f"Totaal gecachte mints: {len(snapshots)}")

    by_wallet = aggregate_wallet_performance(snapshots)
    candidates = repeat_candidates(by_wallet)

    print(
        f"\nWallets met herhaalde winst over >=2 van onze tokens "
        f"(sniper/bundler/dev uitgesloten): {len(candidates)}"
    )
    for wallet, info in candidates[:20]:
        tokens = ", ".join(f"{a['mint'][:8]}(${a['pnl']:,.0f})" for a in info["appearances"])
        print(f"  {wallet}  total_pnl=${info['total_pnl']:,.0f}  n={len(info['appearances'])}  {tokens}")


if __name__ == "__main__":
    asyncio.run(main())
