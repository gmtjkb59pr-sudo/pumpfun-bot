#!/usr/bin/env python3
"""
Prints OKX's currently-profitable "smart money" wallet addresses on
Solana, for use as copytrade.watched_wallets in config.yaml (see
pumpfun_bot/smart_money_wallets.py).

This is deliberately a print-only tool, not an auto-editor of config.yaml
- config.yaml carries a lot of hand-written, hand-maintained comments
tonight, and a script-driven rewrite risks clobbering them. Copy the
addresses it prints into config.yaml's copytrade.watched_wallets list
yourself (or ask Claude to do it).

Usage:
    python list_smart_money_wallets.py
"""
from __future__ import annotations

import asyncio

from pumpfun_bot.config import load_config
from pumpfun_bot.smart_money_wallets import fetch_smart_money_wallets


async def main() -> None:
    cfg = load_config("config.yaml")
    if not (cfg.okx_api_key and cfg.okx_secret_key and cfg.okx_passphrase):
        raise SystemExit(
            "OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE zijn niet allemaal gezet in .env."
        )

    wallets = await fetch_smart_money_wallets(cfg.okx_api_key, cfg.okx_secret_key, cfg.okx_passphrase)
    if wallets is None:
        raise SystemExit("Kon geen smart-money wallets ophalen van OKX (zie logs).")

    print(f"{len(wallets)} winstgevende smart-money wallets (Solana):\n")
    for w in wallets:
        print(f'      - "{w}"')


if __name__ == "__main__":
    asyncio.run(main())
