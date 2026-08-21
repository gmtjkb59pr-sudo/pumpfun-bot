#!/usr/bin/env python3
"""
Entrypoint voor de pump.fun trading bot.

Start alleen de strategieën die in config.yaml op enabled: true staan.
Standaard staat risk.dry_run: true - de bot logt dan wat hij ZOU doen zonder
echte orders te plaatsen. Zet dit pas op false als je het gedrag hebt
gecontroleerd en de risicolimieten bewust hebt ingesteld.

Gebruik:
    python main.py
"""
from __future__ import annotations

import asyncio
import base58
from solders.keypair import Keypair

from pumpfun_bot.alerts import Alerter
from pumpfun_bot.config import load_config
from pumpfun_bot.dashboard_server import start_dashboard_server
from pumpfun_bot.logger_setup import setup_logging
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.pumpportal_client import PumpPortalClient
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.state import bot_state
from pumpfun_bot.strategies.copytrade import CopyTradeStrategy
from pumpfun_bot.strategies.market_maker import MarketMakerStrategy
from pumpfun_bot.strategies.sniper import SniperStrategy


async def main() -> None:
    logger = setup_logging()
    cfg = load_config("config.yaml")

    if cfg.risk.dry_run:
        logger.warning("=== DRY RUN MODUS: er worden GEEN echte orders geplaatst ===")
    else:
        logger.warning(
            "=== LIVE MODUS: er worden ECHTE orders geplaatst met echt geld. "
            "Zorg dat je risicolimieten in config.yaml kloppen. ==="
        )

    try:
        keypair = Keypair.from_bytes(base58.b58decode(cfg.private_key))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Kon private key niet laden (verwacht base58-formaat): {exc}"
        ) from exc

    logger.info("Wallet geladen: %s", keypair.pubkey())

    client = PumpPortalClient(
        ws_url=cfg.pumpportal_ws_url,
        trade_api_url=cfg.pumpportal_trade_api_url,
        rpc_http_url=cfg.rpc_http_url,
        keypair=keypair,
        api_key=cfg.pumpportal_api_key,
    )
    risk = RiskManager(cfg.risk)
    alerter = Alerter(
        console=cfg.alerts_console,
        telegram_enabled=cfg.telegram_enabled,
        bot_token=cfg.telegram_bot_token,
        chat_id=cfg.telegram_chat_id,
    )

    outcome_tracker = OutcomeTracker(ws_url=cfg.pumpportal_ws_url, api_key=cfg.pumpportal_api_key)

    sniper = SniperStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.sniper,
        risk=risk,
        alerter=alerter,
        trade_size_sol=cfg.risk.max_sol_per_trade,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
        outcome_tracker=outcome_tracker,
    )
    copytrade = CopyTradeStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.copytrade,
        risk=risk,
        alerter=alerter,
        max_trade_sol=cfg.risk.max_sol_per_trade,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
    )
    market_maker = MarketMakerStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.market_maker,
        risk=risk,
        alerter=alerter,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
    )

    enabled = [s.cfg.enabled for s in (sniper, copytrade, market_maker)]
    if not any(enabled):
        logger.warning(
            "Geen enkele strategie staat op enabled: true in config.yaml. "
            "De bot doet nu niets - pas config.yaml aan."
        )
        return

    bot_state.init_meta(
        dry_run=cfg.risk.dry_run,
        wallet_pubkey=str(keypair.pubkey()),
        strategies_enabled={
            "sniper": cfg.sniper.enabled,
            "copytrade": cfg.copytrade.enabled,
            "market_maker": cfg.market_maker.enabled,
        },
        max_trades_per_hour=cfg.risk.max_trades_per_hour,
        max_daily_loss_sol=cfg.risk.max_daily_loss_sol,
        max_sol_total_exposure=cfg.risk.max_sol_total_exposure,
    )

    if cfg.dashboard_enabled:
        start_dashboard_server(cfg.dashboard_port)
        logger.info(
            "Dashboard: open http://localhost:%d in je browser om de bot te volgen.",
            cfg.dashboard_port,
        )

    await alerter.send("🤖 Pump.fun bot gestart.")

    tasks = [
        asyncio.create_task(sniper.run()),
        asyncio.create_task(copytrade.run()),
        asyncio.create_task(market_maker.run()),
        asyncio.create_task(outcome_tracker.run()),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot gestopt door gebruiker.")
