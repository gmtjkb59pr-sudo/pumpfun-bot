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
from pumpfun_bot.auto_tuner import AutoTuner
from pumpfun_bot.balance_watch import (
    BalanceFloorReached,
    MaxRealLossReached,
    fetch_sol_balance,
    fetch_sol_usd_price,
    watch_balance_floor,
    watch_max_real_loss,
)
from pumpfun_bot.candidate_price_tracker import CandidatePriceTracker
from pumpfun_bot.config import load_config
from pumpfun_bot.dashboard_server import start_dashboard_server
from pumpfun_bot.logger_setup import setup_logging
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.pumpportal_client import PumpPortalClient
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.scaled_exit_simulator import ScaledExitSimulator
from pumpfun_bot.state import bot_state
from pumpfun_bot.strategies.birdeye_movers import BirdeyeMoversStrategy
from pumpfun_bot.strategies.coingecko_movers import CoinGeckoMoversStrategy
from pumpfun_bot.strategies.moonshot_hunter import MoonshotHunterStrategy
from pumpfun_bot.strategies.copytrade import CopyTradeStrategy
from pumpfun_bot.strategies.market_maker import MarketMakerStrategy
from pumpfun_bot.strategies.sniper import SniperStrategy
from pumpfun_bot.strategies.social_watch import SocialWatchStrategy


REAL_PNL_POLL_INTERVAL_SEC = 30.0


async def _track_real_balance_loop(wallet_pubkey: str, rpc_http_url: str) -> None:
    """Keeps bot_state's real-wallet-vs-session-start P&L up to date, so the
    dashboard shows what actually happened to the wallet - not the bot's
    own per-trade fee model (realized_pnl_sol), which doesn't account for
    real slippage or fees on failed attempts. Confirmed live: the modeled
    P&L read positive while the real wallet balance dropped far more in the
    same window - this is the ground truth check for that gap."""
    while True:
        balance_sol = await fetch_sol_balance(wallet_pubkey, rpc_http_url)
        if balance_sol is not None:
            price_usd = await fetch_sol_usd_price()
            balance_usd = balance_sol * price_usd if price_usd is not None else None
            bot_state.update_real_balance(balance_sol, balance_usd)
        await asyncio.sleep(REAL_PNL_POLL_INTERVAL_SEC)


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
    alerter = Alerter(
        console=cfg.alerts_console,
        telegram_enabled=cfg.telegram_enabled,
        bot_token=cfg.telegram_bot_token,
        chat_id=cfg.telegram_chat_id,
    )
    risk = RiskManager(cfg.risk, alerter=alerter)

    # user-requested "scaled exit" strategy comparison (take half off at
    # +100%, trail the remainder 25% off its peak, wider -50% stop before
    # any partial take) - runs purely as an observer on OutcomeTracker's
    # real price ticks, never affects the actual exit decision (see
    # scaled_exit_simulator.py)
    scaled_exit_simulator = ScaledExitSimulator()

    outcome_tracker = OutcomeTracker(
        ws_url=cfg.pumpportal_ws_url,
        api_key=cfg.pumpportal_api_key,
        risk=risk,
        alerter=alerter,
        take_profit_pct=cfg.sniper.take_profit_pct,
        stop_loss_pct=cfg.sniper.stop_loss_pct,
        trailing_activation_pct=cfg.sniper.trailing_activation_pct,
        trailing_stop_pct=cfg.sniper.trailing_stop_pct,
        client=client,
        dry_run=cfg.risk.dry_run,
        sell_slippage_pct=cfg.risk.default_slippage_pct,
        price_observers=[scaled_exit_simulator.on_price_update],
    )
    outcome_tracker.load_pending()
    # wallet<->tracked-positions reconciliation (both directions: stale
    # tracked positions the wallet no longer holds, and untracked holdings
    # the wallet has) now runs periodically inside outcome_tracker.run()
    # itself (see OutcomeTracker._reconcile_with_wallet), firing immediately
    # on its first cycle and then every WALLET_RECONCILE_INTERVAL_SEC after -
    # covers the startup check this used to be, plus ongoing drift.

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
    # user-requested: real 1m/2m momentum tracked from live trade ticks,
    # shorter than anything DexScreener's API exposes (see
    # candidate_price_tracker.py) - explicitly accepted despite the real
    # PumpPortal subscribeTokenTrade metering cost of watching every
    # candidate, not just ones that end up bought
    candidate_price_tracker = CandidatePriceTracker(
        ws_url=cfg.pumpportal_ws_url, api_key=cfg.pumpportal_api_key,
    )
    social_watch = SocialWatchStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.social_watch,
        risk=risk,
        alerter=alerter,
        trade_size_sol=cfg.social_watch.trade_size_sol or cfg.risk.max_sol_per_trade,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
        outcome_tracker=outcome_tracker,
        price_tracker=candidate_price_tracker,
        scaled_exit_simulator=scaled_exit_simulator,
    )
    # user-requested: discovers already-existing tokens with a real volume/
    # price spike via Birdeye's trending API - social_watch can only ever
    # see brand-new launches, this covers the gap (see birdeye_movers.py)
    birdeye_movers = BirdeyeMoversStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.birdeye_movers,
        risk=risk,
        alerter=alerter,
        trade_size_sol=cfg.birdeye_movers.trade_size_sol or cfg.risk.max_sol_per_trade,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
        outcome_tracker=outcome_tracker,
    )
    # user-requested: same discovery niche as birdeye_movers, but via
    # CoinGecko's free Demo API plan - much higher call budget lets this
    # poll every few minutes instead of every 45 and react to a genuinely
    # short momentum window (see coingecko_movers.py)
    coingecko_movers = CoinGeckoMoversStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.coingecko_movers,
        risk=risk,
        alerter=alerter,
        trade_size_sol=cfg.coingecko_movers.trade_size_sol or cfg.risk.max_sol_per_trade,
        slippage_pct=cfg.risk.default_slippage_pct,
        dry_run=cfg.risk.dry_run,
        outcome_tracker=outcome_tracker,
    )
    # user-requested: a deliberately different bet from every strategy
    # above - wide stop-loss, a ladder that only takes small profit at
    # huge multiples, and a hold time measured in days/weeks instead of
    # minutes, aiming at the rare 100-1000x outlier (see moonshot_hunter.py
    # and MoonshotHunterConfig's docstring in config.py for the full
    # reasoning, including the honest caveat that this has no proven edge
    # like social_watch's evidence-based filters).
    moonshot_hunter = MoonshotHunterStrategy(
        client=PumpPortalClient(cfg.pumpportal_ws_url, cfg.pumpportal_trade_api_url,
                                 cfg.rpc_http_url, keypair, api_key=cfg.pumpportal_api_key),
        cfg=cfg.moonshot_hunter,
        risk=risk,
        alerter=alerter,
        trade_size_sol=cfg.moonshot_hunter.trade_size_sol or cfg.risk.max_sol_per_trade,
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
        max_trade_sol=cfg.copytrade.max_trade_sol or cfg.risk.max_sol_per_trade,
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

    enabled = [
        s.cfg.enabled for s in
        (sniper, social_watch, birdeye_movers, coingecko_movers, moonshot_hunter, copytrade, market_maker)
    ]
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
            "social_watch": cfg.social_watch.enabled,
            "birdeye_movers": cfg.birdeye_movers.enabled,
            "coingecko_movers": cfg.coingecko_movers.enabled,
            "moonshot_hunter": cfg.moonshot_hunter.enabled,
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

    if not cfg.risk.dry_run:
        # ground-truth baseline for real_pnl_* - best-effort, doesn't block
        # startup if the lookup fails (real_pnl_* just stays unset until a
        # later periodic check succeeds)
        start_balance_sol = await fetch_sol_balance(str(keypair.pubkey()), cfg.rpc_http_url)
        if start_balance_sol is not None:
            start_price_usd = await fetch_sol_usd_price()
            start_balance_usd = (
                start_balance_sol * start_price_usd if start_price_usd is not None else None
            )
            bot_state.set_session_start_balance(start_balance_sol, start_balance_usd)
            logger.info(
                "Sessie-startbalans vastgelegd: %.4f SOL (~$%s) - dashboard toont echte "
                "P&L t.o.v. dit punt, los van het eigen fee-model.",
                start_balance_sol, f"{start_balance_usd:.2f}" if start_balance_usd else "?",
            )

    await alerter.send("🤖 Pump.fun bot gestart.")

    tasks = [
        asyncio.create_task(sniper.run()),
        asyncio.create_task(social_watch.run()),
        asyncio.create_task(birdeye_movers.run()),
        asyncio.create_task(coingecko_movers.run()),
        asyncio.create_task(moonshot_hunter.run()),
        asyncio.create_task(copytrade.run()),
        asyncio.create_task(market_maker.run()),
        asyncio.create_task(outcome_tracker.run()),
    ]

    if cfg.social_watch.enabled:
        tasks.append(asyncio.create_task(candidate_price_tracker.run()))
        tasks.append(asyncio.create_task(scaled_exit_simulator.run()))

    if not cfg.risk.dry_run:
        tasks.append(asyncio.create_task(
            _track_real_balance_loop(str(keypair.pubkey()), cfg.rpc_http_url)
        ))

    if cfg.sniper.enabled or cfg.social_watch.enabled or cfg.birdeye_movers.enabled:
        auto_tuner = AutoTuner(
            sniper_cfg=cfg.sniper, risk=risk, alerter=alerter, outcome_tracker=outcome_tracker,
            social_watch_cfg=cfg.social_watch, birdeye_movers_cfg=cfg.birdeye_movers,
        )
        logger.info(
            "Auto-tuner gestart: past sniper/social_watch/birdeye_movers-filters aan op basis "
            "van outcome-stats (alleen strenger, nooit losser, zie pumpfun_bot/auto_tuner.py)."
        )
        tasks.append(asyncio.create_task(auto_tuner.run()))

    if not cfg.risk.dry_run and cfg.risk.min_wallet_balance_usd > 0:
        logger.info(
            "Balance-floor kill-switch actief: bot stopt zichzelf als de wallet onder "
            "$%.2f zakt.", cfg.risk.min_wallet_balance_usd,
        )
        tasks.append(asyncio.create_task(
            watch_balance_floor(
                wallet_pubkey=str(keypair.pubkey()),
                rpc_http_url=cfg.rpc_http_url,
                min_balance_usd=cfg.risk.min_wallet_balance_usd,
                alerter=alerter,
            )
        ))

    if not cfg.risk.dry_run and cfg.risk.max_real_loss_usd > 0:
        logger.info(
            "Max-verlies kill-switch actief: bot stopt zichzelf als het ECHTE verlies "
            "deze sessie $%.2f bereikt.", cfg.risk.max_real_loss_usd,
        )
        tasks.append(asyncio.create_task(
            watch_max_real_loss(max_loss_usd=cfg.risk.max_real_loss_usd, alerter=alerter)
        ))

    try:
        await asyncio.gather(*tasks)
    except (BalanceFloorReached, MaxRealLossReached):
        # user-requested hard stop, not a bug - cancel everything else so
        # the process actually exits instead of leaving orphaned tasks
        # (buys/sells) running after the "stopped" alert already went out.
        # cancel() on an already-finished task (e.g. whichever one raised)
        # is a documented no-op, so no need to track which one it was.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot gestopt door gebruiker.")
