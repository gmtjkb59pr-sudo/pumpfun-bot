"""Copy-trading: spiegelt buy/sell trades van gevolgde wallets, met een eigen cap."""
from __future__ import annotations

import logging
import time

from ..alerts import Alerter
from ..config import CopyTradeConfig
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.copytrade")


class CopyTradeStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: CopyTradeConfig,
        risk: RiskManager,
        alerter: Alerter,
        max_trade_sol: float,
        slippage_pct: float,
        dry_run: bool,
    ):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.alerter = alerter
        self.max_trade_sol = max_trade_sol
        self.slippage_pct = slippage_pct
        self.dry_run = dry_run

    async def run(self) -> None:
        if not self.cfg.enabled or not self.cfg.watched_wallets:
            return
        logger.info(
            "Copy-trade strategie gestart voor %d wallet(s), dry_run=%s.",
            len(self.cfg.watched_wallets), self.dry_run,
        )

        async for event in self.client.stream_wallet_trades(self.cfg.watched_wallets):
            event_ts_ms = event.get("timestamp", time.time() * 1000)
            age_ms = time.time() * 1000 - event_ts_ms
            if age_ms > self.cfg.max_copy_delay_ms:
                logger.debug("Signaal te oud (%.0fms), overslaan.", age_ms)
                continue

            mint = event.get("mint")
            action = event.get("txType")  # "buy" of "sell"
            source_sol = event.get("solAmount", 0)
            if not mint or action not in ("buy", "sell"):
                continue

            my_amount = min(
                self.max_trade_sol,
                round(source_sol * (self.cfg.mirror_pct / 100), 4),
            )
            if my_amount <= 0:
                continue

            liquidity_sol = event.get("vSolInBondingCurve")
            ok, reason = self.risk.can_trade(my_amount, liquidity_sol)
            if not ok:
                logger.info("Copy-trade geblokkeerd door risk manager: %s", reason)
                continue

            await self.alerter.send(
                f"👥 Gevolgde wallet deed {action} op {mint} ({source_sol} SOL) "
                f"-> ik doe {my_amount} SOL"
            )

            if self.dry_run:
                logger.info("[DRY RUN] Zou %s %s SOL van %s", action, my_amount, mint)
                bot_state.log_trade("copytrade", action, mint, my_amount, dry_run=True)
                continue

            try:
                result = await self.client.build_and_send_trade(
                    action=action,
                    mint=mint,
                    amount_sol=my_amount,
                    slippage_pct=self.slippage_pct,
                )
                if action == "buy":
                    self.risk.register_trade_opened(my_amount)
                bot_state.log_trade(
                    "copytrade", action, mint, my_amount,
                    dry_run=False, tx_signature=result["signature"],
                )
                await self.alerter.send(f"✅ Copy-trade uitgevoerd | tx: {result['signature']}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Copy-trade mislukt voor %s: %s", mint, exc)
                await self.alerter.send(f"❌ Copy-trade mislukt: {exc}")
