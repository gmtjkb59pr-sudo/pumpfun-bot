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
        # mints currently held via a copytrade buy - in-memory only (resets
        # on restart), since this is purely to stop mirroring a sell signal
        # for a mint we never actually bought (see the check in run() below;
        # confirmed live: watched wallets sell things this bot never bought,
        # e.g. positions opened before copytrade was enabled this session).
        self._held: set[str] = set()

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

            if action == "sell" and mint not in self._held:
                logger.debug(
                    "Sell-signaal voor %s genegeerd - geen positie via copytrade "
                    "(nooit gekocht, of al verkocht).", mint,
                )
                continue

            if action == "buy" and mint in self._held:
                # confirmed live: multiple watched wallets (or the same one
                # buying in twice) can each fire a separate buy signal for
                # the same mint within seconds - mirroring every one would
                # stack several positions' worth of exposure on a single
                # mint instead of the one capped position intended
                logger.debug(
                    "Buy-signaal voor %s genegeerd - al een positie via copytrade.", mint,
                )
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
                if action == "buy":
                    self.risk.register_trade_opened(my_amount)
                    self._held.add(mint)
                else:
                    self._held.discard(mint)
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
                    self._held.add(mint)
                else:
                    self._held.discard(mint)
                bot_state.log_trade(
                    "copytrade", action, mint, my_amount,
                    dry_run=False, tx_signature=result["signature"],
                )
                await self.alerter.send(f"✅ Copy-trade uitgevoerd | tx: {result['signature']}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Copy-trade mislukt voor %s: %s", mint, exc)
                await self.alerter.send(f"❌ Copy-trade mislukt: {exc}")
