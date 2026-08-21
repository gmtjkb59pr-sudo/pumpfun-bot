"""
Sniper: luistert naar nieuwe token-launches op pump.fun en koopt tokens die
door een paar simpele filters komen. Deze filters vangen NIET alle scams/
rugpulls - pump.fun launches zijn per ontwerp permissionless en risicovol.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..alerts import Alerter
from ..config import SniperConfig
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.sniper")


class SniperStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: SniperConfig,
        risk: RiskManager,
        alerter: Alerter,
        trade_size_sol: float,
        slippage_pct: float,
        dry_run: bool,
        outcome_tracker: OutcomeTracker | None = None,
    ):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.alerter = alerter
        self.trade_size_sol = trade_size_sol
        self.slippage_pct = slippage_pct
        self.dry_run = dry_run
        self.outcome_tracker = outcome_tracker

    def _passes_filters(self, event: dict) -> bool:
        # PumpPortal new-token events bevatten o.a. mint, name, symbol, initial
        # liquidity, en (soms) social links. Veldnamen kunnen wijzigen - check dit
        # tegen een paar live events voordat je live gaat.
        liquidity_sol = event.get("vSolInBondingCurve") or event.get("initialBuy") or 0
        if liquidity_sol and liquidity_sol < self.risk.cfg.min_liquidity_sol:
            return False

        if self.cfg.require_socials:
            has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
            if not has_socials:
                return False

        return True

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        logger.info("Sniper strategie gestart (dry_run=%s).", self.dry_run)

        async for event in self.client.stream_new_tokens():
            mint = event.get("mint")
            name = event.get("name", "?")
            symbol = event.get("symbol", "?")
            if not mint:
                continue

            if not self._passes_filters(event):
                logger.debug("Token %s (%s) afgewezen door filters.", symbol, mint)
                continue

            liquidity_sol = event.get("vSolInBondingCurve")
            ok, reason = self.risk.can_trade(self.trade_size_sol, liquidity_sol)
            if not ok:
                logger.info("Sniper: trade geblokkeerd door risk manager: %s", reason)
                continue

            msg = f"🎯 Snipe kandidaat: {name} ({symbol}) - {mint}"
            await self.alerter.send(msg)

            if self.dry_run:
                logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
                self.risk.register_trade_opened(self.trade_size_sol)
                has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
                bot_state.log_trade(
                    "sniper", "buy", mint, self.trade_size_sol, dry_run=True,
                    meta={"liquidity_sol": liquidity_sol, "has_socials": has_socials},
                )
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, extract_price_ref(event),
                        trade_size_sol=self.trade_size_sol,
                    )
                continue

            try:
                result = await self.client.build_and_send_trade(
                    action="buy",
                    mint=mint,
                    amount_sol=self.trade_size_sol,
                    slippage_pct=self.slippage_pct,
                )
                self.risk.register_trade_opened(self.trade_size_sol)
                has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
                bot_state.log_trade(
                    "sniper", "buy", mint, self.trade_size_sol,
                    dry_run=False, tx_signature=result["signature"],
                    meta={"liquidity_sol": liquidity_sol, "has_socials": has_socials},
                )
                await self.alerter.send(f"✅ Gekocht: {symbol} | tx: {result['signature']}")
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, extract_price_ref(event),
                        trade_size_sol=self.trade_size_sol,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Snipe buy mislukt voor %s: %s", mint, exc)
                await self.alerter.send(f"❌ Snipe mislukt voor {symbol}: {exc}")
