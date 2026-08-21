"""
Sniper: luistert naar nieuwe token-launches op pump.fun en koopt tokens die
door een paar simpele filters komen. Deze filters vangen NIET alle scams/
rugpulls - pump.fun launches zijn per ontwerp permissionless en risicovol.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..activity_log import DATA_LOG_PATH
from ..alerts import Alerter
from ..config import SniperConfig
from ..holder_count import record_holder_count
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state
from ..wallet_reputation import blocked_wallets

logger = logging.getLogger("pumpfun_bot.sniper")

WALLET_BLOCKLIST_REFRESH_SEC = 60


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
        # wallets that our own trade history shows repeatedly launch losing
        # tokens - see wallet_reputation.py. Only ever grows (tighten-only,
        # same philosophy as auto_tuner.py), refreshed periodically below.
        self.blocked_wallets: set[str] = set()

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

        creator = event.get("traderPublicKey")
        if creator and creator in self.blocked_wallets:
            return False

        mint = event.get("mint")
        if (
            mint and self.outcome_tracker is not None
            and self.outcome_tracker.is_tracking(mint)
        ):
            # already held (e.g. social_watch bought this same mint) -
            # buying again would spend real SOL on a position nothing would
            # ever track or exit, since OutcomeTracker keys by mint alone
            return False

        return True

    async def _refresh_blocked_wallets_loop(self) -> None:
        while True:
            await asyncio.sleep(WALLET_BLOCKLIST_REFRESH_SEC)
            newly_blocked = blocked_wallets(DATA_LOG_PATH) - self.blocked_wallets
            if newly_blocked:
                self.blocked_wallets |= newly_blocked
                message = (
                    f"🧠 Wallet-reputatie: {len(newly_blocked)} launcher-wallet(s) geblokkeerd "
                    f"na herhaalde verliezen: {', '.join(sorted(newly_blocked))}"
                )
                logger.warning(message)
                await self.alerter.send(message)

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        logger.info("Sniper strategie gestart (dry_run=%s).", self.dry_run)
        asyncio.create_task(self._refresh_blocked_wallets_loop())

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

            creator = event.get("traderPublicKey")
            if self.dry_run:
                logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
                self.risk.register_trade_opened(self.trade_size_sol)
                has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
                bot_state.log_trade(
                    "sniper", "buy", mint, self.trade_size_sol, dry_run=True,
                    meta={"liquidity_sol": liquidity_sol, "has_socials": has_socials, "creator": creator},
                )
                asyncio.create_task(record_holder_count(mint, self.client.rpc_http_url))
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, extract_price_ref(event),
                        trade_size_sol=self.trade_size_sol,
                        take_profit_pct=self.cfg.take_profit_pct,
                        stop_loss_pct=self.cfg.stop_loss_pct,
                        trailing_activation_pct=self.cfg.trailing_activation_pct,
                        trailing_stop_pct=self.cfg.trailing_stop_pct,
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
                    meta={"liquidity_sol": liquidity_sol, "has_socials": has_socials, "creator": creator},
                )
                asyncio.create_task(record_holder_count(mint, self.client.rpc_http_url))
                await self.alerter.send(f"✅ Gekocht: {symbol} | tx: {result['signature']}")
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, extract_price_ref(event),
                        trade_size_sol=self.trade_size_sol,
                        take_profit_pct=self.cfg.take_profit_pct,
                        stop_loss_pct=self.cfg.stop_loss_pct,
                        trailing_activation_pct=self.cfg.trailing_activation_pct,
                        trailing_stop_pct=self.cfg.trailing_stop_pct,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Snipe buy mislukt voor %s: %s", mint, exc)
                await self.alerter.send(f"❌ Snipe mislukt voor {symbol}: {exc}")
