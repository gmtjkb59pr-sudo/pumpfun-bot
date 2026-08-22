"""
Discovers already-existing Solana tokens with a real volume/price spike via
Birdeye's trending API, instead of waiting for a brand-new launch the way
social_watch.py does - the gap social_watch structurally cannot cover (see
birdeye.py's module docstring for why).

Runs on a slow poll cadence (BirdeyeMoversConfig.poll_interval_sec,
2700s/45min by default) to respect the free tier's ~1,000 calls/month
budget - this is not a real-time strategy, it's a periodic sweep. Reuses
the same buy-quality filters (holder count, top-10 concentration) and the
same risk manager / outcome tracker as social_watch, just with a
different discovery source and a slower, coarser price reference (the
trending snapshot's own price, not a live tick - acceptable at this
cadence, unlike sniper's need for split-second freshness).
"""
from __future__ import annotations

import asyncio
import logging

from ..alerts import Alerter
from ..birdeye import fetch_trending_tokens
from ..config import BirdeyeMoversConfig
from ..holder_concentration import fetch_top10_concentration_pct
from ..holder_count import fetch_holder_count
from ..outcome_tracker import OutcomeTracker
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.birdeye_movers")


class BirdeyeMoversStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: BirdeyeMoversConfig,
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

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        if not self.cfg.api_key:
            logger.warning(
                "Birdeye-movers staat aan maar er is geen BIRDEYE_API_KEY ingesteld - "
                "strategie doet niets totdat die is gezet."
            )
            return
        logger.info(
            "Birdeye-movers strategie gestart (dry_run=%s, poll_interval=%ds).",
            self.dry_run, self.cfg.poll_interval_sec,
        )
        while True:
            await self._poll_once()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def _poll_once(self) -> None:
        tokens = await fetch_trending_tokens(self.cfg.api_key, limit=self.cfg.trending_limit)
        if tokens is None:
            logger.debug("Birdeye-movers: kon trending tokens niet ophalen, sla deze poll over.")
            return
        for token in tokens:
            await self._consider(token)

    async def _consider(self, token: dict) -> None:
        mint = token.get("address")
        if not mint:
            return
        if self.outcome_tracker is not None and self.outcome_tracker.is_tracking(mint):
            # already held (e.g. social_watch bought this same mint) -
            # OutcomeTracker keys by mint alone, a second buy would spend
            # real SOL on a position nothing would ever track or exit
            return

        price_change_pct = token.get("price24hChangePercent")
        if price_change_pct is None or price_change_pct <= 0:
            return

        if self.cfg.max_market_cap_usd > 0:
            # confirmed live: Birdeye's trending list (sorted by volumeUSD)
            # surfaces major/blue-chip tokens like SOL and wrapped ETH, not
            # just memecoins - "buying" those through a pump.fun-style
            # trade is a category error, not a real candidate. Checked
            # before any RPC calls - cheapest possible rejection.
            market_cap_usd = token.get("marketcap")
            if market_cap_usd is None or market_cap_usd > self.cfg.max_market_cap_usd:
                logger.info(
                    "Birdeye-movers: %s heeft $%s market cap, boven de max_market_cap_usd "
                    "van $%.0f (of onbekend), sla over.",
                    token.get("address"), market_cap_usd, self.cfg.max_market_cap_usd,
                )
                return

        name = token.get("name", "?")
        symbol = token.get("symbol", "?")

        open_positions_count = (
            self.outcome_tracker.open_position_count() if self.outcome_tracker is not None else None
        )
        ok, reason = self.risk.can_trade(
            self.trade_size_sol, None, open_positions_count=open_positions_count,
        )
        if not ok:
            logger.info("Birdeye-movers: trade geblokkeerd door risk manager: %s", reason)
            return

        holder_count, top10_concentration_pct = await asyncio.gather(
            fetch_holder_count(mint, self.client.rpc_http_url),
            fetch_top10_concentration_pct(mint, self.client.rpc_http_url),
        )
        if self.cfg.min_holder_count > 0:
            if holder_count is None or holder_count < self.cfg.min_holder_count:
                logger.info(
                    "Birdeye-movers: holder count te laag/onbekend voor %s, sla over.", mint,
                )
                return
        if self.cfg.max_top10_concentration_pct > 0:
            if top10_concentration_pct is None or top10_concentration_pct > self.cfg.max_top10_concentration_pct:
                logger.info(
                    "Birdeye-movers: top-10 concentratie te hoog/onbekend voor %s, sla over.", mint,
                )
                return

        entry_ref = token.get("price")
        volume_24h_usd = token.get("volume24hUSD")
        await self.alerter.send(
            f"📈 Birdeye-mover kandidaat: {name} ({symbol}) - {mint} "
            f"({price_change_pct:+.1f}% 24h, ${volume_24h_usd:,.0f} volume)"
            if volume_24h_usd is not None else
            f"📈 Birdeye-mover kandidaat: {name} ({symbol}) - {mint} ({price_change_pct:+.1f}% 24h)"
        )

        meta = {
            "price_change_24h_pct": price_change_pct,
            "volume_24h_usd": volume_24h_usd,
            "holder_count": holder_count,
            "top10_concentration_pct": top10_concentration_pct,
        }

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen (birdeye-movers): %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            bot_state.log_trade(
                "birdeye_movers", "buy", mint, self.trade_size_sol, dry_run=True, meta=meta,
            )
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                )
            return

        try:
            result = await self.client.build_and_send_trade(
                action="buy", mint=mint, amount_sol=self.trade_size_sol,
                slippage_pct=self.slippage_pct,
            )
            self.risk.register_trade_opened(self.trade_size_sol)
            await self.risk.report_buy_result(success=True)
            bot_state.log_trade(
                "birdeye_movers", "buy", mint, self.trade_size_sol,
                dry_run=False, tx_signature=result["signature"], meta=meta,
            )
            await self.alerter.send(f"✅ Gekocht (birdeye-movers): {symbol} | tx: {result['signature']}")
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Birdeye-movers buy mislukt voor %s: %s", mint, exc)
            await self.risk.report_buy_result(success=False)
            await self.alerter.send(f"❌ Birdeye-movers buy mislukt voor {symbol}: {exc}")
