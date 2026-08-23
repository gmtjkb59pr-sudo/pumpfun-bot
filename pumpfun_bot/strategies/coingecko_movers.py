"""
Same discovery niche as birdeye_movers.py - already-existing Solana tokens
showing a real spike, not brand-new launches - via CoinGecko's free Demo
API plan instead (see coingecko.py's module docstring for the API/budget
details). Much higher call budget than Birdeye lets this poll every few
minutes instead of every 45, reacting to a genuinely short momentum window
(CoinGeckoMoversConfig.momentum_window, m5 by default) instead of only a
24h lagging one.

Reuses birdeye_movers.py's is_pump_fun_mint() filter - CoinGecko's trending
pools are Solana-wide, not pump.fun-specific, the exact same gap that made
Birdeye's trending list surface tokens PumpPortal has no route for at all
(confirmed live - see that module's comment for the full story).
"""
from __future__ import annotations

import asyncio
import logging

from ..alerts import Alerter
from ..coingecko import fetch_trending_pools, parse_pool_candidate
from ..config import CoinGeckoMoversConfig
from ..holder_concentration import fetch_top10_concentration_pct
from ..holder_count import fetch_holder_count
from ..outcome_tracker import OutcomeTracker
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state
from .birdeye_movers import is_pump_fun_mint

logger = logging.getLogger("pumpfun_bot.coingecko_movers")


class CoinGeckoMoversStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: CoinGeckoMoversConfig,
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
                "CoinGecko-movers staat aan maar er is geen COINGECKO_API_KEY ingesteld - "
                "strategie doet niets totdat die is gezet."
            )
            return
        logger.info(
            "CoinGecko-movers strategie gestart (dry_run=%s, poll_interval=%ds, window=%s).",
            self.dry_run, self.cfg.poll_interval_sec, self.cfg.momentum_window,
        )
        while True:
            await self._poll_once()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def _poll_once(self) -> None:
        pools = await fetch_trending_pools(self.cfg.api_key, limit=self.cfg.trending_limit)
        if pools is None:
            logger.debug("CoinGecko-movers: kon trending pools niet ophalen, sla deze poll over.")
            return
        for pool in pools:
            candidate = parse_pool_candidate(pool)
            if candidate is not None:
                await self._consider(candidate)

    async def _consider(self, candidate: dict) -> None:
        mint = candidate["mint"]
        if not is_pump_fun_mint(mint):
            # cheapest possible rejection - no RPC calls, no risk-manager
            # check, just a string check - see birdeye_movers.py's
            # is_pump_fun_mint for why this matters
            logger.info(
                "CoinGecko-movers: %s is geen pump.fun mint (geen \"pump\"-suffix), "
                "PumpPortal heeft hier geen route voor, sla over.", mint,
            )
            return
        if self.outcome_tracker is not None and self.outcome_tracker.is_tracking(mint):
            # already held (e.g. social_watch or birdeye_movers bought this
            # same mint) - OutcomeTracker keys by mint alone, a second buy
            # would spend real SOL on a position nothing would ever track
            return

        price_change_pct = candidate["price_change_pct"].get(self.cfg.momentum_window)
        if price_change_pct is None or price_change_pct <= 0:
            return

        if self.cfg.max_market_cap_usd > 0:
            market_cap_usd = candidate["market_cap_usd"]
            if market_cap_usd is None or market_cap_usd > self.cfg.max_market_cap_usd:
                logger.info(
                    "CoinGecko-movers: %s heeft $%s market cap, boven de max_market_cap_usd "
                    "van $%.0f (of onbekend), sla over.",
                    mint, market_cap_usd, self.cfg.max_market_cap_usd,
                )
                return

        open_positions_count = (
            self.outcome_tracker.open_position_count(strategy="coingecko_movers")
            if self.outcome_tracker is not None else None
        )
        ok, reason = self.risk.can_trade(
            self.trade_size_sol, None, open_positions_count=open_positions_count,
            max_open_positions_override=self.cfg.max_open_positions,
            max_sol_per_trade_override=self.trade_size_sol,
        )
        if not ok:
            logger.info("CoinGecko-movers: trade geblokkeerd door risk manager: %s", reason)
            return

        holder_count, top10_concentration_pct = await asyncio.gather(
            fetch_holder_count(mint, self.client.rpc_http_url),
            fetch_top10_concentration_pct(mint, self.client.rpc_http_url),
        )
        if self.cfg.min_holder_count > 0:
            if holder_count is None or holder_count < self.cfg.min_holder_count:
                logger.info(
                    "CoinGecko-movers: holder count te laag/onbekend voor %s, sla over.", mint,
                )
                return
        if self.cfg.max_top10_concentration_pct > 0:
            if top10_concentration_pct is None or top10_concentration_pct > self.cfg.max_top10_concentration_pct:
                logger.info(
                    "CoinGecko-movers: top-10 concentratie te hoog/onbekend voor %s, sla over.", mint,
                )
                return

        pair_name = candidate["pair_name"]
        volume_24h_usd = candidate["volume_24h_usd"]
        await self.alerter.send(
            f"⚡ CoinGecko-mover kandidaat: {pair_name} - {mint} "
            f"({price_change_pct:+.1f}% {self.cfg.momentum_window}, ${volume_24h_usd:,.0f} volume 24h)"
            if volume_24h_usd is not None else
            f"⚡ CoinGecko-mover kandidaat: {pair_name} - {mint} ({price_change_pct:+.1f}% {self.cfg.momentum_window})"
        )

        entry_ref = candidate["price_usd"]
        meta = {
            "price_change_pct": price_change_pct,
            "momentum_window": self.cfg.momentum_window,
            "volume_24h_usd": volume_24h_usd,
            "holder_count": holder_count,
            "top10_concentration_pct": top10_concentration_pct,
        }

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen (coingecko-movers): %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            bot_state.log_trade(
                "coingecko_movers", "buy", mint, self.trade_size_sol, dry_run=True, meta=meta,
            )
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, pair_name, pair_name, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="coingecko_movers",
                    price_source="usd",
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
                "coingecko_movers", "buy", mint, self.trade_size_sol,
                dry_run=False, tx_signature=result["signature"], meta=meta,
            )
            await self.alerter.send(f"✅ Gekocht (coingecko-movers): {mint} | tx: {result['signature']}")
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, pair_name, pair_name, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="coingecko_movers",
                    price_source="usd",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("CoinGecko-movers buy mislukt voor %s: %s", mint, exc)
            await self.risk.report_buy_result(success=False)
            await self.alerter.send(f"❌ CoinGecko-movers buy mislukt voor {mint}: {exc}")
