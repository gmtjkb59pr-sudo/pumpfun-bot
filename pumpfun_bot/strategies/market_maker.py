"""
Simpele grid market-maker: zet (denkbeeldige) buy/sell levels rond de laatste
prijs en 'vult' een level wanneer de prijs er doorheen beweegt. Dit is een
vereenvoudigd model - een echte market maker houdt ook rekening met
inventory-risico, bonding-curve mechanica van pump.fun, en netto exposure.
"""
from __future__ import annotations

import logging

from ..alerts import Alerter
from ..config import MarketMakerConfig
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.market_maker")


class GridState:
    def __init__(self, center_price: float, levels: int, spacing_pct: float):
        self.center_price = center_price
        self.levels = levels
        self.spacing_pct = spacing_pct
        self.filled_levels: set[int] = set()

    def level_price(self, i: int) -> float:
        # negatieve i = buy levels onder center, positieve i = sell levels erboven
        return self.center_price * (1 + (i * self.spacing_pct / 100))


class MarketMakerStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: MarketMakerConfig,
        risk: RiskManager,
        alerter: Alerter,
        slippage_pct: float,
        dry_run: bool,
    ):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.alerter = alerter
        self.slippage_pct = slippage_pct
        self.dry_run = dry_run
        self.grids: dict[str, GridState] = {}

    async def run(self) -> None:
        if not self.cfg.enabled or not self.cfg.target_tokens:
            return
        logger.info(
            "Market maker gestart voor %d token(s), dry_run=%s.",
            len(self.cfg.target_tokens), self.dry_run,
        )

        async for event in self.client.stream_token_trades(self.cfg.target_tokens):
            mint = event.get("mint")
            price = event.get("price") or event.get("marketCapSol")
            if not mint or not price:
                continue

            grid = self.grids.get(mint)
            if grid is None:
                grid = GridState(price, self.cfg.grid_levels, self.cfg.grid_spacing_pct)
                self.grids[mint] = grid
                logger.info("Grid geïnitialiseerd voor %s rond prijs %.8f", mint, price)
                continue

            half = self.cfg.grid_levels // 2
            for i in range(-half, half + 1):
                if i == 0 or i in grid.filled_levels:
                    continue
                level_price = grid.level_price(i)
                crossed = (i < 0 and price <= level_price) or (i > 0 and price >= level_price)
                if not crossed:
                    continue

                action = "buy" if i < 0 else "sell"
                amount = self.cfg.order_size_sol

                ok, reason = self.risk.can_trade(amount)
                if not ok:
                    logger.info("Grid order geblokkeerd door risk manager: %s", reason)
                    continue

                grid.filled_levels.add(i)
                await self.alerter.send(
                    f"📊 Grid level {i} geraakt voor {mint} @ {level_price:.8f} -> {action} "
                    f"{amount} SOL"
                )

                if self.dry_run:
                    logger.info("[DRY RUN] Grid %s: %s %s SOL van %s", i, action, amount, mint)
                    if action == "buy":
                        self.risk.register_trade_opened(amount)
                    bot_state.log_trade("market_maker", action, mint, amount, dry_run=True)
                    continue

                try:
                    result = await self.client.build_and_send_trade(
                        action=action,
                        mint=mint,
                        amount_sol=amount,
                        slippage_pct=self.slippage_pct,
                    )
                    if action == "buy":
                        self.risk.register_trade_opened(amount)
                    bot_state.log_trade(
                        "market_maker", action, mint, amount,
                        dry_run=False, tx_signature=result["signature"],
                    )
                    await self.alerter.send(f"✅ Grid order uitgevoerd | tx: {result['signature']}")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Grid order mislukt voor %s: %s", mint, exc)
                    await self.alerter.send(f"❌ Grid order mislukt: {exc}")
