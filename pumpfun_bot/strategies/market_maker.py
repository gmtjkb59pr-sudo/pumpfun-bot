"""
Grid market-maker op een token dat de gebruiker al aanwijst: buy/sell levels
rond de huidige bonding-curve spot, met echte constant-product executable
prijzen (niet last-price * (1 ± spacing%)).

Dit is géén wash trading, volume-spoofing of multi-wallet grid. Het plaatst
gewone buy/sell orders op één wallet tegen de curve van tokens in
target_tokens.
"""
from __future__ import annotations

import logging

from ..alerts import Alerter
from ..bonding_curve import BondingCurve
from ..config import MarketMakerConfig
from ..pumpportal_client import MissingPumpPortalFieldError, PumpPortalClient
from ..risk import RiskManager
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.market_maker")


class GridState:
    def __init__(
        self,
        curve: BondingCurve,
        levels: int,
        spacing_pct: float,
        order_size_sol: float,
    ):
        self.curve = curve
        # center is frozen at init so levels don't drift as the curve moves
        self.center_spot = curve.spot_price()
        self.levels = levels
        self.spacing_pct = spacing_pct
        self.order_size_sol = order_size_sol
        self.filled_levels: set[int] = set()

    def level_spot(self, i: int) -> float:
        """Target spot (SOL/token) for grid level i. Negative i = buy below
        center, positive i = sell above. This is the trigger, not the fill."""
        return self.center_spot * (1 + (i * self.spacing_pct / 100))

    def level_executable_price(self, i: int) -> float:
        """Average fill price for order_size_sol once the curve sits at
        level i's spot. Buys pay above that spot (impact); sells receive
        below it. This is the number a naive last*(1±pct) grid was missing."""
        target = self.level_spot(i)
        curve_at_level = self.curve.at_spot(target)
        if i < 0:
            return curve_at_level.avg_buy_price(self.order_size_sol)
        tokens = self.order_size_sol / target
        return curve_at_level.avg_sell_price(tokens)

    def update_curve(self, curve: BondingCurve) -> None:
        self.curve = curve


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
            "Market maker gestart voor %d token(s), dry_run=%s (bonding-curve grid).",
            len(self.cfg.target_tokens), self.dry_run,
        )

        async for event in self.client.stream_token_trades(self.cfg.target_tokens):
            mint = event.get("mint")
            if not mint:
                continue
            try:
                curve = BondingCurve.from_pumpportal_event(event)
            except MissingPumpPortalFieldError as exc:
                logger.warning(
                    "Market maker: event voor %s mist bonding-curve reserves (%s) — "
                    "skip, niet terugvallen op last-price/marketCapSol.",
                    mint, exc,
                )
                continue

            grid = self.grids.get(mint)
            if grid is None:
                grid = GridState(
                    curve, self.cfg.grid_levels, self.cfg.grid_spacing_pct,
                    self.cfg.order_size_sol,
                )
                self.grids[mint] = grid
                logger.info(
                    "Grid geïnitialiseerd voor %s rond spot %.10f SOL/token "
                    "(%d levels, spacing %.2f%%)",
                    mint, grid.center_spot, self.cfg.grid_levels, self.cfg.grid_spacing_pct,
                )
                continue

            grid.update_curve(curve)
            spot = curve.spot_price()

            half = self.cfg.grid_levels // 2
            for i in range(-half, half + 1):
                if i == 0 or i in grid.filled_levels:
                    continue
                target_spot = grid.level_spot(i)
                crossed = (i < 0 and spot <= target_spot) or (i > 0 and spot >= target_spot)
                if not crossed:
                    continue

                action = "buy" if i < 0 else "sell"
                amount = self.cfg.order_size_sol
                exec_price = grid.level_executable_price(i)

                ok, reason = self.risk.can_trade(amount)
                if not ok:
                    logger.info("Grid order geblokkeerd door risk manager: %s", reason)
                    continue

                grid.filled_levels.add(i)
                await self.alerter.send(
                    f"📊 Grid level {i} geraakt voor {mint} @ spot {target_spot:.10f} "
                    f"(executable {exec_price:.10f}) -> {action} {amount} SOL"
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
