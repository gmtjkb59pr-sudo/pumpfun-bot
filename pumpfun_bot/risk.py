"""
Centrale risk-manager. ELKE order (van welke strategie dan ook) moet hier langs
voordat hij uitgevoerd wordt. Dit is expres strikt: het is makkelijker om een
limiet later bewust te verhogen dan om een bot te stoppen die al te veel verloren heeft.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import RiskConfig
from .state import bot_state

logger = logging.getLogger("pumpfun_bot.risk")


@dataclass
class RiskState:
    open_exposure_sol: float = 0.0
    open_positions_count: int = 0
    realized_pnl_sol: float = 0.0
    trade_timestamps: list = field(default_factory=list)
    day_start_ts: float = field(default_factory=time.time)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.state = RiskState()

    def _reset_day_if_needed(self) -> None:
        if time.time() - self.state.day_start_ts > 86400:
            self.state.day_start_ts = time.time()
            self.state.realized_pnl_sol = 0.0
            logger.info("Dagelijkse teller gereset.")

    def _trades_in_last_hour(self) -> int:
        cutoff = time.time() - 3600
        self.state.trade_timestamps = [t for t in self.state.trade_timestamps if t > cutoff]
        return len(self.state.trade_timestamps)

    def can_trade(self, sol_amount: float, liquidity_sol: float | None = None) -> tuple[bool, str]:
        """Controleer of een voorgestelde trade toegestaan is. Geeft (ok, reden)."""
        self._reset_day_if_needed()

        if sol_amount <= 0:
            return False, "Trade grootte moet positief zijn."

        if sol_amount > self.cfg.max_sol_per_trade:
            return False, (
                f"Trade van {sol_amount} SOL overschrijdt max_sol_per_trade "
                f"({self.cfg.max_sol_per_trade})."
            )

        if self.state.open_exposure_sol + sol_amount > self.cfg.max_sol_total_exposure:
            return False, (
                f"Zou totale exposure naar {self.state.open_exposure_sol + sol_amount:.4f} SOL "
                f"brengen, max is {self.cfg.max_sol_total_exposure}."
            )

        if self.state.open_positions_count >= self.cfg.max_open_positions:
            return False, (
                f"Al {self.state.open_positions_count} open posities, max is "
                f"{self.cfg.max_open_positions}."
            )

        if self._trades_in_last_hour() >= self.cfg.max_trades_per_hour:
            return False, f"Limiet van {self.cfg.max_trades_per_hour} trades/uur bereikt."

        if self.state.realized_pnl_sol <= -abs(self.cfg.max_daily_loss_sol):
            return False, (
                f"Dagelijkse verlieslimiet bereikt ({self.state.realized_pnl_sol:.4f} SOL). "
                f"Bot handelt vandaag niet meer."
            )

        if liquidity_sol is not None and liquidity_sol < self.cfg.min_liquidity_sol:
            return False, (
                f"Liquiditeit ({liquidity_sol} SOL) onder minimum ({self.cfg.min_liquidity_sol})."
            )

        return True, "ok"

    def register_trade_opened(self, sol_amount: float) -> None:
        self.state.open_exposure_sol += sol_amount
        self.state.open_positions_count += 1
        self.state.trade_timestamps.append(time.time())
        self._sync_dashboard()

    def register_trade_closed(self, sol_amount: float, pnl_sol: float) -> None:
        self.state.open_exposure_sol = max(0.0, self.state.open_exposure_sol - sol_amount)
        self.state.open_positions_count = max(0, self.state.open_positions_count - 1)
        self.state.realized_pnl_sol += pnl_sol
        logger.info(
            "Trade gesloten: pnl=%.4f SOL | dag totaal pnl=%.4f SOL | open exposure=%.4f SOL",
            pnl_sol, self.state.realized_pnl_sol, self.state.open_exposure_sol,
        )
        self._sync_dashboard()

    def _sync_dashboard(self) -> None:
        bot_state.update_risk_snapshot(
            open_exposure_sol=self.state.open_exposure_sol,
            realized_pnl_sol=self.state.realized_pnl_sol,
            trades_last_hour=self._trades_in_last_hour(),
        )
