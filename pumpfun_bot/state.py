"""
Thread-safe gedeelde status tussen de async trading-loop en de (aparte thread)
webserver. Bevat geen trading-logica zelf - puur een blackboard dat de rest
van de bot bijwerkt en het dashboard uitleest.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from .activity_log import append_jsonl


@dataclass
class TradeLogEntry:
    ts: float
    strategy: str
    action: str
    mint: str
    amount_sol: float
    dry_run: bool
    tx_signature: str = ""
    # entry-time features (e.g. liquidity_sol, has_socials) kept alongside the
    # trade so outcome stats can later be broken down by what the bot saw
    meta: dict = field(default_factory=dict)


class BotState:
    def __init__(self, max_trades: int = 200, max_alerts: int = 300):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.dry_run: bool = True
        self.wallet_pubkey: str = ""
        self.strategies_enabled: dict[str, bool] = {}
        self.open_exposure_sol: float = 0.0
        self.realized_pnl_sol: float = 0.0
        self.trades_last_hour: int = 0
        self.max_trades_per_hour: int = 0
        self.max_daily_loss_sol: float = 0.0
        self.max_sol_total_exposure: float = 0.0
        # whether PumpPortal has rejected the paid subscribeTokenTrade feed
        # this run (missing/unfunded PUMPPORTAL_API_KEY) - lets the dashboard
        # distinguish "your key doesn't work" from "these tokens just never
        # traded", instead of always blaming a missing key
        self.outcome_tracking_api_key_rejected: bool = False
        self._trades: deque = deque(maxlen=max_trades)
        self._alerts: deque = deque(maxlen=max_alerts)

    def init_meta(
        self,
        dry_run: bool,
        wallet_pubkey: str,
        strategies_enabled: dict[str, bool],
        max_trades_per_hour: int,
        max_daily_loss_sol: float,
        max_sol_total_exposure: float,
    ) -> None:
        with self._lock:
            self.dry_run = dry_run
            self.wallet_pubkey = wallet_pubkey
            self.strategies_enabled = strategies_enabled
            self.max_trades_per_hour = max_trades_per_hour
            self.max_daily_loss_sol = max_daily_loss_sol
            self.max_sol_total_exposure = max_sol_total_exposure

    def update_risk_snapshot(
        self, open_exposure_sol: float, realized_pnl_sol: float, trades_last_hour: int
    ) -> None:
        with self._lock:
            self.open_exposure_sol = open_exposure_sol
            self.realized_pnl_sol = realized_pnl_sol
            self.trades_last_hour = trades_last_hour

    def set_outcome_tracking_rejected(self, rejected: bool) -> None:
        with self._lock:
            self.outcome_tracking_api_key_rejected = rejected

    def log_trade(
        self,
        strategy: str,
        action: str,
        mint: str,
        amount_sol: float,
        dry_run: bool,
        tx_signature: str = "",
        meta: dict | None = None,
    ) -> None:
        entry = TradeLogEntry(
            ts=time.time(),
            strategy=strategy,
            action=action,
            mint=mint,
            amount_sol=amount_sol,
            dry_run=dry_run,
            tx_signature=tx_signature,
            meta=meta or {},
        )
        with self._lock:
            self._trades.appendleft(entry)
        append_jsonl({"type": "trade", **asdict(entry)})

    def log_alert(self, message: str) -> None:
        entry = {"ts": time.time(), "message": message}
        with self._lock:
            self._alerts.appendleft(entry)
        append_jsonl({"type": "alert", **entry})

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "started_at": self.started_at,
                "uptime_seconds": time.time() - self.started_at,
                "dry_run": self.dry_run,
                "wallet_pubkey": self.wallet_pubkey,
                "strategies_enabled": self.strategies_enabled,
                "open_exposure_sol": self.open_exposure_sol,
                "realized_pnl_sol": self.realized_pnl_sol,
                "trades_last_hour": self.trades_last_hour,
                "max_trades_per_hour": self.max_trades_per_hour,
                "max_daily_loss_sol": self.max_daily_loss_sol,
                "max_sol_total_exposure": self.max_sol_total_exposure,
                "outcome_tracking_api_key_rejected": self.outcome_tracking_api_key_rejected,
                "trades": [asdict(t) for t in list(self._trades)[:50]],
                "alerts": list(self._alerts)[:50],
            }


# Eén gedeelde instantie voor het hele proces - simpel en genoeg voor lokaal gebruik.
bot_state = BotState()
