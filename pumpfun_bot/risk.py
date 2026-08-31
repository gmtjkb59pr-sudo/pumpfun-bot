"""
Centrale risk-manager. ELKE order (van welke strategie dan ook) moet hier langs
voordat hij uitgevoerd wordt. Dit is expres strikt: het is makkelijker om een
limiet later bewust te verhogen dan om een bot te stoppen die al te veel verloren heeft.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import risk_store
from .config import RiskConfig
from .state import bot_state

logger = logging.getLogger("pumpfun_bot.risk")

# circuit breaker: repeated buy failures (not sells - a failing sell already
# has its own retry/reconciliation safety net, and blocking sells would be
# the one thing this must never do) mean something systemic is likely wrong
# (RPC issues, wallet out of fee money, PumpPortal down) - pausing new buys
# and testing the waters before resuming beats blindly spending real SOL on
# a broken pipeline. Classic closed -> open -> half-open -> closed pattern:
# only ever narrows what gets bought (a failed test just keeps it paused
# longer), same tighten-only spirit as everything else self-adjusting here.
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_WINDOW_SEC = 300
CIRCUIT_BREAKER_COOLDOWN_SEC = 120
# if the caller never reports back after a half-open test buy was allowed
# through (e.g. an unrelated crash between can_trade() and
# report_buy_result()), don't stay stuck half-open forever, blocking every
# future trade - treat it as a failed test and go back to a full cooldown
CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SEC = 30


@dataclass
class RiskState:
    open_exposure_sol: float = 0.0
    realized_pnl_sol: float = 0.0
    trade_timestamps: list = field(default_factory=list)
    day_start_ts: float = field(default_factory=time.time)
    buy_failure_timestamps: list = field(default_factory=list)
    circuit_breaker_state: str = "closed"  # closed | open | half_open
    circuit_breaker_opened_at: float = 0.0
    # real, live wallet SOL balance - see RiskManager.update_live_balance()
    # and can_trade()'s max_exposure_pct_of_balance check. None until the
    # first successful balance lookup lands (fails open, same stance as
    # every other RPC-dependent check in this codebase).
    live_balance_sol: float | None = None


class RiskManager:
    def __init__(self, cfg: RiskConfig, alerter=None, store_path=None):
        self.cfg = cfg
        self.state = RiskState()
        # optional so tests/simple usage don't need one - only used to
        # announce circuit breaker trips/recoveries, nothing else here sends
        # alerts
        self.alerter = alerter
        # None (default) = do not persist, so existing tests that construct
        # a RiskManager without a path keep their in-memory-only behavior
        # and never write into data/. main.py passes path_for_mode().
        self.store_path = store_path
        if store_path is not None:
            self._load_persisted()

    def _reset_day_if_needed(self) -> None:
        if time.time() - self.state.day_start_ts > 86400:
            self.state.day_start_ts = time.time()
            self.state.realized_pnl_sol = 0.0
            logger.info("Dagelijkse teller gereset.")
            self._persist()

    def _trades_in_last_hour(self) -> int:
        cutoff = time.time() - 3600
        self.state.trade_timestamps = [t for t in self.state.trade_timestamps if t > cutoff]
        return len(self.state.trade_timestamps)

    def update_live_balance(self, balance_sol: float) -> None:
        """Called periodically by a background task (see balance_watch.py's
        watch_and_update_live_balance) with the wallet's real, on-chain SOL
        balance - see can_trade()'s max_exposure_pct_of_balance check for
        why this exists."""
        self.state.live_balance_sol = balance_sol

    def can_trade(
        self, sol_amount: float, liquidity_sol: float | None = None,
        open_positions_count: int | None = None,
        max_open_positions_override: int | None = None,
        max_sol_per_trade_override: float | None = None,
    ) -> tuple[bool, str]:
        """Controleer of een voorgestelde trade toegestaan is. Geeft (ok, reden).

        open_positions_count should come from OutcomeTracker.open_position_count()
        - the real, current count of tracked positions - not a separately
        maintained counter here. A counter incremented on every buy and
        decremented on every sell can drift from reality (e.g. a buy whose
        track() call returns early never actually gets tracked, so it can
        never be "sold" to free its slot back up) - "is there room" must
        reflect what's actually held, not what was once assumed opened.

        max_open_positions_override: user-requested per-strategy position
        budget - pass open_positions_count scoped to one strategy (see
        OutcomeTracker.open_position_count(strategy=...)) together with
        that strategy's own cap, so e.g. birdeye_movers firing a burst of
        buys can't crowd social_watch out of the whole shared pool. Falls
        back to self.cfg.max_open_positions (the old shared-pool behavior)
        when not given - exposure_sol/daily-loss limits stay global/shared
        on purpose, those are real wallet-wide budgets, not per-strategy.

        max_sol_per_trade_override: user-requested per-strategy trade-size
        cap (see CopyTradeConfig.trade_size_sol and the equivalent fields
        on SocialWatchConfig/BirdeyeMoversConfig) - without this, a
        strategy configured to trade LARGER than the shared
        risk.max_sol_per_trade would have every one of its trades silently
        rejected here regardless of its own config, since this check used
        to only ever compare against the shared value. Falls back to
        self.cfg.max_sol_per_trade when not given.

        In dry_run, the open-positions check, total-exposure cap,
        trades/hour cap, and daily-loss circuit breaker are all skipped
        entirely - user-requested: no real capital is at risk in dry-run,
        so cap nothing and let every qualifying candidate open a position
        to farm as much outcome data as possible. Confirmed live this cap
        was starving social_watch/coingecko_movers of almost every buy
        opportunity once sniper was also enabled, since sniper alone fires
        often enough to keep the shared exposure pool saturated. Per-trade
        size (max_sol_per_trade) and candidate-quality filters
        (min_liquidity_sol) still apply even in dry_run - those aren't
        capital constraints, they shape what's actually worth farming data
        on."""
        self._reset_day_if_needed()

        if sol_amount <= 0:
            return False, "Trade grootte moet positief zijn."

        effective_max_sol_per_trade = (
            max_sol_per_trade_override
            if max_sol_per_trade_override is not None
            else self.cfg.max_sol_per_trade
        )
        if sol_amount > effective_max_sol_per_trade:
            return False, (
                f"Trade van {sol_amount} SOL overschrijdt max_sol_per_trade "
                f"({effective_max_sol_per_trade})."
            )

        if (
            not self.cfg.dry_run
            and self.state.open_exposure_sol + sol_amount > self.cfg.max_sol_total_exposure
        ):
            return False, (
                f"Zou totale exposure naar {self.state.open_exposure_sol + sol_amount:.4f} SOL "
                f"brengen, max is {self.cfg.max_sol_total_exposure}."
            )

        # see max_exposure_pct_of_balance's docstring in config.py for the
        # real incident this fixes - a FIXED exposure cap can be larger
        # than the wallet's actual current balance, letting a burst of
        # buys spend the wallet down to near-zero. 0 (default) or no live
        # balance reading yet (fails open, same stance as every other RPC-
        # dependent check here) skips this and relies on the fixed cap
        # above alone, unchanged from before.
        if (
            not self.cfg.dry_run
            and self.cfg.max_exposure_pct_of_balance > 0
            and self.state.live_balance_sol is not None
        ):
            allowed = self.state.live_balance_sol * (self.cfg.max_exposure_pct_of_balance / 100)
            if self.state.open_exposure_sol + sol_amount > allowed:
                return False, (
                    f"Zou totale exposure naar {self.state.open_exposure_sol + sol_amount:.4f} SOL "
                    f"brengen, max is {self.cfg.max_exposure_pct_of_balance}% van live wallet-"
                    f"balance ({self.state.live_balance_sol:.4f} SOL) = {allowed:.4f} SOL."
                )

        if not self.cfg.dry_run and open_positions_count is not None:
            effective_max = (
                max_open_positions_override
                if max_open_positions_override is not None
                else self.cfg.max_open_positions
            )
            if open_positions_count >= effective_max:
                return False, (
                    f"Al {open_positions_count} open posities, max is {effective_max}."
                )

        if not self.cfg.dry_run and self._trades_in_last_hour() >= self.cfg.max_trades_per_hour:
            return False, f"Limiet van {self.cfg.max_trades_per_hour} trades/uur bereikt."

        if not self.cfg.dry_run and self.state.realized_pnl_sol <= -abs(self.cfg.max_daily_loss_sol):
            return False, (
                f"Dagelijkse verlieslimiet bereikt ({self.state.realized_pnl_sol:.4f} SOL). "
                f"Bot handelt vandaag niet meer."
            )

        if liquidity_sol is not None and liquidity_sol < self.cfg.min_liquidity_sol:
            return False, (
                f"Liquiditeit ({liquidity_sol} SOL) onder minimum ({self.cfg.min_liquidity_sol})."
            )

        # checked LAST, deliberately - a half-open breaker allows exactly
        # one buy through as a test, and this transition only fires once
        # every OTHER check above has already passed. If it were checked
        # earlier, a candidate that trips the half-open test but then fails
        # a later check (e.g. liquidity) would never actually attempt a buy,
        # so report_buy_result() would never be called, and the breaker
        # would stay stuck in half_open forever - deadlocked, blocking
        # every future trade including genuinely fine ones.
        breaker_ok, breaker_reason = self._check_circuit_breaker()
        if not breaker_ok:
            return False, breaker_reason

        return True, "ok"

    def _check_circuit_breaker(self) -> tuple[bool, str]:
        state = self.state
        now = time.time()

        if state.circuit_breaker_state == "closed":
            return True, "ok"

        if state.circuit_breaker_state == "half_open":
            if now - state.circuit_breaker_opened_at >= CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SEC:
                # test buy never reported back - don't stay stuck forever
                logger.warning(
                    "Circuit breaker: half-open test kreeg nooit een resultaat - "
                    "opnieuw open, volledige cooldown."
                )
                state.circuit_breaker_state = "open"
                state.circuit_breaker_opened_at = now
                return False, "Circuit breaker open - vorige test-buy geen resultaat, wacht opnieuw."
            # a test is already in flight - block everything else until it resolves
            return False, "Circuit breaker half-open - wacht op resultaat van test-buy."

        # state == "open"
        if now - state.circuit_breaker_opened_at >= CIRCUIT_BREAKER_COOLDOWN_SEC:
            state.circuit_breaker_state = "half_open"
            state.circuit_breaker_opened_at = now  # reused as "half-open since" for the timeout above
            logger.warning("Circuit breaker: cooldown voorbij, probeert 1 test-buy (half-open).")
            return True, "ok"
        remaining = CIRCUIT_BREAKER_COOLDOWN_SEC - (now - state.circuit_breaker_opened_at)
        return False, (
            f"Circuit breaker open - {len(state.buy_failure_timestamps)} mislukte buys "
            f"recent, wacht nog {remaining:.0f}s."
        )

    async def report_buy_result(self, success: bool) -> None:
        """Call after every LIVE buy attempt (not dry-run - nothing there
        can actually fail this way). Feeds the circuit breaker: enough real
        failures pauses new buys until a test buy proves the pipeline is
        healthy again. Never affects selling."""
        state = self.state
        now = time.time()

        if success:
            if state.circuit_breaker_state != "closed":
                state.circuit_breaker_state = "closed"
                state.buy_failure_timestamps = []
                message = "🟢 Circuit breaker gesloten - test-buy geslaagd, normale werking hervat."
                logger.warning(message)
                if self.alerter is not None:
                    await self.alerter.send(message)
            return

        if state.circuit_breaker_state == "half_open":
            # the test failed - stay paused, restart the cooldown
            state.circuit_breaker_state = "open"
            state.circuit_breaker_opened_at = now
            message = (
                f"🔴 Circuit breaker blijft open - test-buy ook mislukt, "
                f"nog {CIRCUIT_BREAKER_COOLDOWN_SEC}s cooldown."
            )
            logger.error(message)
            if self.alerter is not None:
                await self.alerter.send(message)
            return

        state.buy_failure_timestamps.append(now)
        cutoff = now - CIRCUIT_BREAKER_WINDOW_SEC
        state.buy_failure_timestamps = [t for t in state.buy_failure_timestamps if t > cutoff]

        if (
            state.circuit_breaker_state == "closed"
            and len(state.buy_failure_timestamps) >= CIRCUIT_BREAKER_FAILURE_THRESHOLD
        ):
            state.circuit_breaker_state = "open"
            state.circuit_breaker_opened_at = now
            message = (
                f"🔴 Circuit breaker open - {len(state.buy_failure_timestamps)} mislukte buys "
                f"binnen {CIRCUIT_BREAKER_WINDOW_SEC}s, nieuwe buys gepauzeerd voor "
                f"{CIRCUIT_BREAKER_COOLDOWN_SEC}s (verkopen blijft gewoon werken)."
            )
            logger.error(message)
            if self.alerter is not None:
                await self.alerter.send(message)

    def register_trade_opened(self, sol_amount: float) -> None:
        self.state.open_exposure_sol += sol_amount
        self.state.trade_timestamps.append(time.time())
        self._sync_dashboard()
        self._persist()

    def register_trade_closed(self, sol_amount: float, pnl_sol: float) -> None:
        self.state.open_exposure_sol = max(0.0, self.state.open_exposure_sol - sol_amount)
        self.state.realized_pnl_sol += pnl_sol
        logger.info(
            "Trade gesloten: pnl=%.4f SOL | dag totaal pnl=%.4f SOL | open exposure=%.4f SOL",
            pnl_sol, self.state.realized_pnl_sol, self.state.open_exposure_sol,
        )
        self._sync_dashboard()
        self._persist()

    def sync_open_exposure_from_positions(self, positions: dict) -> None:
        """Positions are the source of truth for open exposure after a
        restart (and after wallet reconcile drops stale ones). Daily-loss
        and hourly trade counters come from the persisted risk file;
        exposure is reconstructed so it cannot drift from
        remaining_fraction / sell_paused+reputation_logged state.

        A sell_paused position whose exposure was already released
        (reputation_logged) is skipped — counting it again would re-block
        buying with capital that's already been written off."""
        total = 0.0
        for info in positions.values():
            if info.get("sell_paused") and info.get("reputation_logged"):
                continue
            size = float(info.get("trade_size_sol") or 0.0)
            remaining = float(info.get("remaining_fraction") if info.get("remaining_fraction") is not None else 1.0)
            total += size * remaining
        self.state.open_exposure_sol = max(0.0, total)
        self._sync_dashboard()
        self._persist()

    def _load_persisted(self) -> None:
        raw = risk_store.load(self.store_path)
        if not raw:
            return
        try:
            self.state.realized_pnl_sol = float(raw.get("realized_pnl_sol") or 0.0)
            self.state.day_start_ts = float(raw.get("day_start_ts") or time.time())
            timestamps = raw.get("trade_timestamps") or []
            self.state.trade_timestamps = [float(t) for t in timestamps]
            # snapshot only — overwritten by sync_open_exposure_from_positions
            # once positions are loaded. Used if that sync never runs.
            if "open_exposure_sol" in raw:
                self.state.open_exposure_sol = max(0.0, float(raw["open_exposure_sol"] or 0.0))
        except (TypeError, ValueError):
            logger.warning("Kon risk-state niet laden van %s — start leeg.", self.store_path)
            self.state = RiskState()
            return
        self._reset_day_if_needed()
        self._trades_in_last_hour()  # prune stale timestamps
        self._sync_dashboard()
        logger.info(
            "Risk-tellers hersteld: dag-pnl=%.4f SOL | trades/uur=%d | exposure=%.4f SOL",
            self.state.realized_pnl_sol,
            len(self.state.trade_timestamps),
            self.state.open_exposure_sol,
        )

    def _persist(self) -> None:
        if self.store_path is None:
            return
        risk_store.save(
            {
                "realized_pnl_sol": self.state.realized_pnl_sol,
                "day_start_ts": self.state.day_start_ts,
                "trade_timestamps": list(self.state.trade_timestamps),
                "open_exposure_sol": self.state.open_exposure_sol,
            },
            self.store_path,
        )

    def _sync_dashboard(self) -> None:
        bot_state.update_risk_snapshot(
            open_exposure_sol=self.state.open_exposure_sol,
            realized_pnl_sol=self.state.realized_pnl_sol,
            trades_last_hour=self._trades_in_last_hour(),
        )
