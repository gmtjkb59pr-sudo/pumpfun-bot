import asyncio
import unittest

from pumpfun_bot.config import RiskConfig
from pumpfun_bot.risk import (
    CIRCUIT_BREAKER_COOLDOWN_SEC,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SEC,
    RiskManager,
)


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


def make_manager(**overrides) -> RiskManager:
    cfg = RiskConfig(
        max_sol_per_trade=0.05,
        max_sol_total_exposure=0.3,
        max_trades_per_hour=10,
        max_daily_loss_sol=0.2,
        min_liquidity_sol=5,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return RiskManager(cfg, alerter=FakeAlerter())


class RiskManagerCanTradeTests(unittest.TestCase):
    def test_allows_trade_within_limits(self):
        risk = make_manager()
        ok, reason = risk.can_trade(0.02, liquidity_sol=10)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_rejects_non_positive_amount(self):
        risk = make_manager()
        ok, _ = risk.can_trade(0)
        self.assertFalse(ok)

    def test_rejects_trade_above_max_per_trade(self):
        risk = make_manager(max_sol_per_trade=0.05)
        ok, reason = risk.can_trade(0.06)
        self.assertFalse(ok)
        self.assertIn("max_sol_per_trade", reason)

    def test_rejects_trade_that_exceeds_total_exposure(self):
        risk = make_manager(max_sol_total_exposure=0.1)
        risk.register_trade_opened(0.08)
        ok, _ = risk.can_trade(0.05)
        self.assertFalse(ok)

    def test_rejects_when_hourly_trade_limit_reached(self):
        risk = make_manager(max_trades_per_hour=2)
        risk.register_trade_opened(0.01)
        risk.register_trade_opened(0.01)
        ok, reason = risk.can_trade(0.01)
        self.assertFalse(ok)
        self.assertIn("trades/uur", reason)

    def test_rejects_when_daily_loss_limit_hit(self):
        risk = make_manager(max_daily_loss_sol=0.1)
        risk.register_trade_closed(0.05, pnl_sol=-0.1)
        ok, reason = risk.can_trade(0.01)
        self.assertFalse(ok)
        self.assertIn("verlieslimiet", reason)

    def test_rejects_low_liquidity(self):
        risk = make_manager(min_liquidity_sol=5)
        ok, reason = risk.can_trade(0.01, liquidity_sol=1)
        self.assertFalse(ok)
        self.assertIn("Liquiditeit", reason)

    def test_rejects_when_max_open_positions_reached(self):
        # open_positions_count comes from the caller (OutcomeTracker.open_
        # position_count() in real usage) - passed directly, not tracked as
        # a separate counter here, so it can't drift from what's actually
        # held (see docstring on can_trade). Only enforced in live mode -
        # see DryRunSkipsOpenPositionsCheckTests below for dry_run.
        risk = make_manager(max_open_positions=5, dry_run=False)
        ok, reason = risk.can_trade(0.01, open_positions_count=5)
        self.assertFalse(ok)
        self.assertIn("open posities", reason)

    def test_allows_trade_below_max_open_positions(self):
        risk = make_manager(max_open_positions=5, dry_run=False)
        ok, _ = risk.can_trade(0.01, open_positions_count=4)
        self.assertTrue(ok)

    def test_allows_trade_when_open_positions_count_not_provided(self):
        # callers without an OutcomeTracker (or that don't pass it) simply
        # don't get this check applied, rather than being blocked on None
        risk = make_manager(max_open_positions=5, dry_run=False)
        ok, _ = risk.can_trade(0.01)
        self.assertTrue(ok)

    def test_max_open_positions_override_takes_precedence(self):
        # user-requested per-strategy position budgets - a strategy's own
        # cap (passed per-call) overrides the global risk.max_open_positions
        risk = make_manager(max_open_positions=100, dry_run=False)
        ok, reason = risk.can_trade(
            0.01, open_positions_count=2, max_open_positions_override=2,
        )
        self.assertFalse(ok)
        self.assertIn("max is 2", reason)

    def test_falls_back_to_global_cap_when_no_override_given(self):
        risk = make_manager(max_open_positions=3, dry_run=False)
        ok, reason = risk.can_trade(0.01, open_positions_count=3)
        self.assertFalse(ok)
        self.assertIn("max is 3", reason)


class DryRunSkipsOpenPositionsCheckTests(unittest.TestCase):
    """User-requested: no real capital is at risk in dry-run, so cap
    nothing and let every qualifying candidate open a position to farm as
    much outcome data as possible."""

    def test_allows_trade_in_dry_run_even_at_or_above_the_cap(self):
        risk = make_manager(max_open_positions=5, dry_run=True)
        ok, _ = risk.can_trade(0.01, open_positions_count=999)
        self.assertTrue(ok)

    def test_allows_trade_in_dry_run_even_with_a_tight_override(self):
        risk = make_manager(max_open_positions=5, dry_run=True)
        ok, _ = risk.can_trade(
            0.01, open_positions_count=999, max_open_positions_override=1,
        )
        self.assertTrue(ok)


class RiskManagerStateTests(unittest.TestCase):
    def test_register_trade_opened_increases_exposure(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)
        self.assertEqual(len(risk.state.trade_timestamps), 1)

    def test_register_trade_closed_updates_exposure_and_pnl(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        risk.register_trade_closed(0.03, pnl_sol=-0.01)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, -0.01)

    def test_register_trade_closed_never_drops_exposure_below_zero(self):
        risk = make_manager()
        risk.register_trade_closed(0.03, pnl_sol=0.0)
        self.assertEqual(risk.state.open_exposure_sol, 0.0)


class CircuitBreakerTests(unittest.TestCase):
    """Repeated buy failures mean something systemic is likely wrong - the
    breaker pauses new buys and tests the waters before resuming, rather
    than blindly spending real SOL into a broken pipeline. Never affects
    selling - can_trade() is the only thing it gates."""

    def _fail_n_times(self, risk, n):
        for _ in range(n):
            asyncio.run(risk.report_buy_result(success=False))

    def test_stays_closed_below_the_failure_threshold(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD - 1)
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertTrue(ok)

    def test_trips_open_at_the_failure_threshold(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        ok, reason = risk.can_trade(0.01, liquidity_sol=10)
        self.assertFalse(ok)
        self.assertIn("Circuit breaker", reason)

    def test_sends_an_alert_when_it_trips(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        self.assertTrue(any("Circuit breaker open" in m for m in risk.alerter.messages))

    def test_failures_outside_the_window_do_not_count(self):
        risk = make_manager()
        # backdate old failures past the rolling window - they must not
        # contribute to tripping the breaker
        old = risk.state.buy_failure_timestamps
        import time as time_module
        from pumpfun_bot.risk import CIRCUIT_BREAKER_WINDOW_SEC
        for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
            old.append(time_module.time() - CIRCUIT_BREAKER_WINDOW_SEC - 10)
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertTrue(ok)

    def test_open_blocks_trades_until_cooldown_elapses(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_COOLDOWN_SEC - 1  # not quite elapsed
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertFalse(ok)

    def test_transitions_to_half_open_and_allows_one_test_buy_after_cooldown(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_COOLDOWN_SEC + 1

        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertTrue(ok)  # the one test buy gets through
        self.assertEqual(risk.state.circuit_breaker_state, "half_open")

        # a second check before the test buy resolves must be blocked -
        # exactly one test buy at a time, not a flood of them
        ok2, reason2 = risk.can_trade(0.01, liquidity_sol=10)
        self.assertFalse(ok2)
        self.assertIn("half-open", reason2)

    def test_half_open_success_closes_the_breaker_and_clears_failures(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_COOLDOWN_SEC + 1
        risk.can_trade(0.01, liquidity_sol=10)  # triggers the half-open transition
        self.assertEqual(risk.state.circuit_breaker_state, "half_open")

        asyncio.run(risk.report_buy_result(success=True))

        self.assertEqual(risk.state.circuit_breaker_state, "closed")
        self.assertEqual(risk.state.buy_failure_timestamps, [])
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertTrue(ok)

    def test_half_open_failure_reopens_with_a_fresh_cooldown(self):
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_COOLDOWN_SEC + 1
        risk.can_trade(0.01, liquidity_sol=10)  # triggers the half-open transition

        asyncio.run(risk.report_buy_result(success=False))

        self.assertEqual(risk.state.circuit_breaker_state, "open")
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)
        self.assertFalse(ok)  # fresh cooldown, not immediately retestable

    def test_half_open_with_no_result_reported_eventually_reopens(self):
        # safety net: if the caller never reports back (e.g. an unrelated
        # crash between can_trade() and report_buy_result()), don't stay
        # stuck half-open forever, blocking every future trade
        risk = make_manager()
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_COOLDOWN_SEC + 1
        risk.can_trade(0.01, liquidity_sol=10)  # triggers the half-open transition
        self.assertEqual(risk.state.circuit_breaker_state, "half_open")

        risk.state.circuit_breaker_opened_at -= CIRCUIT_BREAKER_HALF_OPEN_TIMEOUT_SEC + 1
        ok, _ = risk.can_trade(0.01, liquidity_sol=10)

        self.assertFalse(ok)
        self.assertEqual(risk.state.circuit_breaker_state, "open")

    def test_breaker_never_blocks_selling(self):
        # register_trade_closed (used on every sell) must never be gated by
        # the breaker - it doesn't go through can_trade() at all, and has
        # no circuit-breaker check of its own
        risk = make_manager()
        risk.register_trade_opened(0.03)
        self._fail_n_times(risk, CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        self.assertEqual(risk.state.circuit_breaker_state, "open")

        risk.register_trade_closed(0.03, pnl_sol=0.01)  # must not raise or be blocked
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)


if __name__ == "__main__":
    unittest.main()
