import unittest

from pumpfun_bot.config import RiskConfig
from pumpfun_bot.risk import RiskManager


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
    return RiskManager(cfg)


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
        risk = make_manager(max_open_positions=5)
        for _ in range(5):
            risk.register_trade_opened(0.01)
        ok, reason = risk.can_trade(0.01)
        self.assertFalse(ok)
        self.assertIn("open posities", reason)

    def test_allows_trade_below_max_open_positions(self):
        risk = make_manager(max_open_positions=5)
        for _ in range(4):
            risk.register_trade_opened(0.01)
        ok, _ = risk.can_trade(0.01)
        self.assertTrue(ok)

    def test_a_closed_position_frees_up_a_slot(self):
        risk = make_manager(max_open_positions=5)
        for _ in range(5):
            risk.register_trade_opened(0.01)
        risk.register_trade_closed(0.01, pnl_sol=0.0)
        ok, _ = risk.can_trade(0.01)
        self.assertTrue(ok)


class RiskManagerStateTests(unittest.TestCase):
    def test_register_trade_opened_increases_exposure(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)
        self.assertEqual(len(risk.state.trade_timestamps), 1)

    def test_register_trade_opened_increments_open_positions_count(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        risk.register_trade_opened(0.03)
        self.assertEqual(risk.state.open_positions_count, 2)

    def test_register_trade_closed_updates_exposure_and_pnl(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        risk.register_trade_closed(0.03, pnl_sol=-0.01)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, -0.01)

    def test_register_trade_closed_decrements_open_positions_count(self):
        risk = make_manager()
        risk.register_trade_opened(0.03)
        risk.register_trade_closed(0.03, pnl_sol=0.0)
        self.assertEqual(risk.state.open_positions_count, 0)

    def test_register_trade_closed_never_drops_exposure_below_zero(self):
        risk = make_manager()
        risk.register_trade_closed(0.03, pnl_sol=0.0)
        self.assertEqual(risk.state.open_exposure_sol, 0.0)

    def test_register_trade_closed_never_drops_position_count_below_zero(self):
        risk = make_manager()
        risk.register_trade_closed(0.03, pnl_sol=0.0)
        self.assertEqual(risk.state.open_positions_count, 0)


if __name__ == "__main__":
    unittest.main()
