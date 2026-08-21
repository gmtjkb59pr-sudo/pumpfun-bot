import asyncio
import time
import unittest

from pumpfun_bot.config import RiskConfig
from pumpfun_bot.outcome_tracker import CHECKPOINTS_SEC, OutcomeTracker, is_funded_key_rejection
from pumpfun_bot.risk import RiskManager


class IsFundedKeyRejectionTests(unittest.TestCase):
    def test_detects_the_actual_rejection_message(self):
        message = (
            "'subscribeTokenTrade' and 'subscribeAccountTrade' methods are only "
            "available when connecting with an API key funded with at least 0.02 SOL."
        )
        self.assertTrue(is_funded_key_rejection(message))

    def test_does_not_flag_success_confirmation(self):
        self.assertFalse(is_funded_key_rejection("Successfully subscribed to keys."))

    def test_does_not_flag_unrelated_message(self):
        self.assertFalse(is_funded_key_rejection("Successfully subscribed to token creation events."))


class ClosesPositionAtFinalCheckpointTests(unittest.TestCase):
    def _make_tracker_with_pending(self, *, has_real_update: bool, entry_ref=100.0, last_ref=120.0):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk)
        # backdate entry_ts so the final checkpoint is immediately "due",
        # instead of sleeping for real in a test
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - CHECKPOINTS_SEC[-1] - 1,
            "entry_ref": entry_ref,
            "last_ref": last_ref,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(CHECKPOINTS_SEC[:-1]),  # earlier checkpoints already fired
            "has_real_update": has_real_update,
        }
        return tracker, risk

    def test_closes_position_and_registers_pnl_when_measured(self):
        tracker, risk = self._make_tracker_with_pending(
            has_real_update=True, entry_ref=100.0, last_ref=120.0
        )
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.20)
        self.assertNotIn("MINT", tracker._pending)

    def test_leaves_position_open_when_unmeasured(self):
        tracker, risk = self._make_tracker_with_pending(has_real_update=False)
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.05)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)


class TakeProfitStopLossExitTests(unittest.TestCase):
    def _make_tracker(self, *, take_profit_pct=50.0, stop_loss_pct=25.0, entry_ref=100.0):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk,
            take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
        )
        tracker._pending["MINT"] = {
            "entry_ts": time.time(),
            "entry_ref": entry_ref,
            "last_ref": entry_ref,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
        }
        return tracker, risk

    def test_take_profit_closes_position_immediately(self):
        tracker, risk = self._make_tracker(take_profit_pct=50.0, entry_ref=100.0)
        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, crosses +50%

        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.51, places=4)

    def test_stop_loss_closes_position_immediately(self):
        tracker, risk = self._make_tracker(stop_loss_pct=25.0, entry_ref=100.0)
        asyncio.run(tracker._handle_price_update("MINT", 70.0))  # -30%, crosses -25%

        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * -0.30, places=4)

    def test_small_move_does_not_trigger_exit(self):
        tracker, risk = self._make_tracker(take_profit_pct=50.0, stop_loss_pct=25.0, entry_ref=100.0)
        asyncio.run(tracker._handle_price_update("MINT", 110.0))  # +10%, neither threshold

        self.assertIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.05)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)
        self.assertTrue(tracker._pending["MINT"]["has_real_update"])
        self.assertEqual(tracker._pending["MINT"]["last_ref"], 110.0)

    def test_ignores_updates_for_untracked_mints(self):
        tracker, risk = self._make_tracker()
        # should not raise even though "OTHER" was never tracked
        asyncio.run(tracker._handle_price_update("OTHER", 999.0))
        self.assertNotIn("OTHER", tracker._pending)


if __name__ == "__main__":
    unittest.main()
