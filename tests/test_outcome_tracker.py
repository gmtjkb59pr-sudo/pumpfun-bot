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


if __name__ == "__main__":
    unittest.main()
