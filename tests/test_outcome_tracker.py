import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import RiskConfig
from pumpfun_bot.outcome_tracker import CHECKPOINTS_SEC, OutcomeTracker, is_funded_key_rejection
from pumpfun_bot.risk import RiskManager

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH
_TEST_LOG_FILE = None


def setUpModule():
    # append_jsonl() always writes to activity_log.DATA_LOG_PATH - most tests
    # in this file exercise the real un-mocked path (only a couple of tests
    # mock append_jsonl directly), so without this every run of this module
    # would append junk "MINT" records into the real, live activity_log.jsonl
    global _TEST_LOG_FILE
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)


class FakeClient:
    """Stands in for PumpPortalClient - no real network calls."""

    def __init__(self, *, should_fail=False, signature="fake_sig"):
        self.should_fail = should_fail
        self.signature = signature
        self.sell_calls = []

    async def build_and_send_full_sell(self, mint, slippage_pct):
        self.sell_calls.append((mint, slippage_pct))
        if self.should_fail:
            raise RuntimeError("simulated RPC failure")
        return {"signature": self.signature, "action": "sell", "mint": mint, "amount": "100%"}


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
            "peak_ref": entry_ref,
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

    def test_exit_starts_post_exit_tracking(self):
        tracker, risk = self._make_tracker(take_profit_pct=50.0, entry_ref=100.0)
        asyncio.run(tracker._handle_price_update("MINT", 150.0))  # +50%, exits

        self.assertIn("MINT", tracker._post_exit)
        post = tracker._post_exit["MINT"]
        self.assertEqual(post["reason"], "take_profit")
        self.assertEqual(post["realized_pct_change"], 50.0)
        self.assertEqual(post["entry_ref"], 100.0)
        self.assertEqual(post["exit_ref"], 150.0)
        self.assertFalse(post["has_real_update"])


class PostExitCheckpointTests(unittest.TestCase):
    def _make_tracker_with_post_exit(self, *, has_real_update: bool, last_ref=180.0):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        tracker._post_exit["MINT"] = {
            "exit_ts": time.time() - CHECKPOINTS_SEC[-1] - 1,
            "entry_ref": 100.0,
            "exit_ref": 150.0,  # exited at +50%
            "realized_pct_change": 50.0,
            "reason": "take_profit",
            "name": "Test Token",
            "symbol": "TEST",
            "last_ref": last_ref,
            "hit": set(CHECKPOINTS_SEC[:-1]),
            "has_real_update": has_real_update,
        }
        return tracker

    def test_holding_would_have_done_better(self):
        # price kept climbing to 180 (i.e. +80% from entry) after we exited at +50%
        tracker = self._make_tracker_with_post_exit(has_real_update=True, last_ref=180.0)
        asyncio.run(tracker._emit_post_exit_checkpoints())

        self.assertNotIn("MINT", tracker._post_exit)  # final checkpoint clears it

    def test_computes_vs_realized_pct_correctly(self):
        import pumpfun_bot.outcome_tracker as outcome_tracker_module

        captured = []
        original_append = outcome_tracker_module.append_jsonl
        outcome_tracker_module.append_jsonl = captured.append
        try:
            tracker = self._make_tracker_with_post_exit(has_real_update=True, last_ref=180.0)
            asyncio.run(tracker._emit_post_exit_checkpoints())
        finally:
            outcome_tracker_module.append_jsonl = original_append

        checks = [r for r in captured if r["type"] == "post_exit_check"]
        self.assertEqual(len(checks), 1)
        record = checks[0]
        self.assertEqual(record["checkpoint_sec_after_exit"], CHECKPOINTS_SEC[-1])
        # held from 100 to 180 = +80%, realized was +50% -> holding was +30pp better
        self.assertEqual(record["pct_change_if_held_from_entry"], 80.0)
        self.assertEqual(record["vs_realized_pct"], 30.0)

    def test_leaves_unmeasured_flag_when_no_real_update(self):
        import pumpfun_bot.outcome_tracker as outcome_tracker_module

        captured = []
        original_append = outcome_tracker_module.append_jsonl
        outcome_tracker_module.append_jsonl = captured.append
        try:
            tracker = self._make_tracker_with_post_exit(has_real_update=False)
            asyncio.run(tracker._emit_post_exit_checkpoints())
        finally:
            outcome_tracker_module.append_jsonl = original_append

        checks = [r for r in captured if r["type"] == "post_exit_check"]
        self.assertEqual(len(checks), 1)
        self.assertIsNone(checks[0]["vs_realized_pct"])
        self.assertFalse(checks[0]["measured"])


class LiveExitTests(unittest.TestCase):
    def _make_tracker(self, *, client, dry_run, entry_ref=100.0, take_profit_pct=50.0):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, client=client, dry_run=dry_run,
            take_profit_pct=take_profit_pct,
        )
        tracker._pending["MINT"] = {
            "entry_ts": time.time(),
            "entry_ref": entry_ref,
            "last_ref": entry_ref,
            "peak_ref": entry_ref,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
        }
        return tracker, risk

    def test_successful_real_sell_closes_position(self):
        client = FakeClient(should_fail=False, signature="abc123")
        tracker, risk = self._make_tracker(client=client, dry_run=False, entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, crosses TP

        self.assertEqual(len(client.sell_calls), 1)
        self.assertEqual(client.sell_calls[0][0], "MINT")
        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.51, places=4)

    def test_failed_real_sell_leaves_position_open(self):
        client = FakeClient(should_fail=True)
        tracker, risk = self._make_tracker(client=client, dry_run=False, entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, crosses TP

        self.assertEqual(len(client.sell_calls), 1)
        # position must still be tracked - the wallet still genuinely holds it
        self.assertIn("MINT", tracker._pending)
        self.assertNotIn("MINT", tracker._post_exit)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.05)  # never closed
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_failed_sell_does_not_retry_within_cooldown(self):
        client = FakeClient(should_fail=True)
        tracker, risk = self._make_tracker(client=client, dry_run=False, entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 151.0))
        asyncio.run(tracker._handle_price_update("MINT", 152.0))  # still above TP, right after

        self.assertEqual(len(client.sell_calls), 1)  # second attempt was suppressed by cooldown

    def test_live_without_client_does_not_close_or_crash(self):
        tracker, risk = self._make_tracker(client=None, dry_run=False, entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 151.0))

        self.assertIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_dry_run_never_calls_the_client(self):
        client = FakeClient(should_fail=False)
        tracker, risk = self._make_tracker(client=client, dry_run=True, entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 151.0))

        self.assertEqual(client.sell_calls, [])
        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.51, places=4)


class TrailingStopTests(unittest.TestCase):
    def _make_tracker(
        self, *,
        take_profit_pct=100.0,   # kept high so TP doesn't preempt the trailing-stop tests
        stop_loss_pct=90.0,      # kept high so SL doesn't preempt the trailing-stop tests
        trailing_activation_pct=20.0,
        trailing_stop_pct=15.0,
        entry_ref=100.0,
    ):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk,
            take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct,
            trailing_activation_pct=trailing_activation_pct, trailing_stop_pct=trailing_stop_pct,
        )
        tracker._pending["MINT"] = {
            "entry_ts": time.time(),
            "entry_ref": entry_ref,
            "last_ref": entry_ref,
            "peak_ref": entry_ref,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
        }
        return tracker, risk

    def test_triggers_after_activation_and_drawdown_from_peak(self):
        tracker, risk = self._make_tracker(entry_ref=100.0)

        asyncio.run(tracker._handle_price_update("MINT", 140.0))  # +40%, arms trailing (>=20%)
        self.assertIn("MINT", tracker._pending)  # no exit yet, just tracking the peak

        asyncio.run(tracker._handle_price_update("MINT", 118.0))  # -15.7% from peak of 140

        self.assertNotIn("MINT", tracker._pending)
        # locked in +18% from entry, not the full round trip back to ~0/negative
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.18, places=4)

    def test_does_not_arm_before_activation_threshold(self):
        tracker, risk = self._make_tracker(entry_ref=100.0, trailing_activation_pct=20.0)

        asyncio.run(tracker._handle_price_update("MINT", 110.0))  # only +10%, below activation
        asyncio.run(tracker._handle_price_update("MINT", 95.0))   # drops, but never armed

        self.assertIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_peak_tracks_highest_price_not_last_price(self):
        tracker, risk = self._make_tracker(entry_ref=100.0, trailing_activation_pct=20.0, trailing_stop_pct=15.0)

        asyncio.run(tracker._handle_price_update("MINT", 150.0))  # peak = 150
        asyncio.run(tracker._handle_price_update("MINT", 145.0))  # small dip, doesn't trigger (-3.3%)
        self.assertIn("MINT", tracker._pending)
        self.assertEqual(tracker._pending["MINT"]["peak_ref"], 150.0)  # peak stays at 150, not 145

        asyncio.run(tracker._handle_price_update("MINT", 127.0))  # -15.3% from peak of 150 -> triggers
        self.assertNotIn("MINT", tracker._pending)

    def test_take_profit_still_takes_priority_over_trailing_stop(self):
        tracker, risk = self._make_tracker(
            entry_ref=100.0, take_profit_pct=50.0, trailing_activation_pct=20.0, trailing_stop_pct=15.0,
        )

        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # crosses TP before any trailing logic matters

        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.05 * 0.51, places=4)


if __name__ == "__main__":
    unittest.main()
