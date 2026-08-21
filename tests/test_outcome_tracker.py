import asyncio
import tempfile
import time
import unittest
from pathlib import Path

import pumpfun_bot.activity_log as activity_log
import pumpfun_bot.position_store as position_store
from pumpfun_bot.config import RiskConfig
from pumpfun_bot.fees import ROUND_TRIP_PRIORITY_FEE_SOL, net_pct_change_after_fees
from pumpfun_bot.outcome_tracker import (
    CHECKPOINTS_SEC,
    STALE_PRICE_TIMEOUT_SEC,
    OutcomeTracker,
    is_funded_key_rejection,
)
from pumpfun_bot.risk import RiskManager


def _net_pnl_sol(trade_size_sol: float, gross_pct_change: float) -> float:
    return round(trade_size_sol * (net_pct_change_after_fees(gross_pct_change) / 100), 6)

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH
_ORIGINAL_STORE_PATH = position_store.DEFAULT_STORE_PATH
_TEST_LOG_FILE = None
_TEST_STORE_FILE = None


def setUpModule():
    # append_jsonl() always writes to activity_log.DATA_LOG_PATH, and
    # OutcomeTracker persists open positions to position_store.DEFAULT_STORE_PATH
    # by default - most tests in this file exercise the real un-mocked paths,
    # so without this every run of this module would write junk records into
    # the real, live activity_log.jsonl / data/open_positions.json
    global _TEST_LOG_FILE, _TEST_STORE_FILE
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)

    _TEST_STORE_FILE = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    _TEST_STORE_FILE.close()
    position_store.DEFAULT_STORE_PATH = Path(_TEST_STORE_FILE.name)


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)

    position_store.DEFAULT_STORE_PATH = _ORIGINAL_STORE_PATH
    if _TEST_STORE_FILE is not None:
        Path(_TEST_STORE_FILE.name).unlink(missing_ok=True)


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


class PerPositionThresholdTests(unittest.TestCase):
    """sniper and social_watch share one OutcomeTracker instance, which only
    has ONE set of take_profit_pct/stop_loss_pct/trailing_*_pct - found live
    that changing social_watch's config values did nothing at all, because
    nothing ever passed them into track(). Exit thresholds must be stored
    per-position (from whichever strategy actually opened it), not just
    read from the shared instance's own defaults."""

    def _make_tracker(self, *, instance_take_profit_pct=50.0):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, take_profit_pct=instance_take_profit_pct,
        )
        return tracker, risk

    def test_track_stores_explicit_thresholds_on_the_position(self):
        tracker, _ = self._make_tracker(instance_take_profit_pct=50.0)
        asyncio.run(tracker.track(
            "MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.05,
            take_profit_pct=100.0, stop_loss_pct=25.0,
            trailing_activation_pct=20.0, trailing_stop_pct=15.0,
        ))
        self.assertEqual(tracker._pending["MINT"]["take_profit_pct"], 100.0)

    def test_track_falls_back_to_instance_default_when_not_given(self):
        tracker, _ = self._make_tracker(instance_take_profit_pct=50.0)
        asyncio.run(tracker.track("MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.05))
        self.assertEqual(tracker._pending["MINT"]["take_profit_pct"], 50.0)

    def test_a_100pct_target_does_not_exit_at_the_instance_default_of_50pct(self):
        # the actual bug: instance-wide take_profit_pct=50 (as if constructed
        # for sniper), but THIS position was opened with an explicit 100%
        # target (as if by social_watch) - a +51% move must NOT trigger
        # take-profit for this position, even though it would cross the
        # tracker's own instance-level default
        tracker, risk = self._make_tracker(instance_take_profit_pct=50.0)
        asyncio.run(tracker.track(
            "MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.05,
            take_profit_pct=100.0, stop_loss_pct=25.0,
            trailing_activation_pct=20.0, trailing_stop_pct=15.0,
        ))
        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%

        self.assertIn("MINT", tracker._pending)  # still open, no exit yet
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_exits_at_its_own_100pct_target_once_actually_crossed(self):
        tracker, risk = self._make_tracker(instance_take_profit_pct=50.0)
        asyncio.run(tracker.track(
            "MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.05,
            take_profit_pct=100.0, stop_loss_pct=25.0,
            trailing_activation_pct=20.0, trailing_stop_pct=15.0,
        ))
        asyncio.run(tracker._handle_price_update("MINT", 201.0))  # +101%

        self.assertNotIn("MINT", tracker._pending)
        self.assertGreater(risk.state.realized_pnl_sol, 0.0)

    def test_old_persisted_positions_without_the_field_use_instance_defaults(self):
        # backward compatibility: a position tracked before this fix existed
        # (or a hand-built test fixture) has no take_profit_pct key at all
        tracker, risk = self._make_tracker(instance_take_profit_pct=50.0)
        tracker._pending["MINT"] = {
            "entry_ts": 0.0, "entry_ref": 100.0, "last_ref": 100.0, "peak_ref": 100.0,
            "name": "Test", "symbol": "TEST", "trade_size_sol": 0.05,
            "hit": set(), "has_real_update": False,
        }
        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, crosses the 50% default

        self.assertNotIn("MINT", tracker._pending)
        self.assertGreater(risk.state.realized_pnl_sol, 0.0)


class OpenPositionCountTests(unittest.TestCase):
    """max_open_positions must reflect what's actually held, not a separate
    counter that can drift - e.g. if track() ever returns early (no price
    ref available), a naive "increment on every buy" counter would
    permanently believe a slot is taken even though nothing is being
    tracked/managed there, and real space would never re-open."""

    def test_reflects_real_tracked_positions(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        self.assertEqual(tracker.open_position_count(), 0)
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        self.assertEqual(tracker.open_position_count(), 1)
        asyncio.run(tracker.track("MINT2", "Test2", "TEST2", entry_ref=100.0, trade_size_sol=0.03))
        self.assertEqual(tracker.open_position_count(), 2)

    def test_does_not_count_a_track_call_that_returned_early(self):
        # entry_ref=None is a real, documented early-return case in track() -
        # nothing gets added to _pending, so open_position_count() must not
        # count it either, unlike a separately-incremented "buys attempted"
        # counter would
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=None, trade_size_sol=0.03))
        self.assertEqual(tracker.open_position_count(), 0)

    def test_decreases_after_a_position_closes(self):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, take_profit_pct=50.0,
        )
        asyncio.run(tracker.track("MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.05))
        self.assertEqual(tracker.open_position_count(), 1)
        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, take-profit
        self.assertEqual(tracker.open_position_count(), 0)


class SharedTrackerCollisionTests(unittest.TestCase):
    """Two strategies (sniper, social_watch) share one OutcomeTracker keyed
    only by mint - if both ever bought the same mint, the second track()
    call would silently overwrite the first's entry_ref/P&L and leak
    exposure that would never get released (only one _pending entry exists
    to ever register_trade_closed against)."""

    def test_is_tracking_reflects_an_open_position(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        self.assertFalse(tracker.is_tracking("MINT"))
        asyncio.run(tracker.track("MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        self.assertTrue(tracker.is_tracking("MINT"))

    def test_second_track_call_for_same_mint_is_ignored(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.track("MINT", "First Buyer", "FIRST", entry_ref=100.0, trade_size_sol=0.03))
        asyncio.run(tracker.track("MINT", "Second Buyer", "SECOND", entry_ref=999.0, trade_size_sol=0.05))

        self.assertEqual(tracker._pending["MINT"]["name"], "First Buyer")
        self.assertEqual(tracker._pending["MINT"]["entry_ref"], 100.0)
        self.assertEqual(tracker._pending["MINT"]["trade_size_sol"], 0.03)


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
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 20.0))
        self.assertNotIn("MINT", tracker._pending)

    def test_leaves_position_open_when_unmeasured(self):
        tracker, risk = self._make_tracker_with_pending(has_real_update=False)
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.05)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_live_unmeasured_position_gets_force_sold_not_abandoned(self):
        # a real held position with no price data and no manual-sell path
        # anywhere in the bot must not be left stuck open forever - it should
        # get force-sold blind, with pnl recorded as 0 (genuinely unknown)
        # rather than guessed
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        client = FakeClient(signature="blind_sell_sig")
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False,
        )
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - CHECKPOINTS_SEC[-1] - 1,
            "entry_ref": 100.0,
            "last_ref": 100.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(CHECKPOINTS_SEC[:-1]),
            "has_real_update": False,
        }
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertEqual(client.sell_calls, [("MINT", tracker.sell_slippage_pct)])
        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        # unknown price -> 0 pnl from the trade itself, but the real priority
        # fee on the forced sell (and its earlier buy) still gets subtracted
        self.assertAlmostEqual(risk.state.realized_pnl_sol, -ROUND_TRIP_PRIORITY_FEE_SOL)


class PositionPersistenceAcrossRestartTests(unittest.TestCase):
    """A restart used to silently abandon whatever was open at that exact
    moment - the position was never sold wrong, it was just never looked at
    again by any process. This is the regression guard for the fix: a new
    OutcomeTracker instance pointed at the same store file must pick up
    exactly where the old one left off."""

    def _store_path(self):
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        path = Path(f.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_new_instance_resumes_a_position_the_old_one_opened(self):
        path = self._store_path()

        tracker_before_restart = OutcomeTracker(
            ws_url="wss://example.invalid", position_store_path=path,
        )
        asyncio.run(tracker_before_restart.track(
            "MINT", "Test Token", "TEST", entry_ref=100.0, trade_size_sol=0.03,
        ))
        # simulates the process dying right here - tracker_before_restart is
        # simply discarded, nothing more happens to it

        tracker_after_restart = OutcomeTracker(
            ws_url="wss://example.invalid", position_store_path=path,
        )
        tracker_after_restart.load_pending()

        self.assertIn("MINT", tracker_after_restart._pending)
        self.assertEqual(tracker_after_restart._pending["MINT"]["entry_ref"], 100.0)
        self.assertEqual(tracker_after_restart._pending["MINT"]["trade_size_sol"], 0.03)

    def test_a_closed_position_is_not_resumed_after_restart(self):
        path = self._store_path()
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker_before_restart = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, position_store_path=path,
            take_profit_pct=50.0,
        )
        asyncio.run(tracker_before_restart.track(
            "MINT", "Test Token", "TEST", entry_ref=100.0, trade_size_sol=0.05,
        ))
        asyncio.run(tracker_before_restart._handle_price_update("MINT", 151.0))  # take-profit exit

        tracker_after_restart = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=path)
        tracker_after_restart.load_pending()

        self.assertNotIn("MINT", tracker_after_restart._pending)

    def test_fresh_start_with_no_store_file_has_nothing_to_resume(self):
        path = self._store_path()
        path.unlink()  # no file at all - first run ever
        tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=path)
        tracker.load_pending()
        self.assertEqual(tracker._pending, {})


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
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 51.0), places=4)

    def test_stop_loss_closes_position_immediately(self):
        tracker, risk = self._make_tracker(stop_loss_pct=25.0, entry_ref=100.0)
        asyncio.run(tracker._handle_price_update("MINT", 70.0))  # -30%, crosses -25%

        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, -30.0), places=4)

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
        self.assertAlmostEqual(
            risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 51.0) - ROUND_TRIP_PRIORITY_FEE_SOL, places=4
        )

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
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 51.0), places=4)


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
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 18.0), places=4)

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
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 51.0), places=4)


class StalePriceExitTests(unittest.TestCase):
    def test_exits_with_last_known_price_after_going_quiet(self):
        # got real ticks early on, then nothing for STALE_PRICE_TIMEOUT_SEC -
        # likely dead/rugged, shouldn't wait for the full MAX_HOLD_SEC timeout
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, dry_run=True)
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - STALE_PRICE_TIMEOUT_SEC - 1,
            "entry_ref": 100.0,
            "last_ref": 112.0,  # last real tick was +12%
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": True,
            "last_update_ts": time.time() - STALE_PRICE_TIMEOUT_SEC - 1,
        }
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, _net_pnl_sol(0.05, 12.0), places=4)

    def test_does_not_exit_before_stale_threshold(self):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, dry_run=True)
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - 10,
            "entry_ref": 100.0,
            "last_ref": 112.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": True,
            "last_update_ts": time.time() - 10,  # recent, well under the threshold
        }
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_live_never_measured_position_force_sold_after_stale_threshold(self):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        client = FakeClient(signature="stale_blind_sell_sig")
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False,
        )
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - STALE_PRICE_TIMEOUT_SEC - 1,
            "entry_ref": 100.0,
            "last_ref": 100.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
            "last_update_ts": time.time() - STALE_PRICE_TIMEOUT_SEC - 1,
        }
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertEqual(client.sell_calls, [("MINT", tracker.sell_slippage_pct)])
        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, -ROUND_TRIP_PRIORITY_FEE_SOL)


if __name__ == "__main__":
    unittest.main()
