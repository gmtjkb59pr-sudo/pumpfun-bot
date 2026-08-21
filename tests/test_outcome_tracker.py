import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
import pumpfun_bot.position_store as position_store
from pumpfun_bot.config import RiskConfig
from pumpfun_bot.fees import ROUND_TRIP_PRIORITY_FEE_SOL, net_pct_change_after_fees
from pumpfun_bot.outcome_tracker import (
    CHECKPOINTS_SEC,
    MIN_SELL_DELAY_SEC,
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


class _FakeKeypair:
    def pubkey(self):
        return "FAKE_WALLET_PUBKEY"


class FakeClient:
    """Stands in for PumpPortalClient - no real network calls."""

    def __init__(self, *, should_fail=False, signature="fake_sig"):
        self.should_fail = should_fail
        self.signature = signature
        self.sell_calls = []
        self.keypair = _FakeKeypair()
        self.rpc_http_url = "https://example.invalid/rpc"

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


class TrackedMintsTests(unittest.TestCase):
    """Used at startup to reconcile against real wallet holdings (see
    wallet_reconciliation.py) - must expose exactly what's tracked, no
    more, no less."""

    def test_returns_the_set_of_currently_tracked_mints(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        self.assertEqual(tracker.tracked_mints(), set())
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        asyncio.run(tracker.track("MINT2", "Test2", "TEST2", entry_ref=100.0, trade_size_sol=0.03))
        self.assertEqual(tracker.tracked_mints(), {"MINT1", "MINT2"})


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

    def test_a_live_instance_never_resumes_a_dry_run_instance_s_position(self):
        # real bug this guards against: starting a live session loaded
        # leftover positions from an earlier dry-run farming session (never
        # real purchases) and tried to actually sell them - burning real
        # transaction fees on phantom positions and crowding out real
        # trading capacity. Neither tracker gets an explicit
        # position_store_path here - both must resolve their own default
        # (via position_store.path_for_mode) and never see each other's file.
        dry_run_tracker = OutcomeTracker(ws_url="wss://example.invalid", dry_run=True)
        asyncio.run(dry_run_tracker.track(
            "PHANTOM_MINT", "Simulated Token", "SIM", entry_ref=100.0, trade_size_sol=0.03,
        ))

        live_tracker = OutcomeTracker(ws_url="wss://example.invalid", dry_run=False)
        live_tracker.load_pending()

        self.assertNotIn("PHANTOM_MINT", live_tracker._pending)

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

    def test_does_not_crash_when_the_original_exit_was_unmeasured(self):
        # real crash found live: a position force-exited blind
        # (timeout_unmeasured/stale_price_unmeasured, realized_pct_change=
        # None) later started getting real post-exit price ticks - computing
        # vs_realized_pct then tried float - None and crashed the entire
        # bot (an unhandled exception here brings down the whole
        # asyncio.gather in main.py, abandoning every open position)
        import pumpfun_bot.outcome_tracker as outcome_tracker_module

        captured = []
        original_append = outcome_tracker_module.append_jsonl
        outcome_tracker_module.append_jsonl = captured.append
        try:
            tracker = self._make_tracker_with_post_exit(has_real_update=True, last_ref=180.0)
            tracker._post_exit["MINT"]["realized_pct_change"] = None
            tracker._post_exit["MINT"]["reason"] = "stale_price_unmeasured"
            asyncio.run(tracker._emit_post_exit_checkpoints())  # must not raise
        finally:
            outcome_tracker_module.append_jsonl = original_append

        checks = [r for r in captured if r["type"] == "post_exit_check"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["pct_change_if_held_from_entry"], 80.0)
        self.assertIsNone(checks[0]["vs_realized_pct"])

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


class MinSellDelayTests(unittest.TestCase):
    """Real bug found live: a real sell attempted too soon after the buy
    reliably came back SellZeroAmount - PumpPortal's own balance index
    hadn't caught up with our purchase yet - and still cost a real fee to
    fail. A sell must be deferred (not submitted at all) until
    MIN_SELL_DELAY_SEC has passed since entry, confirmed directly from a
    failed transaction's on-chain logs."""

    def _make_tracker(self, *, entry_ts: float):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        client = FakeClient(signature="sig")
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False,
            take_profit_pct=50.0,
        )
        tracker._pending["MINT"] = {
            "entry_ts": entry_ts,
            "entry_ref": 100.0,
            "last_ref": 100.0,
            "peak_ref": 100.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
        }
        return tracker, risk, client

    def test_defers_a_sell_attempted_too_soon_after_entry(self):
        tracker, risk, client = self._make_tracker(entry_ts=time.time())
        asyncio.run(tracker._handle_price_update("MINT", 151.0))  # +51%, crosses TP

        self.assertEqual(client.sell_calls, [])  # never even attempted
        self.assertIn("MINT", tracker._pending)  # still tracked, will retry
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.05)

    def test_attempts_a_sell_once_past_the_minimum_delay(self):
        tracker, risk, client = self._make_tracker(
            entry_ts=time.time() - MIN_SELL_DELAY_SEC - 1,
        )
        asyncio.run(tracker._handle_price_update("MINT", 151.0))

        self.assertEqual(len(client.sell_calls), 1)
        self.assertNotIn("MINT", tracker._pending)


class RunLoopResilienceTests(unittest.TestCase):
    """This loop manages real, live positions - a bug in any single
    connection attempt (confirmed live: a TypeError in post-exit checkpoint
    code) must never crash the whole bot. Before this fix, an unhandled
    exception here took down the entire asyncio.gather in main.py,
    abandoning every open position mid-session with no graceful shutdown at
    all."""

    def test_a_failing_connection_does_not_crash_the_run_loop(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        call_count = 0

        async def _failing_then_ok_connection():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated bug in a connection cycle")

        async def _drive():
            with patch.object(tracker, "_run_connection", _failing_then_ok_connection), patch(
                "pumpfun_bot.outcome_tracker.RECONNECT_BACKOFF_SEC", 0,
            ):
                try:
                    await asyncio.wait_for(tracker.run(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass  # expected - run() never returns on its own

        asyncio.run(_drive())

        # proves the loop survived the RuntimeError and kept iterating,
        # rather than the exception propagating out of run() entirely
        self.assertGreaterEqual(call_count, 2)


class FakeWebSocket:
    """Stands in for a websockets connection - only the .send() calls made
    by _sync_subscription matter for these tests, nothing is actually sent
    anywhere. Async-iterable but never actually yields a message, so a test
    driving _run_connection() under asyncio.wait_for() reliably times out
    instead of raising TypeError for not being iterable."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


class SubscriptionSyncTests(unittest.TestCase):
    """A single WS connection now stays open for the tracker's whole
    lifetime instead of reconnecting every cycle (see _run_connection's
    docstring for why that mattered) - _sync_subscription is what lets a
    position opened mid-connection start getting price ticks immediately,
    by adding just the newly-tracked mint to the existing subscription
    rather than waiting for a reconnect that may not happen for a long
    time."""

    def test_subscribes_to_newly_tracked_mints(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))

        asyncio.run(tracker._sync_subscription(ws))

        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(ws.sent[0]["method"], "subscribeTokenTrade")
        self.assertEqual(ws.sent[0]["keys"], ["MINT1"])
        self.assertEqual(tracker._subscribed_mints, {"MINT1"})

    def test_does_not_resend_a_mint_already_subscribed(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        asyncio.run(tracker._sync_subscription(ws))

        asyncio.run(tracker._sync_subscription(ws))  # nothing new tracked since

        self.assertEqual(len(ws.sent), 1)  # still just the one call

    def test_only_sends_the_newly_tracked_mint_on_a_later_sync(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        asyncio.run(tracker._sync_subscription(ws))

        asyncio.run(tracker.track("MINT2", "Test2", "TEST2", entry_ref=100.0, trade_size_sol=0.03))
        asyncio.run(tracker._sync_subscription(ws))

        self.assertEqual(len(ws.sent), 2)
        self.assertEqual(ws.sent[1]["keys"], ["MINT2"])  # only the new one, not MINT1 again
        self.assertEqual(tracker._subscribed_mints, {"MINT1", "MINT2"})

    def test_a_fresh_connection_resets_the_subscribed_set(self):
        # a new WS connection has no existing subscription, even if the
        # previous connection had already subscribed to these mints - a
        # stale _subscribed_mints would make _sync_subscription think the
        # new connection already knows about them and skip resubscribing,
        # silently going blind on every position after a reconnect
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        tracker._subscribed_mints = {"STALE_FROM_OLD_CONNECTION"}
        asyncio.run(tracker.track("MINT1", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))
        ws = FakeWebSocket()

        class _Ctx:
            async def __aenter__(self_inner):
                return ws

            async def __aexit__(self_inner, *exc):
                return False

        def _fake_connect(*args, **kwargs):
            return _Ctx()

        async def _drive():
            with patch("pumpfun_bot.outcome_tracker.websockets.connect", side_effect=_fake_connect):
                try:
                    await asyncio.wait_for(tracker._run_connection(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass  # expected - _run_connection blocks on `async for raw in ws`

        asyncio.run(_drive())

        self.assertEqual(tracker._subscribed_mints, {"MINT1"})
        self.assertEqual(ws.sent[0]["keys"], ["MINT1"])


class WsMessageHandlingTests(unittest.TestCase):
    """_handle_ws_message replaces the per-cycle parsing that used to live
    inline in the old reconnect-every-cycle _poll_once - same parsing
    behavior, now driven by a persistent `async for` over one long-lived
    connection instead of a fresh 8s-window connection every ~11s."""

    def test_a_trade_event_updates_the_tracked_position(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.track("MINT", "Test", "TEST", entry_ref=100.0, trade_size_sol=0.03))

        raw = json.dumps({"mint": "MINT", "price": 110.0})
        asyncio.run(tracker._handle_ws_message(raw))

        self.assertEqual(tracker._pending["MINT"]["last_ref"], 110.0)

    def test_malformed_json_is_ignored_without_raising(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker._handle_ws_message("not json"))  # must not raise

    def test_funded_key_rejection_sets_the_dashboard_flag(self):
        tracker = OutcomeTracker(ws_url="wss://example.invalid")
        raw = json.dumps({"message": "This feed requires a funded API key."})

        asyncio.run(tracker._handle_ws_message(raw))

        self.assertTrue(tracker._warned_no_access)


class ReconcileWithWalletTests(unittest.TestCase):
    """Real bug found live: a manually-sold position kept getting retried
    forever - nothing was left to sell, so every attempt failed with
    SellZeroAmount and still cost a real fee. _reconcile_with_wallet() must
    drop tracking for anything the wallet no longer actually holds, and
    surface (without auto-adopting) anything held that isn't tracked."""

    def _make_tracker(self):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.03)
        client = FakeClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)
        tracker._pending["SOLD_MINT"] = {
            "entry_ts": time.time(), "entry_ref": 100.0, "last_ref": 100.0, "peak_ref": 100.0,
            "name": "Sold Elsewhere", "symbol": "SOLD", "trade_size_sol": 0.03,
            "hit": set(), "has_real_update": False,
        }
        return tracker, risk

    def test_a_single_miss_does_not_drop_the_position(self):
        # confirmed live: a genuinely still-held position (verified directly
        # on-chain) was missing from one fetch_wallet_token_mints() call and
        # present again the very next cycle, nothing bought/sold in between
        # - a single miss must never be enough to abandon a real position
        tracker, risk = self._make_tracker()

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return set()  # wallet appears to hold nothing (one incomplete read)

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
            asyncio.run(tracker._reconcile_with_wallet())

        self.assertIn("SOLD_MINT", tracker._pending)
        self.assertEqual(tracker._pending["SOLD_MINT"]["wallet_miss_streak"], 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)  # not released yet

    def test_drops_a_position_the_wallet_no_longer_holds(self):
        from pumpfun_bot.outcome_tracker import WALLET_MISS_CONFIRMATION_COUNT

        tracker, risk = self._make_tracker()

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return set()  # wallet holds nothing

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
            for _ in range(WALLET_MISS_CONFIRMATION_COUNT):
                asyncio.run(tracker._reconcile_with_wallet())

        self.assertNotIn("SOLD_MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)  # slot released

    def test_a_miss_streak_resets_once_the_position_is_seen_again(self):
        tracker, risk = self._make_tracker()

        async def _missing(wallet_pubkey, rpc_http_url):
            return set()

        async def _present(wallet_pubkey, rpc_http_url):
            return {"SOLD_MINT"}

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _missing):
            asyncio.run(tracker._reconcile_with_wallet())
        self.assertEqual(tracker._pending["SOLD_MINT"]["wallet_miss_streak"], 1)

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _present):
            asyncio.run(tracker._reconcile_with_wallet())
        self.assertEqual(tracker._pending["SOLD_MINT"]["wallet_miss_streak"], 0)
        self.assertIn("SOLD_MINT", tracker._pending)

    def test_keeps_a_position_the_wallet_still_holds(self):
        tracker, risk = self._make_tracker()

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return {"SOLD_MINT"}

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
            asyncio.run(tracker._reconcile_with_wallet())

        self.assertIn("SOLD_MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_does_nothing_when_the_lookup_fails(self):
        # unknown state, not "wallet holds nothing" - must not drop anything
        tracker, risk = self._make_tracker()

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return None

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
            asyncio.run(tracker._reconcile_with_wallet())

        self.assertIn("SOLD_MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_dry_run_never_calls_the_wallet(self):
        risk = RiskManager(RiskConfig())
        client = FakeClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=True)
        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints") as mock_fetch:
            asyncio.run(tracker._reconcile_with_wallet_if_due())
            mock_fetch.assert_not_called()

    def test_skips_reconciliation_before_the_interval_elapses(self):
        risk = RiskManager(RiskConfig())
        client = FakeClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)
        tracker._last_wallet_reconcile_ts = time.time()  # just reconciled
        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints") as mock_fetch:
            asyncio.run(tracker._reconcile_with_wallet_if_due())
            mock_fetch.assert_not_called()

    def test_reconciles_once_the_interval_has_passed(self):
        from pumpfun_bot.outcome_tracker import WALLET_RECONCILE_INTERVAL_SEC

        risk = RiskManager(RiskConfig())
        client = FakeClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)
        tracker._last_wallet_reconcile_ts = time.time() - WALLET_RECONCILE_INTERVAL_SEC - 1

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return set()

        with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
            asyncio.run(tracker._reconcile_with_wallet_if_due())

        self.assertGreater(tracker._last_wallet_reconcile_ts, time.time() - 1)


class LiquidationDispatchTests(unittest.TestCase):
    """Real bug found live: _reconcile_with_wallet() used to await the
    liquidation sweep inline - since a failing sell can take up to 30s to
    time out, and the sweep tries every untracked mint in sequence, this
    blocked the ENTIRE housekeeping loop (checkpoints, stale-price
    detection for REAL tracked positions) for as long as the sweep took.
    A real position's stale-price detection was delayed from the expected
    ~10-15s to 44s because of exactly this. The sweep must run as a
    detached background task, never blocking reconciliation itself."""

    class _SlowFailingClient(FakeClient):
        async def build_and_send_full_sell(self, mint, slippage_pct):
            await asyncio.sleep(0.2)
            raise RuntimeError("simulated slow failure")

    def test_reconcile_returns_promptly_even_with_a_slow_liquidation_attempt(self):
        client = self._SlowFailingClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=False)

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return {"UNTRACKED_MINT"}

        async def _drive():
            with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
                # would time out here if the sweep were awaited inline -
                # the fake client's sell takes 0.2s, well over this bound
                await asyncio.wait_for(tracker._reconcile_with_wallet(), timeout=0.05)
                self.assertIsNotNone(tracker._liquidation_task)
                self.assertFalse(tracker._liquidation_task.done())
                await asyncio.wait_for(tracker._liquidation_task, timeout=1.0)  # avoid leaking

        asyncio.run(_drive())

    def test_does_not_start_a_second_sweep_while_one_is_in_flight(self):
        client = self._SlowFailingClient()
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=False)

        async def _fake_fetch(wallet_pubkey, rpc_http_url):
            return {"UNTRACKED_MINT"}

        async def _drive():
            with patch("pumpfun_bot.outcome_tracker.fetch_wallet_token_mints", _fake_fetch):
                await tracker._reconcile_with_wallet()
                first_task = tracker._liquidation_task
                await tracker._reconcile_with_wallet()  # sweep still in flight
                self.assertIs(tracker._liquidation_task, first_task)
                await asyncio.wait_for(first_task, timeout=1.0)

        asyncio.run(_drive())


class LiquidateUntrackedHoldingsTests(unittest.TestCase):
    """Untracked wallet holdings (leftovers from earlier sessions/bugs, no
    known entry price) are never auto-adopted as real positions - see
    _reconcile_with_wallet's docstring - but sitting there forever as
    unmanaged capital isn't useful either. _liquidate_untracked_holdings is
    a best-effort "convert back to SOL" cleanup pass, deliberately separate
    from risk/exposure since these were never counted as open exposure."""

    def test_successful_liquidation_removes_tracking_state_and_logs(self):
        client = FakeClient(should_fail=False, signature="liq_sig")
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=False)

        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))

        self.assertEqual(len(client.sell_calls), 1)
        self.assertEqual(client.sell_calls[0][0], "UNTRACKED_MINT")
        self.assertNotIn("UNTRACKED_MINT", tracker._untracked_liquidation)

    def test_never_touches_risk_or_exposure(self):
        # these were never counted as open exposure to begin with - a
        # successful (or failed) liquidation must not change it either way
        risk = RiskManager(RiskConfig())
        client = FakeClient(should_fail=False)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)

        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, 0.0)

    def test_dry_run_never_attempts_liquidation(self):
        client = FakeClient(should_fail=False)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=True)

        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))

        self.assertEqual(client.sell_calls, [])

    def test_failed_liquidation_is_retried_up_to_the_cap_then_paused(self):
        from pumpfun_bot.outcome_tracker import MAX_CONSECUTIVE_SELL_FAILURES

        client = FakeClient(should_fail=True)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=False)

        for _ in range(MAX_CONSECUTIVE_SELL_FAILURES):
            asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))
            tracker._untracked_liquidation["UNTRACKED_MINT"]["last_attempt_ts"] = 0  # bypass cooldown

        self.assertTrue(tracker._untracked_liquidation["UNTRACKED_MINT"]["paused"])
        calls_before = len(client.sell_calls)

        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))

        self.assertEqual(len(client.sell_calls), calls_before)  # paused, no further attempts

    def test_respects_the_retry_cooldown(self):
        client = FakeClient(should_fail=True)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", client=client, dry_run=False)

        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))
        asyncio.run(tracker._liquidate_untracked_holdings({"UNTRACKED_MINT"}))  # immediately again

        self.assertEqual(len(client.sell_calls), 1)  # second attempt suppressed by cooldown


class LiveExitTests(unittest.TestCase):
    def _make_tracker(self, *, client, dry_run, entry_ref=100.0, take_profit_pct=50.0):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(
            ws_url="wss://example.invalid", risk=risk, client=client, dry_run=dry_run,
            take_profit_pct=take_profit_pct,
        )
        tracker._pending["MINT"] = {
            # past MIN_SELL_DELAY_SEC - a real sell attempt right at buy
            # time would be deferred (PumpPortal's own balance index hasn't
            # caught up yet), which isn't what these tests are exercising
            "entry_ts": time.time() - MIN_SELL_DELAY_SEC - 1,
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


class SellFailurePauseTests(unittest.TestCase):
    """Real bug found live: a position hit pump.fun's own program throwing
    'AnchorError ... Error Code: Overflow' (Custom 6024) on every sell
    attempt - a deterministic on-chain error, not a transient one, so it
    failed identically 18 times in ~3 minutes before being stopped by hand,
    burning a real priority fee on every doomed attempt. Nothing capped
    that retry loop before this - MAX_CONSECUTIVE_SELL_FAILURES must stop
    auto-retrying once a sell has failed enough times in a row, while still
    leaving the position tracked so it isn't silently abandoned."""

    def _make_tracker(self, *, client):
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)
        info = {
            "entry_ts": time.time() - MIN_SELL_DELAY_SEC - 1,
            "entry_ref": 100.0, "last_ref": 151.0, "peak_ref": 151.0,
            "name": "Test Token", "symbol": "TEST", "trade_size_sol": 0.05,
            "hit": set(), "has_real_update": True,
        }
        tracker._pending["MINT"] = info
        return tracker, risk, info

    def test_pauses_after_max_consecutive_failures(self):
        from pumpfun_bot.outcome_tracker import MAX_CONSECUTIVE_SELL_FAILURES

        client = FakeClient(should_fail=True)
        tracker, risk, info = self._make_tracker(client=client)

        for _ in range(MAX_CONSECUTIVE_SELL_FAILURES):
            asyncio.run(tracker._exit("MINT", dict(tracker._pending["MINT"]), "stale_price", 51.0))

        self.assertEqual(tracker._pending["MINT"]["consecutive_sell_failures"], MAX_CONSECUTIVE_SELL_FAILURES)
        self.assertTrue(tracker._pending["MINT"]["sell_paused"])
        # position stays tracked/visible - never silently dropped by this
        self.assertIn("MINT", tracker._pending)

    def test_stops_attempting_once_paused(self):
        from pumpfun_bot.outcome_tracker import MAX_CONSECUTIVE_SELL_FAILURES

        client = FakeClient(should_fail=True)
        tracker, risk, info = self._make_tracker(client=client)

        for _ in range(MAX_CONSECUTIVE_SELL_FAILURES):
            asyncio.run(tracker._exit("MINT", dict(tracker._pending["MINT"]), "stale_price", 51.0))
        calls_before = len(client.sell_calls)

        # cooldown bypassed directly - even so, a paused position must
        # refuse to attempt at all
        tracker._pending["MINT"]["last_exit_attempt_ts"] = 0
        allowed = tracker._exit_attempt_allowed(tracker._pending["MINT"])

        self.assertFalse(allowed)
        self.assertEqual(len(client.sell_calls), calls_before)

    def test_does_not_pause_below_the_threshold(self):
        from pumpfun_bot.outcome_tracker import MAX_CONSECUTIVE_SELL_FAILURES

        client = FakeClient(should_fail=True)
        tracker, risk, info = self._make_tracker(client=client)

        for _ in range(MAX_CONSECUTIVE_SELL_FAILURES - 1):
            asyncio.run(tracker._exit("MINT", dict(tracker._pending["MINT"]), "stale_price", 51.0))

        self.assertFalse(tracker._pending["MINT"].get("sell_paused", False))

    def test_a_deferred_sell_min_sell_delay_does_not_count_as_a_failure(self):
        # too-soon-since-entry is a deliberate defer (see MIN_SELL_DELAY_SEC),
        # not a real failure - it must never count toward the pause threshold
        client = FakeClient(should_fail=False)
        risk = RiskManager(RiskConfig())
        risk.register_trade_opened(0.05)
        tracker = OutcomeTracker(ws_url="wss://example.invalid", risk=risk, client=client, dry_run=False)
        info = {
            "entry_ts": time.time(),  # well within MIN_SELL_DELAY_SEC
            "entry_ref": 100.0, "last_ref": 151.0, "peak_ref": 151.0,
            "name": "Test Token", "symbol": "TEST", "trade_size_sol": 0.05,
            "hit": set(), "has_real_update": True,
        }
        tracker._pending["MINT"] = info

        closed = asyncio.run(tracker._exit("MINT", dict(info), "stale_price", 51.0))

        self.assertFalse(closed)
        self.assertEqual(client.sell_calls, [])  # never even attempted
        self.assertEqual(tracker._pending["MINT"].get("consecutive_sell_failures", 0), 0)

    def test_a_successful_sell_after_failures_closes_normally(self):
        # not a resume-from-pause scenario (paused stays paused - see
        # _exit_attempt_allowed) - just confirms failures short of the
        # threshold don't prevent a later real success from closing cleanly
        client = FakeClient(should_fail=True)
        tracker, risk, info = self._make_tracker(client=client)
        asyncio.run(tracker._exit("MINT", dict(tracker._pending["MINT"]), "stale_price", 51.0))
        self.assertEqual(tracker._pending["MINT"]["consecutive_sell_failures"], 1)

        client.should_fail = False
        asyncio.run(tracker._attempt_exit("MINT", dict(tracker._pending["MINT"]), "stale_price", 51.0))

        self.assertNotIn("MINT", tracker._pending)


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
        recent = time.time() - min(1, STALE_PRICE_TIMEOUT_SEC / 2)
        tracker._pending["MINT"] = {
            "entry_ts": recent,
            "entry_ref": 100.0,
            "last_ref": 112.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": True,
            "last_update_ts": recent,  # well under the threshold, whatever it's set to
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
        # past both STALE_PRICE_TIMEOUT_SEC (so staleness is detected) and
        # MIN_SELL_DELAY_SEC (so the resulting sell attempt isn't deferred)
        stale_and_sellable = max(STALE_PRICE_TIMEOUT_SEC, MIN_SELL_DELAY_SEC) + 1
        tracker._pending["MINT"] = {
            "entry_ts": time.time() - stale_and_sellable,
            "entry_ref": 100.0,
            "last_ref": 100.0,
            "name": "Test Token",
            "symbol": "TEST",
            "trade_size_sol": 0.05,
            "hit": set(),
            "has_real_update": False,
            "last_update_ts": time.time() - stale_and_sellable,
        }
        asyncio.run(tracker._emit_due_checkpoints())

        self.assertEqual(client.sell_calls, [("MINT", tracker.sell_slippage_pct)])
        self.assertNotIn("MINT", tracker._pending)
        self.assertAlmostEqual(risk.state.realized_pnl_sol, -ROUND_TRIP_PRIORITY_FEE_SOL)


if __name__ == "__main__":
    unittest.main()
