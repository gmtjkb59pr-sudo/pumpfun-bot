import asyncio
import json
import time
import unittest
from unittest.mock import patch

from pumpfun_bot.candidate_price_tracker import CandidatePriceTracker


class PriceChangePctTests(unittest.TestCase):
    """price_change_pct must never guess at a window shorter than requested -
    only a real buffered tick at least window_sec old counts as a valid base."""

    def test_returns_none_when_nothing_has_been_watched(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        self.assertIsNone(tracker.price_change_pct("MINT", 60))

    def test_returns_none_when_watched_but_no_ticks_arrived_yet(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))
        self.assertIsNone(tracker.price_change_pct("MINT", 60))

    def test_returns_none_when_no_tick_is_old_enough_for_the_window(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._history["MINT"] = [(now - 10, 100.0), (now, 110.0)]
        self.assertIsNone(tracker.price_change_pct("MINT", 60))

    def test_computes_pct_change_from_the_oldest_tick_covering_the_window(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._history["MINT"] = [
            (now - 90, 100.0),  # covers the 60s window
            (now - 30, 150.0),  # too recent to be the base for a 60s window
            (now, 120.0),
        ]
        pct = tracker.price_change_pct("MINT", 60)
        self.assertAlmostEqual(pct, 20.0)

    def test_returns_none_when_base_price_is_zero_or_negative(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._history["MINT"] = [(now - 90, 0.0), (now, 120.0)]
        self.assertIsNone(tracker.price_change_pct("MINT", 60))


class WatchUnwatchTests(unittest.TestCase):
    def test_watch_initializes_an_empty_history_bucket(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))
        self.assertIn("MINT", tracker._history)
        self.assertIn("MINT", tracker._watched)

    def test_unwatch_removes_from_watched_but_keeps_history(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))
        tracker._history["MINT"] = [(time.time(), 100.0)]

        asyncio.run(tracker.unwatch("MINT"))

        self.assertNotIn("MINT", tracker._watched)
        self.assertIn("MINT", tracker._history)  # a pending _buy() can still read it


class PruneOldHistoryTests(unittest.TestCase):
    def test_drops_ticks_older_than_the_retention_window(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._watched = {"MINT"}
        tracker._history["MINT"] = [(now - 99999, 100.0), (now, 110.0)]

        tracker._prune_old_history()

        self.assertEqual(len(tracker._history["MINT"]), 1)
        self.assertAlmostEqual(tracker._history["MINT"][0][1], 110.0)

    def test_deletes_an_unwatched_mint_once_all_its_ticks_age_out(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._watched = set()  # already unwatched
        tracker._history["MINT"] = [(now - 99999, 100.0)]

        tracker._prune_old_history()

        self.assertNotIn("MINT", tracker._history)

    def test_keeps_a_still_watched_mint_even_with_no_ticks_left(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        now = time.time()
        tracker._watched = {"MINT"}
        tracker._history["MINT"] = [(now - 99999, 100.0)]

        tracker._prune_old_history()

        self.assertIn("MINT", tracker._history)
        self.assertEqual(tracker._history["MINT"], [])


class HandleWsMessageTests(unittest.TestCase):
    def test_appends_a_tick_for_a_watched_mint(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))

        asyncio.run(tracker._handle_ws_message(json.dumps({"mint": "MINT", "marketCapSol": 42.0})))

        self.assertEqual(len(tracker._history["MINT"]), 1)
        self.assertAlmostEqual(tracker._history["MINT"][0][1], 42.0)

    def test_ignores_a_tick_for_a_mint_that_was_never_watched(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")

        asyncio.run(tracker._handle_ws_message(json.dumps({"mint": "UNTRACKED", "marketCapSol": 42.0})))

        self.assertNotIn("UNTRACKED", tracker._history)

    def test_ignores_malformed_json(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))

        asyncio.run(tracker._handle_ws_message("not json"))

        self.assertEqual(tracker._history["MINT"], [])

    def test_ignores_a_message_without_an_extractable_price_ref(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        asyncio.run(tracker.watch("MINT"))

        asyncio.run(tracker._handle_ws_message(json.dumps({"mint": "MINT"})))

        self.assertEqual(tracker._history["MINT"], [])


class FakeWebSocket:
    """Same fake used by test_outcome_tracker.py's SubscriptionSyncTests -
    only .send() calls matter here, the async iteration never actually
    yields so a test driving _run_connection() under asyncio.wait_for()
    reliably times out instead of raising TypeError."""

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
    def test_subscribes_to_newly_watched_mints(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.watch("MINT1"))

        asyncio.run(tracker._sync_subscription(ws))

        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(ws.sent[0]["method"], "subscribeTokenTrade")
        self.assertEqual(ws.sent[0]["keys"], ["MINT1"])
        self.assertEqual(tracker._subscribed_mints, {"MINT1"})

    def test_does_not_resend_a_mint_already_subscribed(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.watch("MINT1"))
        asyncio.run(tracker._sync_subscription(ws))

        asyncio.run(tracker._sync_subscription(ws))

        self.assertEqual(len(ws.sent), 1)

    def test_unwatching_a_mint_does_not_resubscribe_it(self):
        # there's no unsubscribe method on PumpPortal's protocol - unwatch()
        # just stops it from being re-sent, it was already subscribed
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        ws = FakeWebSocket()
        asyncio.run(tracker.watch("MINT1"))
        asyncio.run(tracker._sync_subscription(ws))
        asyncio.run(tracker.unwatch("MINT1"))

        asyncio.run(tracker._sync_subscription(ws))

        self.assertEqual(len(ws.sent), 1)  # no new subscribe message sent

    def test_a_fresh_connection_resets_the_subscribed_set(self):
        tracker = CandidatePriceTracker(ws_url="wss://example.invalid")
        tracker._subscribed_mints = {"STALE_FROM_OLD_CONNECTION"}
        asyncio.run(tracker.watch("MINT1"))
        ws = FakeWebSocket()

        class _Ctx:
            async def __aenter__(self_inner):
                return ws

            async def __aexit__(self_inner, *exc):
                return False

        def _fake_connect(*args, **kwargs):
            return _Ctx()

        async def _drive():
            with patch("pumpfun_bot.candidate_price_tracker.websockets.connect", side_effect=_fake_connect):
                try:
                    await asyncio.wait_for(tracker._run_connection(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

        asyncio.run(_drive())

        self.assertEqual(tracker._subscribed_mints, {"MINT1"})
        self.assertEqual(ws.sent[0]["keys"], ["MINT1"])


if __name__ == "__main__":
    unittest.main()
