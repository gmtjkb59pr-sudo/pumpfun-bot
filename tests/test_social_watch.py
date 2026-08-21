import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import RiskConfig, SocialWatchConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.social_watch import SocialWatchStrategy

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH
_TEST_LOG_FILE = None


def setUpModule():
    # bot_state.log_trade() -> activity_log.append_jsonl() always writes to
    # activity_log.DATA_LOG_PATH - the live buy path (dry_run=False) here
    # isn't mocked, so without this every run of this module wrote fake
    # "MINT" trade records into the real, live activity_log.jsonl
    global _TEST_LOG_FILE
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)


class FakeClient:
    def __init__(self, *, trade_events=None, should_fail_buy=False):
        # trade_events: list of events yielded by stream_token_trades, per call
        self._trade_events = trade_events if trade_events is not None else []
        self.should_fail_buy = should_fail_buy
        self.buy_calls = []
        self.rpc_http_url = "https://example.invalid/rpc"

    async def stream_token_trades(self, mints):
        for event in self._trade_events:
            yield event
        # simulates a feed that just never sends anything further - the
        # caller's own asyncio.wait_for timeout is what ends this, not
        # the generator itself finishing on its own in the real client

    async def build_and_send_trade(self, action, mint, amount_sol, slippage_pct):
        self.buy_calls.append((action, mint, amount_sol))
        if self.should_fail_buy:
            raise RuntimeError("simulated buy failure")
        return {"signature": "fake_sig", "action": action, "mint": mint, "amount_sol": amount_sol}


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


def _make_strategy(client, *, dry_run=True, outcome_tracker=None):
    risk = RiskManager(RiskConfig())
    strategy = SocialWatchStrategy(
        client=client,
        cfg=SocialWatchConfig(enabled=True, watch_window_sec=60, poll_interval_sec=10),
        risk=risk,
        alerter=FakeAlerter(),
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=dry_run,
        outcome_tracker=outcome_tracker,
        fresh_ref_timeout_sec=0.05,
    )
    return strategy, risk


class SharedTrackerCollisionTests(unittest.TestCase):
    """If a mint is already held (e.g. by sniper), social_watch must not
    also buy it - OutcomeTracker is shared and keyed by mint alone, so a
    second buy would spend real SOL on a position nothing would ever track
    or exit."""

    def test_skips_buy_when_mint_already_tracked(self):
        import tempfile
        from pathlib import Path

        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        asyncio.run(outcome_tracker.track(
            "MINT", "Already Held", "HELD", entry_ref=100.0, trade_size_sol=0.03,
        ))

        strategy, risk = _make_strategy(client, dry_run=False, outcome_tracker=outcome_tracker)
        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST",
            "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
        }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)


class HolderCountIndexingDelayTests(unittest.TestCase):
    """A candidate can get bought on the very first poll cycle if socials
    were already present at launch - well under the indexing delay needed
    for holder count to be real (confirmed live: a mint read 0 holders at
    buy time, then showed real holders minutes later). _buy() must top up
    to that minimum, measured from the token's own launch (added_ts), not
    skip it just because it already waited watch_window_sec."""

    def test_tops_up_to_the_indexing_delay_when_bought_on_first_poll(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(client, dry_run=True)

        holder_count_calls = []

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            holder_count_calls.append(time.time())
            return 5

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0.05,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            start = time.time()
            # added_ts == now: bought on the very first poll cycle, zero
            # elapsed time since launch, so the full (patched, tiny) delay
            # must still be topped up before the holder-count check runs
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(holder_count_calls), 1)
        self.assertGreaterEqual(holder_count_calls[0] - start, 0.05)

    def test_does_not_sleep_when_already_past_the_indexing_delay(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(client, dry_run=True)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 5

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 20,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            start = time.time()
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time() - 25))  # already 25s old, past the 20s delay
            elapsed = time.time() - start

        self.assertLess(elapsed, 1.0)


class FetchFreshRefTests(unittest.TestCase):
    def test_returns_price_from_first_trade_event(self):
        client = FakeClient(trade_events=[{"marketCapSol": 42.0}])
        strategy, _ = _make_strategy(client)
        ref = asyncio.run(strategy._fetch_fresh_ref("MINT"))
        self.assertEqual(ref, 42.0)

    def test_falls_back_to_none_when_nothing_arrives_in_time(self):
        client = FakeClient(trade_events=[])
        strategy, _ = _make_strategy(client)
        ref = asyncio.run(strategy._fetch_fresh_ref("MINT"))
        self.assertIsNone(ref)


class PollOnceTests(unittest.TestCase):
    def test_expires_candidate_past_watch_window_without_buying(self):
        client = FakeClient()
        strategy, _ = _make_strategy(client)
        strategy._watching["MINT"] = {
            "event": {"mint": "MINT", "uri": "https://example.invalid/meta.json"},
            "added_ts": time.time() - 61,  # past the 60s window
        }
        with patch("pumpfun_bot.strategies.social_watch.fetch_has_socials") as mock_fetch:
            asyncio.run(strategy._poll_once())
            mock_fetch.assert_not_called()

        self.assertNotIn("MINT", strategy._watching)
        self.assertEqual(client.buy_calls, [])

    def test_buys_when_socials_found_within_window(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False)
        strategy._watching["MINT"] = {
            "event": {
                "mint": "MINT", "name": "Test", "symbol": "TEST",
                "uri": "https://example.invalid/meta.json", "vSolInBondingCurve": 30.0,
            },
            # past the holder-count indexing delay too, so _buy() doesn't
            # actually sleep in this test
            "added_ts": time.time() - 25,
        }

        async def _fake_has_socials(uri):
            return True

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 42

        with patch(
            "pumpfun_bot.strategies.social_watch.fetch_has_socials", _fake_has_socials,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            asyncio.run(strategy._poll_once())

        self.assertNotIn("MINT", strategy._watching)
        self.assertEqual(len(client.buy_calls), 1)
        self.assertEqual(client.buy_calls[0], ("buy", "MINT", 0.03))
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_does_not_buy_when_no_socials_found(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        strategy._watching["MINT"] = {
            "event": {"mint": "MINT", "uri": "https://example.invalid/meta.json"},
            "added_ts": time.time() - 5,
        }

        async def _fake_no_socials(uri):
            return False

        with patch(
            "pumpfun_bot.strategies.social_watch.fetch_has_socials", _fake_no_socials,
        ):
            asyncio.run(strategy._poll_once())

        # still watching - hasn't expired yet, and no socials found this round
        self.assertIn("MINT", strategy._watching)
        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)


if __name__ == "__main__":
    unittest.main()
