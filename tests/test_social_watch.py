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


_market_cap_patcher = None
_concentration_patcher = None
_momentum_patcher = None


def setUpModule():
    # bot_state.log_trade() -> activity_log.append_jsonl() always writes to
    # activity_log.DATA_LOG_PATH - the live buy path (dry_run=False) here
    # isn't mocked, so without this every run of this module wrote fake
    # "MINT" trade records into the real, live activity_log.jsonl
    global _TEST_LOG_FILE, _market_cap_patcher, _concentration_patcher, _momentum_patcher
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)

    # fetch_market_cap_usd is called unconditionally inside _buy() (part of
    # the same asyncio.gather as the holder-count/price-ref lookups) -
    # default it module-wide to a high value so the many existing tests
    # here (which don't care about market cap - min_market_cap_usd defaults
    # to 0/disabled) never make a real network call. Tests that DO care
    # about market cap override this locally with their own patch.
    async def _fake_fetch_market_cap_usd(mint):
        return 1_000_000.0

    _market_cap_patcher = patch(
        "pumpfun_bot.strategies.social_watch.fetch_market_cap_usd", _fake_fetch_market_cap_usd,
    )
    _market_cap_patcher.start()

    # same reasoning as above, for fetch_top10_concentration_pct - default
    # to a low (healthy) value so tests that don't care about concentration
    # never make a real RPC call.
    async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
        return 5.0

    _concentration_patcher = patch(
        "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
        _fake_fetch_top10_concentration_pct,
    )
    _concentration_patcher.start()

    # same reasoning as above, for fetch_price_change_1h_pct - default to a
    # positive value so tests that don't care about momentum never make a
    # real network call.
    async def _fake_fetch_price_change_1h_pct(mint):
        return 10.0

    _momentum_patcher = patch(
        "pumpfun_bot.strategies.social_watch.fetch_price_change_1h_pct",
        _fake_fetch_price_change_1h_pct,
    )
    _momentum_patcher.start()


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)
    if _market_cap_patcher is not None:
        _market_cap_patcher.stop()
    if _concentration_patcher is not None:
        _concentration_patcher.stop()
    if _momentum_patcher is not None:
        _momentum_patcher.stop()


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


def _make_strategy(
    client, *, dry_run=True, outcome_tracker=None, min_holder_count=0, min_market_cap_usd=0,
    max_top10_concentration_pct=0, require_positive_momentum_1h=False,
):
    risk = RiskManager(RiskConfig())
    strategy = SocialWatchStrategy(
        client=client,
        cfg=SocialWatchConfig(
            enabled=True, watch_window_sec=60, poll_interval_sec=10,
            min_holder_count=min_holder_count, min_market_cap_usd=min_market_cap_usd,
            max_top10_concentration_pct=max_top10_concentration_pct,
            require_positive_momentum_1h=require_positive_momentum_1h,
        ),
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


class MinHolderCountGateTests(unittest.TestCase):
    """min_holder_count is only ever set by auto_tuner.py once there's real
    evidence (see holder_count_tuning.py) - once it IS set, _buy() must
    actually enforce it."""

    def test_skips_buy_below_min_holder_count(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=16)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 5  # below the 16 minimum

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_buys_at_or_above_min_holder_count(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=16)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 16

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_skips_buy_when_min_holder_count_set_but_lookup_fails(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=16)

        async def _failing_fetch_holder_count(mint, rpc_http_url):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _failing_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_buys_regardless_of_holder_count_when_no_minimum_set(self):
        # default min_holder_count=0 - the current, pre-evidence behavior
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=0)

        async def _failing_fetch_holder_count(mint, rpc_http_url):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _failing_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class MinMarketCapGateTests(unittest.TestCase):
    """User-requested after live sessions showed thin/tiny market caps
    correlated with the worst stop-loss overshoots (a single sell can
    crater an illiquid curve 30-50% in one trade tick) - min_market_cap_usd
    must actually be enforced once set."""

    def test_skips_buy_below_min_market_cap(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_market_cap_usd=7000)

        async def _fake_fetch_market_cap_usd(mint):
            return 3000.0  # below the $7000 minimum

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_market_cap_usd", _fake_fetch_market_cap_usd,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_buys_at_or_above_min_market_cap(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_market_cap_usd=7000)

        async def _fake_fetch_market_cap_usd(mint):
            return 7000.0

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_market_cap_usd", _fake_fetch_market_cap_usd,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_skips_buy_when_min_market_cap_set_but_lookup_fails(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_market_cap_usd=7000)

        async def _failing_fetch_market_cap_usd(mint):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_market_cap_usd", _failing_fetch_market_cap_usd,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_buys_regardless_of_market_cap_when_no_minimum_set(self):
        # default min_market_cap_usd=0 - the filter is off unless explicitly set
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_market_cap_usd=0)

        async def _failing_fetch_market_cap_usd(mint):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_market_cap_usd", _failing_fetch_market_cap_usd,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class MaxTop10ConcentrationGateTests(unittest.TestCase):
    """User-requested after investigating luminos.capital as a bundled-
    launch detector - their top-10-concentration band is the one signal
    cheap enough to compute ourselves via RPC (see holder_concentration.py).
    max_top10_concentration_pct must actually be enforced once set."""

    def test_skips_buy_above_max_concentration(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=50)

        async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
            return 75.0  # above the 50% maximum

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.CONCENTRATION_SETTLING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _fake_fetch_top10_concentration_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_buys_at_or_below_max_concentration(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=50)

        async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
            return 50.0

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.CONCENTRATION_SETTLING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _fake_fetch_top10_concentration_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_skips_buy_when_max_concentration_set_but_lookup_fails(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=50)

        async def _failing_fetch_top10_concentration_pct(mint, rpc_http_url):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.CONCENTRATION_SETTLING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _failing_fetch_top10_concentration_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_buys_regardless_of_concentration_when_no_maximum_set(self):
        # default max_top10_concentration_pct=0 - the filter is off unless explicitly set
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=0)

        async def _failing_fetch_top10_concentration_pct(mint, rpc_http_url):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _failing_fetch_top10_concentration_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class PositiveMomentumGateTests(unittest.TestCase):
    """User-requested "movers"-style filter, built on DexScreener's public
    API after pump.fun's own Movers tab turned out to be ToS-off-limits to
    scrape (see dexscreener.py). require_positive_momentum_1h must actually
    be enforced once set."""

    def test_skips_buy_when_momentum_is_not_positive(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, require_positive_momentum_1h=True)

        async def _fake_fetch_price_change_1h_pct(mint):
            return 0.0  # flat, not positive

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_price_change_1h_pct",
            _fake_fetch_price_change_1h_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_buys_when_momentum_is_positive(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, require_positive_momentum_1h=True)

        async def _fake_fetch_price_change_1h_pct(mint):
            return 0.1

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_price_change_1h_pct",
            _fake_fetch_price_change_1h_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_skips_buy_when_momentum_required_but_lookup_fails(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, require_positive_momentum_1h=True)

        async def _failing_fetch_price_change_1h_pct(mint):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_price_change_1h_pct",
            _failing_fetch_price_change_1h_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_buys_regardless_of_momentum_when_not_required(self):
        # default require_positive_momentum_1h=False - the filter is off unless explicitly set
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, require_positive_momentum_1h=False)

        async def _failing_fetch_price_change_1h_pct(mint):
            return None

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_price_change_1h_pct",
            _failing_fetch_price_change_1h_pct,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class HolderCountIndexingDelayTests(unittest.TestCase):
    """A candidate can get bought on the very first poll cycle if socials
    were already present at launch - well under the indexing delay needed
    for holder count to be real (confirmed live: a mint read 0 holders at
    buy time, then showed real holders minutes later). _buy() must top up
    to that minimum, measured from the token's own launch (added_ts), not
    skip it just because it already waited watch_window_sec.

    Only applies when min_holder_count > 1 - see the "skip this wait
    entirely" test below for the user-requested speed exemption at the
    trivial (<=1) threshold."""

    def test_tops_up_to_the_indexing_delay_when_bought_on_first_poll(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(client, dry_run=True, min_holder_count=16)

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
        strategy, _ = _make_strategy(client, dry_run=True, min_holder_count=16)

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

    def test_skips_the_wait_entirely_when_min_holder_count_is_trivial(self):
        # user-requested speed tradeoff: at min_holder_count <= 1 the count
        # barely gates anything, so don't pay the up-to-20s latency for it
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(client, dry_run=True, min_holder_count=1)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 5

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 20,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ):
            start = time.time()
            # added_ts == now: on the old behavior this would have to wait
            # the full 20s before proceeding
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))
            elapsed = time.time() - start

        self.assertLess(elapsed, 1.0)

    def test_tops_up_to_the_settling_delay_when_concentration_filter_is_active(self):
        # right after launch, virtually all supply sits with the deployer/
        # curve - not an indexing lag, just "no one has traded yet" - so the
        # concentration check needs its own short settling window too
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(client, dry_run=True, max_top10_concentration_pct=70)

        concentration_calls = []

        async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
            concentration_calls.append(time.time())
            return 30.0

        with patch(
            "pumpfun_bot.strategies.social_watch.CONCENTRATION_SETTLING_DELAY_SEC", 0.05,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _fake_fetch_top10_concentration_pct,
        ):
            start = time.time()
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(concentration_calls), 1)
        self.assertGreaterEqual(concentration_calls[0] - start, 0.05)

    def test_waits_for_the_longer_of_the_two_delays_when_both_filters_are_active(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, _ = _make_strategy(
            client, dry_run=True, min_holder_count=16, max_top10_concentration_pct=70,
        )

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 20

        async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
            return 30.0

        with patch(
            "pumpfun_bot.strategies.social_watch.HOLDER_COUNT_INDEXING_DELAY_SEC", 0.2,
        ), patch(
            "pumpfun_bot.strategies.social_watch.CONCENTRATION_SETTLING_DELAY_SEC", 0.05,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_holder_count", _fake_fetch_holder_count,
        ), patch(
            "pumpfun_bot.strategies.social_watch.fetch_top10_concentration_pct",
            _fake_fetch_top10_concentration_pct,
        ):
            start = time.time()
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))
            elapsed = time.time() - start

        # must wait for the longer (holder count, 0.2s) delay, not just the
        # shorter (concentration, 0.05s) one, and not both added together
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 0.3)


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
