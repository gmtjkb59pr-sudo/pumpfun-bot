import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import MoonshotHunterConfig, RiskConfig, TakeProfitLevel
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.moonshot_hunter import MoonshotHunterStrategy

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH
_TEST_LOG_FILE = None
_concentration_patcher = None
_momentum_patcher = None


def setUpModule():
    global _TEST_LOG_FILE, _concentration_patcher, _momentum_patcher
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)

    async def _fake_fetch_top10_concentration_pct(mint, rpc_http_url):
        return 5.0

    _concentration_patcher = patch(
        "pumpfun_bot.strategies.moonshot_hunter.fetch_top10_concentration_pct",
        _fake_fetch_top10_concentration_pct,
    )
    _concentration_patcher.start()

    # default momentum well above the default min_price_change_5m_pct
    # (100) so tests that don't care about momentum never fail that gate
    async def _fake_fetch_price_changes_pct(mint):
        return {"m5": 150.0, "h1": 150.0, "h6": 150.0, "h24": 150.0}

    _momentum_patcher = patch(
        "pumpfun_bot.strategies.moonshot_hunter.fetch_price_changes_pct",
        _fake_fetch_price_changes_pct,
    )
    _momentum_patcher.start()


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)
    if _concentration_patcher is not None:
        _concentration_patcher.stop()
    if _momentum_patcher is not None:
        _momentum_patcher.stop()


class FakeClient:
    def __init__(self, *, trade_events=None, should_fail_buy=False, new_token_events=None):
        self._trade_events = trade_events if trade_events is not None else []
        self._new_token_events = new_token_events if new_token_events is not None else []
        self.should_fail_buy = should_fail_buy
        self.buy_calls = []
        self.rpc_http_url = "https://example.invalid/rpc"

    async def stream_new_tokens(self):
        for event in self._new_token_events:
            yield event

    async def stream_token_trades(self, mints):
        for event in self._trade_events:
            yield event

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
    client, *, dry_run=True, outcome_tracker=None, min_holder_count=0,
    max_top10_concentration_pct=0, min_price_change_5m_pct=0, max_open_positions=1,
    min_seconds_between_buys=0, take_profit_ladder=None,
):
    risk = RiskManager(RiskConfig())
    strategy = MoonshotHunterStrategy(
        client=client,
        cfg=MoonshotHunterConfig(
            enabled=True, watch_window_sec=60, poll_interval_sec=3,
            min_holder_count=min_holder_count,
            max_top10_concentration_pct=max_top10_concentration_pct,
            min_price_change_5m_pct=min_price_change_5m_pct,
            max_open_positions=max_open_positions,
            min_seconds_between_buys=min_seconds_between_buys,
            take_profit_ladder=(
                take_profit_ladder if take_profit_ladder is not None
                else [TakeProfitLevel(multiplier=10, sell_pct=10)]
            ),
            stop_loss_pct=70, trailing_activation_pct=300, trailing_stop_pct=35,
            max_hold_sec=2_592_000, stale_price_timeout_sec=21_600,
        ),
        risk=risk,
        alerter=FakeAlerter(),
        trade_size_sol=0.02,
        slippage_pct=10,
        dry_run=dry_run,
        outcome_tracker=outcome_tracker,
        fresh_ref_timeout_sec=0.05,
    )
    return strategy, risk


class MinHolderCountGateTests(unittest.TestCase):
    def test_skips_buy_below_min_holder_count(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=300)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 50

        with patch(
            "pumpfun_bot.strategies.moonshot_hunter.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_holder_count", _fake_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_buys_at_or_above_min_holder_count(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=300)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 300

        with patch(
            "pumpfun_bot.strategies.moonshot_hunter.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_holder_count", _fake_fetch_holder_count,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.02)

    def test_retries_across_polls_instead_of_giving_up_after_one_miss(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=300)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 50

        with patch(
            "pumpfun_bot.strategies.moonshot_hunter.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_holder_count", _fake_fetch_holder_count,
        ):
            done = asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertFalse(done)  # not terminal - retryable


class MinPriceChange5mGateTests(unittest.TestCase):
    """INVERTED from every other strategy: requires strong momentum instead
    of capping it - see MoonshotHunterConfig's docstring."""

    def test_skips_buy_below_the_momentum_floor(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_price_change_5m_pct=100)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 500

        async def _fake_low_momentum(mint):
            return {"m5": 20.0, "h1": 20.0, "h6": 20.0, "h24": 20.0}

        with patch(
            "pumpfun_bot.strategies.moonshot_hunter.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_holder_count", _fake_fetch_holder_count,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_price_changes_pct", _fake_low_momentum,
        ):
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_buys_at_or_above_the_momentum_floor(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_price_change_5m_pct=100)

        async def _fake_fetch_holder_count(mint, rpc_http_url):
            return 500

        with patch(
            "pumpfun_bot.strategies.moonshot_hunter.HOLDER_COUNT_INDEXING_DELAY_SEC", 0,
        ), patch(
            "pumpfun_bot.strategies.moonshot_hunter.fetch_holder_count", _fake_fetch_holder_count,
        ):
            # module default momentum (150.0) is already above the 100 floor
            asyncio.run(strategy._buy("MINT", {
                "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
            }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class CooldownTests(unittest.TestCase):
    def test_skips_a_second_buy_within_the_cooldown(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_seconds_between_buys=3600)
        strategy._last_buy_ts = time.time()  # a bet just landed

        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
        }, time.time()))

        self.assertEqual(client.buy_calls, [])

    def test_allows_a_buy_once_the_cooldown_has_elapsed(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False, min_seconds_between_buys=1)
        strategy._last_buy_ts = time.time() - 2  # cooldown already elapsed

        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
        }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)


class ExitConfigWiringTests(unittest.TestCase):
    """Confirms the whole point of this strategy - wide stop-loss, a huge-
    multiple ladder, and a hold time in days/weeks - actually reaches
    OutcomeTracker, not just the config dataclass."""

    def test_dry_run_buy_passes_the_moonshot_exit_config_through(self):
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        strategy, risk = _make_strategy(
            client, dry_run=True, outcome_tracker=outcome_tracker,
            take_profit_ladder=[TakeProfitLevel(multiplier=10, sell_pct=10)],
        )

        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
        }, time.time()))

        pos = outcome_tracker._pending["MINT"]
        self.assertEqual(pos["stop_loss_pct"], 70)
        self.assertEqual(pos["trailing_activation_pct"], 300)
        self.assertEqual(pos["trailing_stop_pct"], 35)
        self.assertEqual(pos["take_profit_ladder"], [{"multiplier": 10, "sell_pct": 10}])
        self.assertEqual(pos["max_hold_sec"], 2_592_000)
        self.assertEqual(pos["stale_price_timeout_sec"], 21_600)
        self.assertEqual(pos["strategy"], "moonshot_hunter")


class SharedTrackerCollisionTests(unittest.TestCase):
    def test_skips_buy_when_mint_already_tracked(self):
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        asyncio.run(outcome_tracker.track(
            "MINT", "Already Held", "HELD", entry_ref=100.0, trade_size_sol=0.02,
        ))

        strategy, risk = _make_strategy(client, dry_run=False, outcome_tracker=outcome_tracker)
        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST",
            "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
        }, time.time()))

        self.assertEqual(client.buy_calls, [])


class LiveBuyTests(unittest.TestCase):
    def test_live_buy_sends_a_real_trade(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}])
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
        }, time.time()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertEqual(client.buy_calls[0], ("buy", "MINT", 0.02))

    def test_live_buy_failure_is_handled_gracefully(self):
        client = FakeClient(trade_events=[{"marketCapSol": 30.0}], should_fail_buy=True)
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._buy("MINT", {
            "mint": "MINT", "name": "Test", "symbol": "TEST", "vSolInBondingCurve": 30.0,
        }, time.time()))  # must not raise

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)


if __name__ == "__main__":
    unittest.main()
