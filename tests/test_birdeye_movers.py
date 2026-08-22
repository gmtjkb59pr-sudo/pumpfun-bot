import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import BirdeyeMoversConfig, RiskConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.birdeye_movers import BirdeyeMoversStrategy

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH
_TEST_LOG_FILE = None


def setUpModule():
    global _TEST_LOG_FILE
    _TEST_LOG_FILE = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    _TEST_LOG_FILE.close()
    activity_log.DATA_LOG_PATH = Path(_TEST_LOG_FILE.name)


def tearDownModule():
    activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
    if _TEST_LOG_FILE is not None:
        Path(_TEST_LOG_FILE.name).unlink(missing_ok=True)


class FakeClient:
    def __init__(self, *, should_fail_buy=False):
        self.should_fail_buy = should_fail_buy
        self.buy_calls = []
        self.rpc_http_url = "https://example.invalid/rpc"

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
    max_top10_concentration_pct=0, api_key="fake-key",
):
    risk = RiskManager(RiskConfig())
    strategy = BirdeyeMoversStrategy(
        client=client,
        cfg=BirdeyeMoversConfig(
            enabled=True, api_key=api_key, poll_interval_sec=2700, trending_limit=20,
            min_holder_count=min_holder_count, max_top10_concentration_pct=max_top10_concentration_pct,
        ),
        risk=risk,
        alerter=FakeAlerter(),
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=dry_run,
        outcome_tracker=outcome_tracker,
    )
    return strategy, risk


def _token(**overrides):
    base = {
        "address": "MINT", "name": "Test Token", "symbol": "TEST",
        "price": 1.5, "price24hChangePercent": 42.0, "volume24hUSD": 100000.0,
    }
    base.update(overrides)
    return base


_DEFAULT_HOLDER_PATCH = patch(
    "pumpfun_bot.strategies.birdeye_movers.fetch_holder_count",
    lambda mint, rpc_http_url: _async_return(10),
)
_DEFAULT_CONCENTRATION_PATCH = patch(
    "pumpfun_bot.strategies.birdeye_movers.fetch_top10_concentration_pct",
    lambda mint, rpc_http_url: _async_return(5.0),
)


async def _async_return(value):
    return value


class ConsiderTests(unittest.TestCase):
    def test_skips_when_already_tracked(self):
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

        asyncio.run(strategy._consider(_token()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_price_change_is_missing(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_token(price24hChangePercent=None)))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_price_change_is_not_positive(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_token(price24hChangePercent=0.0)))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_risk_manager_blocks(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        risk.cfg.max_sol_per_trade = 0.001  # below the strategy's own 0.03 trade size

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_holder_count_too_low(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=50)

        with patch(
            "pumpfun_bot.strategies.birdeye_movers.fetch_holder_count",
            lambda mint, rpc_http_url: _async_return(5),
        ), _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_concentration_too_high(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=50)

        with _DEFAULT_HOLDER_PATCH, patch(
            "pumpfun_bot.strategies.birdeye_movers.fetch_top10_concentration_pct",
            lambda mint, rpc_http_url: _async_return(90.0),
        ):
            asyncio.run(strategy._consider(_token()))

        self.assertEqual(client.buy_calls, [])

    def test_dry_run_buy_logs_meta_and_tracks(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=True)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)

    def test_live_buy_sends_a_real_trade(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertEqual(client.buy_calls[0], ("buy", "MINT", 0.03))

    def test_live_buy_failure_is_handled_gracefully(self):
        client = FakeClient(should_fail_buy=True)
        strategy, risk = _make_strategy(client, dry_run=False)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))  # must not raise

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)

    def test_ignores_a_token_with_no_address(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_token(address=None)))  # must not raise

        self.assertEqual(client.buy_calls, [])


class RunGatingTests(unittest.TestCase):
    def test_run_returns_immediately_when_disabled(self):
        client = FakeClient()
        risk = RiskManager(RiskConfig())
        strategy = BirdeyeMoversStrategy(
            client=client, cfg=BirdeyeMoversConfig(enabled=False), risk=risk,
            alerter=FakeAlerter(), trade_size_sol=0.03, slippage_pct=10, dry_run=True,
        )
        asyncio.run(asyncio.wait_for(strategy.run(), timeout=1.0))  # must return, not hang

    def test_run_returns_immediately_when_no_api_key_is_set(self):
        client = FakeClient()
        risk = RiskManager(RiskConfig())
        strategy = BirdeyeMoversStrategy(
            client=client, cfg=BirdeyeMoversConfig(enabled=True, api_key=""), risk=risk,
            alerter=FakeAlerter(), trade_size_sol=0.03, slippage_pct=10, dry_run=True,
        )
        asyncio.run(asyncio.wait_for(strategy.run(), timeout=1.0))  # must return, not hang


class PollOnceTests(unittest.TestCase):
    def test_skips_entirely_when_the_trending_fetch_fails(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        async def _failing_fetch(api_key, limit=20):
            return None

        with patch(
            "pumpfun_bot.strategies.birdeye_movers.fetch_trending_tokens", _failing_fetch,
        ):
            asyncio.run(strategy._poll_once())

        self.assertEqual(client.buy_calls, [])

    def test_considers_every_returned_token(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        tokens = [_token(address="MINT1"), _token(address="MINT2")]

        async def _fake_fetch(api_key, limit=20):
            return tokens

        with patch(
            "pumpfun_bot.strategies.birdeye_movers.fetch_trending_tokens", _fake_fetch,
        ), _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._poll_once())

        self.assertEqual(len(client.buy_calls), 2)


if __name__ == "__main__":
    unittest.main()
