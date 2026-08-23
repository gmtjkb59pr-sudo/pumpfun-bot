import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import BirdeyeMoversConfig, RiskConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.birdeye_movers import BirdeyeMoversStrategy, is_pump_fun_mint

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
    max_top10_concentration_pct=0, max_market_cap_usd=20_000_000, api_key="fake-key",
):
    risk = RiskManager(RiskConfig())
    strategy = BirdeyeMoversStrategy(
        client=client,
        cfg=BirdeyeMoversConfig(
            enabled=True, api_key=api_key, poll_interval_sec=2700, trending_limit=20,
            min_holder_count=min_holder_count, max_top10_concentration_pct=max_top10_concentration_pct,
            max_market_cap_usd=max_market_cap_usd,
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
        "address": "MINTpump", "name": "Test Token", "symbol": "TEST",
        "price": 1.5, "price24hChangePercent": 42.0, "volume24hUSD": 100000.0,
        "marketcap": 50000.0,  # realistic small-memecoin cap, well under the default ceiling
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


class IsPumpFunMintTests(unittest.TestCase):
    def test_accepts_a_real_pump_fun_style_mint(self):
        self.assertTrue(is_pump_fun_mint("BUXBsh4FxjSGbR2H5MJFg3c3Xf5hYRV4hG3u51iUpump"))

    def test_rejects_a_mint_confirmed_live_to_have_no_pumpportal_route(self):
        # Truth Coin and OpenAI PreStocks - both real Birdeye trending
        # candidates tonight, both failed to buy with a 400 from
        # PumpPortal under every pool value tested
        self.assertFalse(is_pump_fun_mint("8wEZ5cavCg2zvGzo91FnjaEbYv4bWXyfTFQkSW6QwBHP"))
        self.assertFalse(is_pump_fun_mint("HiKvhwS1eV4yP5h4p1ZxKGjkvukhxMii23C4W64BApZ"))

    def test_rejects_a_mint_that_merely_contains_pump_not_ending_in_it(self):
        self.assertFalse(is_pump_fun_mint("pumpXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"))


class ConsiderTests(unittest.TestCase):
    def test_skips_a_candidate_that_is_not_a_real_pump_fun_mint(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_token(address="8wEZ5cavCg2zvGzo91FnjaEbYv4bWXyfTFQkSW6QwBHP")))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_already_tracked(self):
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        asyncio.run(outcome_tracker.track(
            "MINTpump", "Already Held", "HELD", entry_ref=100.0, trade_size_sol=0.03,
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
        # max_sol_total_exposure, not max_sol_per_trade - the strategy now
        # passes its own trade_size_sol as a max_sol_per_trade_override (see
        # TradeSizeAboveSharedRiskCapTests below), so a low shared
        # max_sol_per_trade alone no longer blocks anything here. Exposure
        # stays a real, unoverridden shared wallet-wide budget.
        risk.cfg.max_sol_total_exposure = 0.001

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
        self.assertEqual(client.buy_calls[0], ("buy", "MINTpump", 0.03))

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


class TradeSizeAboveSharedRiskCapTests(unittest.TestCase):
    """trade_size_sol can be configured ABOVE the shared risk.max_sol_per_
    trade (see main.py's `cfg.birdeye_movers.trade_size_sol or
    cfg.risk.max_sol_per_trade` wiring) - the risk manager must check
    against this strategy's OWN trade size, not silently reject every
    trade above some unrelated shared default."""

    def test_buy_above_the_shared_cap_is_not_blocked_by_the_risk_manager(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        strategy.trade_size_sol = 0.053  # deliberately above the shared cap
        risk.cfg.max_sol_per_trade = 0.015

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.053)


class MaxMarketCapGateTests(unittest.TestCase):
    """Confirmed live: Birdeye's trending list (sorted by volumeUSD)
    surfaces major/blue-chip tokens (SOL, wrapped ETH, PUMP itself), not
    just memecoins - "buying" those via a pump.fun-style trade is a
    category error. Unlike the other 0-disables-it filters, this one
    defaults ON (see BirdeyeMoversConfig.max_market_cap_usd)."""

    def test_skips_a_token_above_the_market_cap_ceiling(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=20_000_000)

        asyncio.run(strategy._consider(_token(marketcap=90_000_000_000)))  # SOL-scale

        self.assertEqual(client.buy_calls, [])

    def test_buys_a_token_at_or_below_the_ceiling(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=20_000_000)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token(marketcap=20_000_000)))

        self.assertEqual(len(client.buy_calls), 1)

    def test_skips_when_market_cap_is_unknown(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=20_000_000)

        asyncio.run(strategy._consider(_token(marketcap=None)))

        self.assertEqual(client.buy_calls, [])

    def test_default_ceiling_is_disabled_now_that_pool_auto_handles_buyability(self):
        # SUPERSEDED: confirmed live that a REAL pump.fun-suffixed mint at
        # ~$260M market cap builds a valid transaction fine via pool="auto"
        # (only pool="pump" fails for it) - the original ~$69k graduation
        # ceiling was working around a routing limitation that no longer
        # exists (see pumpportal_client.py's pool="auto" default). Market
        # cap no longer predicts buyability for a real pump.fun token, so
        # the default is 0 (disabled) - is_pump_fun_mint() is what actually
        # excludes blue chips (SOL/PUMP/wrapped ETH never end in "pump").
        self.assertEqual(BirdeyeMoversConfig().max_market_cap_usd, 0)

    def test_a_migrated_looking_candidate_is_bought_by_default_now(self):
        client = FakeClient()
        strategy, risk = _make_strategy(
            client, dry_run=False, max_market_cap_usd=BirdeyeMoversConfig().max_market_cap_usd,
        )

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token(marketcap=18_000_000)))  # real observed candidate scale

        self.assertEqual(len(client.buy_calls), 1)

    def test_buys_regardless_of_market_cap_when_ceiling_is_disabled(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=0)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_token(marketcap=90_000_000_000)))

        self.assertEqual(len(client.buy_calls), 1)


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
        tokens = [_token(address="MINT1pump"), _token(address="MINT2pump")]

        async def _fake_fetch(api_key, limit=20):
            return tokens

        with patch(
            "pumpfun_bot.strategies.birdeye_movers.fetch_trending_tokens", _fake_fetch,
        ), _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._poll_once())

        self.assertEqual(len(client.buy_calls), 2)


if __name__ == "__main__":
    unittest.main()
