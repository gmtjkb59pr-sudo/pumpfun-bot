import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.config import CoinGeckoMoversConfig, RiskConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.coingecko_movers import CoinGeckoMoversStrategy

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
    max_top10_concentration_pct=0, max_market_cap_usd=100_000, api_key="fake-key",
    momentum_window="m5", take_profit_ladder=(),
):
    # dry_run must match the strategy's own dry_run below - main.py always
    # constructs both from the same cfg.risk.dry_run - otherwise can_trade's
    # dry-run capital-limit bypass (see risk.py) reads a stale/mismatched
    # flag from what these tests are actually exercising.
    risk = RiskManager(RiskConfig(dry_run=dry_run))
    strategy = CoinGeckoMoversStrategy(
        client=client,
        cfg=CoinGeckoMoversConfig(
            enabled=True, api_key=api_key, poll_interval_sec=300, trending_limit=20,
            momentum_window=momentum_window,
            min_holder_count=min_holder_count, max_top10_concentration_pct=max_top10_concentration_pct,
            max_market_cap_usd=max_market_cap_usd, take_profit_ladder=list(take_profit_ladder),
        ),
        risk=risk,
        alerter=FakeAlerter(),
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=dry_run,
        outcome_tracker=outcome_tracker,
    )
    return strategy, risk


def _candidate(**overrides):
    base = {
        "mint": "MINTpump", "pair_name": "TEST / SOL", "price_usd": 0.001,
        "price_change_pct": {"m5": 5.0, "m15": 6.0, "m30": 7.0, "h1": 8.0, "h6": 9.0, "h24": 10.0},
        "market_cap_usd": 50000.0,  # realistic small-memecoin cap, well under the default ceiling
        "volume_24h_usd": 100000.0,
    }
    base.update(overrides)
    return base


_DEFAULT_HOLDER_PATCH = patch(
    "pumpfun_bot.strategies.coingecko_movers.fetch_holder_count",
    lambda mint, rpc_http_url: _async_return(10),
)
_DEFAULT_CONCENTRATION_PATCH = patch(
    "pumpfun_bot.strategies.coingecko_movers.fetch_top10_concentration_pct",
    lambda mint, rpc_http_url: _async_return(5.0),
)


async def _async_return(value):
    return value


class ConsiderTests(unittest.TestCase):
    def test_skips_a_candidate_that_is_not_a_real_pump_fun_mint(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_candidate(mint="8wEZ5cavCg2zvGzo91FnjaEbYv4bWXyfTFQkSW6QwBHP")))

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

        asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_momentum_window_value_is_missing(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_candidate(price_change_pct={"m5": None})))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_momentum_is_not_positive(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_candidate(price_change_pct={"m5": 0.0})))

        self.assertEqual(client.buy_calls, [])

    def test_skips_a_noise_level_move_below_min_price_change_pct(self):
        # user-requested 2026-08-24 ("narrow the buy trigger"): the
        # original trigger was "> 0" - real data showed sub-1% moves
        # (0.04%, 0.08% confirmed live) carry no real signal, indistinguishable
        # from a token sitting flat
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        asyncio.run(strategy._consider(_candidate(price_change_pct={"m5": 2.0})))

        self.assertEqual(client.buy_calls, [])

    def test_buys_a_candidate_at_or_above_min_price_change_pct(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate(price_change_pct={"m5": 5.0})))

        self.assertEqual(len(client.buy_calls), 1)

    def test_uses_the_configured_momentum_window_not_always_m5(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, momentum_window="h1")

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate(price_change_pct={"m5": -5.0, "h1": 5.0})))

        self.assertEqual(len(client.buy_calls), 1)

    def test_skips_when_risk_manager_blocks(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        risk.cfg.max_sol_total_exposure = 0.001

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_holder_count_too_low(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, min_holder_count=50)

        with patch(
            "pumpfun_bot.strategies.coingecko_movers.fetch_holder_count",
            lambda mint, rpc_http_url: _async_return(5),
        ), _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(client.buy_calls, [])

    def test_skips_when_concentration_too_high(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_top10_concentration_pct=50)

        with _DEFAULT_HOLDER_PATCH, patch(
            "pumpfun_bot.strategies.coingecko_movers.fetch_top10_concentration_pct",
            lambda mint, rpc_http_url: _async_return(90.0),
        ):
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(client.buy_calls, [])

    def test_dry_run_buy_registers_exposure_and_tracks_with_real_entry_ref(self):
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        strategy, risk = _make_strategy(client, dry_run=True, outcome_tracker=outcome_tracker)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate(price_usd=0.00042)))

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.03)
        self.assertIn("MINTpump", outcome_tracker._pending)
        self.assertEqual(outcome_tracker._pending["MINTpump"]["entry_ref"], 0.00042)
        # confirmed live: a real WS tick's bonding-curve-scale ref applied
        # to this USD-priced entry_ref produced a nonsensical exit - the
        # tracker must know this position's entry is USD-denominated so it
        # can ignore WS ticks and rely on the REST fallback instead
        self.assertEqual(outcome_tracker._pending["MINTpump"]["price_source"], "usd")

    def test_dry_run_buy_passes_take_profit_ladder_through_to_the_tracker(self):
        from pumpfun_bot.config import TakeProfitLevel

        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        strategy, risk = _make_strategy(
            client, dry_run=True, outcome_tracker=outcome_tracker,
            take_profit_ladder=[TakeProfitLevel(multiplier=2, sell_pct=30)],
        )

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(
            outcome_tracker._pending["MINTpump"]["take_profit_ladder"],
            [{"multiplier": 2, "sell_pct": 30}],
        )

    def test_dry_run_buy_passes_the_movers_specific_hold_time_settings(self):
        # user-requested 2026-08-24 ("what is the best exit strategy for
        # this kind of strategy") - was silently falling back to
        # OutcomeTracker's shared sniper-scale defaults (15-min max hold,
        # 10s stale-price timeout), which assume a brand-new bonding curve
        # trading multiple times per second - premature panic-selling on
        # an already-liquid, slower-moving "mover"
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        client = FakeClient()
        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        strategy, risk = _make_strategy(client, dry_run=True, outcome_tracker=outcome_tracker)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(outcome_tracker._pending["MINTpump"]["stale_price_timeout_sec"], 60)
        self.assertEqual(outcome_tracker._pending["MINTpump"]["max_hold_sec"], 3600)

    def test_live_buy_sends_a_real_trade(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertEqual(client.buy_calls[0], ("buy", "MINTpump", 0.03))

    def test_live_buy_failure_is_handled_gracefully(self):
        client = FakeClient(should_fail_buy=True)
        strategy, risk = _make_strategy(client, dry_run=False)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))  # must not raise

        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.0)


class TradeSizeAboveSharedRiskCapTests(unittest.TestCase):
    def test_buy_above_the_shared_cap_is_not_blocked_by_the_risk_manager(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        strategy.trade_size_sol = 0.053  # deliberately above the shared cap
        risk.cfg.max_sol_per_trade = 0.015

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate()))

        self.assertEqual(len(client.buy_calls), 1)
        self.assertAlmostEqual(risk.state.open_exposure_sol, 0.053)


class MaxMarketCapGateTests(unittest.TestCase):
    def test_skips_a_candidate_above_the_market_cap_ceiling(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=100_000)

        asyncio.run(strategy._consider(_candidate(market_cap_usd=18_000_000)))

        self.assertEqual(client.buy_calls, [])

    def test_buys_a_candidate_at_or_below_the_ceiling(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=100_000)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate(market_cap_usd=100_000)))

        self.assertEqual(len(client.buy_calls), 1)

    def test_skips_when_market_cap_is_unknown(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=100_000)

        asyncio.run(strategy._consider(_candidate(market_cap_usd=None)))

        self.assertEqual(client.buy_calls, [])

    def test_default_ceiling_is_disabled_now_that_pool_auto_handles_buyability(self):
        # see the equivalent test in test_birdeye_movers.py for the full
        # reasoning - market cap no longer predicts buyability for a real
        # pump.fun token now that pool="auto" is the default
        self.assertEqual(CoinGeckoMoversConfig().max_market_cap_usd, 0)

    def test_buys_regardless_of_market_cap_when_ceiling_is_disabled(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False, max_market_cap_usd=0)

        with _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._consider(_candidate(market_cap_usd=90_000_000_000)))

        self.assertEqual(len(client.buy_calls), 1)


class RunGatingTests(unittest.TestCase):
    def test_run_returns_immediately_when_disabled(self):
        client = FakeClient()
        risk = RiskManager(RiskConfig())
        strategy = CoinGeckoMoversStrategy(
            client=client, cfg=CoinGeckoMoversConfig(enabled=False), risk=risk,
            alerter=FakeAlerter(), trade_size_sol=0.03, slippage_pct=10, dry_run=True,
        )
        asyncio.run(asyncio.wait_for(strategy.run(), timeout=1.0))  # must return, not hang

    def test_run_returns_immediately_when_no_api_key_is_set(self):
        client = FakeClient()
        risk = RiskManager(RiskConfig())
        strategy = CoinGeckoMoversStrategy(
            client=client, cfg=CoinGeckoMoversConfig(enabled=True, api_key=""), risk=risk,
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
            "pumpfun_bot.strategies.coingecko_movers.fetch_trending_pools", _failing_fetch,
        ):
            asyncio.run(strategy._poll_once())

        self.assertEqual(client.buy_calls, [])

    def test_considers_every_parseable_pool(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        pools = [{"pool": "one"}, {"pool": "two"}]

        async def _fake_fetch(api_key, limit=20):
            return pools

        with patch(
            "pumpfun_bot.strategies.coingecko_movers.fetch_trending_pools", _fake_fetch,
        ), patch(
            "pumpfun_bot.strategies.coingecko_movers.parse_pool_candidate",
            side_effect=[_candidate(mint="MINT1pump"), _candidate(mint="MINT2pump")],
        ), _DEFAULT_HOLDER_PATCH, _DEFAULT_CONCENTRATION_PATCH:
            asyncio.run(strategy._poll_once())

        self.assertEqual(len(client.buy_calls), 2)

    def test_skips_a_pool_that_fails_to_parse(self):
        client = FakeClient()
        strategy, risk = _make_strategy(client, dry_run=False)
        pools = [{"pool": "unparseable"}]

        async def _fake_fetch(api_key, limit=20):
            return pools

        with patch(
            "pumpfun_bot.strategies.coingecko_movers.fetch_trending_pools", _fake_fetch,
        ), patch(
            "pumpfun_bot.strategies.coingecko_movers.parse_pool_candidate", return_value=None,
        ):
            asyncio.run(strategy._poll_once())  # must not raise

        self.assertEqual(client.buy_calls, [])


if __name__ == "__main__":
    unittest.main()
