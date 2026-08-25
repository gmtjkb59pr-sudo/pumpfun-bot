import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pumpfun_bot.config import RiskConfig, SniperConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.sniper import SniperStrategy


class _FakeAlerter:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


def _make_strategy(*, outcome_tracker=None, cfg=None, client=None, alerter=None, dry_run=True):
    risk = RiskManager(RiskConfig())
    return SniperStrategy(
        client=client,
        cfg=cfg if cfg is not None else SniperConfig(enabled=True),
        risk=risk,
        alerter=alerter,
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=dry_run,
        outcome_tracker=outcome_tracker,
    )


class FakeTokenTradeStreamClient:
    """Stands in for PumpPortalClient's stream_token_trades - yields a fixed
    list of trade events, then hangs (like a real WS with no more activity)
    so the bundle-check's own timeout is what ends the wait, matching how a
    real quiet token behaves."""

    def __init__(self, events, rpc_http_url="https://example.invalid/rpc"):
        self._events = events
        self.rpc_http_url = rpc_http_url
        self.requested_mints = []

    def stream_token_trades(self, mints):
        self.requested_mints.append(mints)
        return self._stream()

    async def _stream(self):
        for event in self._events:
            yield event
        await asyncio.sleep(3600)  # never yields again within any test's window


class PassesFiltersTests(unittest.TestCase):
    def test_passes_a_plain_qualifying_event(self):
        strategy = _make_strategy()
        event = {"mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR"}
        self.assertTrue(strategy._passes_filters(event))

    def test_rejects_below_min_liquidity(self):
        strategy = _make_strategy()
        event = {"mint": "MINT", "vSolInBondingCurve": 0.5, "traderPublicKey": "CREATOR"}
        self.assertFalse(strategy._passes_filters(event))

    def test_rejects_a_blocked_wallet(self):
        strategy = _make_strategy()
        strategy.blocked_wallets = {"CREATOR"}
        event = {"mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR"}
        self.assertFalse(strategy._passes_filters(event))

    def test_rejects_a_mint_already_tracked_by_outcome_tracker(self):
        """Regression guard: sniper and social_watch share one
        OutcomeTracker keyed by mint alone - buying a mint that's already
        held (e.g. social_watch bought it first) would spend real SOL on a
        position nothing would ever track or exit."""
        store_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        store_file.close()
        store_path = Path(store_file.name)
        self.addCleanup(lambda: store_path.unlink(missing_ok=True))

        outcome_tracker = OutcomeTracker(ws_url="wss://example.invalid", position_store_path=store_path)
        import asyncio
        asyncio.run(outcome_tracker.track(
            "MINT", "Already Held", "HELD", entry_ref=100.0, trade_size_sol=0.03,
        ))

        strategy = _make_strategy(outcome_tracker=outcome_tracker)
        event = {"mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR"}
        self.assertFalse(strategy._passes_filters(event))


class MaxInitialBuyPctFilterTests(unittest.TestCase):
    """user-requested, ported from an earlier version of this bot's sniper:
    free filter (no RPC call) that rejects a launch where the creator's own
    initialBuy already claims too much of the supply in the creation tx
    itself - a common sign of a planned dump."""

    def test_disabled_by_default_ignores_a_large_initial_buy(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, max_initial_buy_pct=0))
        event = {
            "mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            "initialBuy": 500_000_000,  # 50% of total supply
        }
        self.assertTrue(strategy._passes_filters(event))

    def test_rejects_above_the_threshold(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, max_initial_buy_pct=15))
        event = {
            "mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            "initialBuy": 200_000_000,  # 20% of total supply, above the 15% cap
        }
        self.assertFalse(strategy._passes_filters(event))

    def test_passes_at_or_below_the_threshold(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, max_initial_buy_pct=15))
        event = {
            "mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            "initialBuy": 100_000_000,  # exactly 10%, under the 15% cap
        }
        self.assertTrue(strategy._passes_filters(event))

    def test_missing_initial_buy_does_not_reject(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, max_initial_buy_pct=15))
        event = {"mint": "MINT", "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR"}
        self.assertTrue(strategy._passes_filters(event))


class HolderConcentrationFilterTests(unittest.TestCase):
    """user-requested, ported from an earlier version of this bot's sniper -
    see max_top10_concentration_pct's docstring in config.py for the caveat
    about sniper checking this before holder_concentration.py's own
    settling delay has had a chance to pass."""

    def test_disabled_by_default_never_calls_the_lookup(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, max_top10_concentration_pct=0))
        with patch(
            "pumpfun_bot.strategies.sniper.fetch_top10_concentration_pct",
        ) as mock_fetch:
            result = asyncio.run(strategy._holder_concentration_flags_risk("MINT"))
        self.assertFalse(result)
        mock_fetch.assert_not_called()

    def test_flags_concentration_above_the_threshold(self):
        strategy = _make_strategy(
            client=FakeTokenTradeStreamClient(events=[]),
            cfg=SniperConfig(enabled=True, max_top10_concentration_pct=50),
        )

        async def _fake_fetch(mint, rpc_http_url):
            return 80.0

        with patch("pumpfun_bot.strategies.sniper.fetch_top10_concentration_pct", _fake_fetch):
            result = asyncio.run(strategy._holder_concentration_flags_risk("MINT"))
        self.assertTrue(result)

    def test_does_not_flag_concentration_at_or_below_the_threshold(self):
        strategy = _make_strategy(
            client=FakeTokenTradeStreamClient(events=[]),
            cfg=SniperConfig(enabled=True, max_top10_concentration_pct=50),
        )

        async def _fake_fetch(mint, rpc_http_url):
            return 30.0

        with patch("pumpfun_bot.strategies.sniper.fetch_top10_concentration_pct", _fake_fetch):
            result = asyncio.run(strategy._holder_concentration_flags_risk("MINT"))
        self.assertFalse(result)

    def test_a_failed_lookup_fails_open_not_closed(self):
        # ported behavior: sniper prioritizes speed and never blocks a
        # candidate on a failed/unknown lookup, unlike birdeye_movers'
        # fail-closed equivalent (which has time to spare)
        strategy = _make_strategy(
            client=FakeTokenTradeStreamClient(events=[]),
            cfg=SniperConfig(enabled=True, max_top10_concentration_pct=50),
        )

        async def _fake_fetch(mint, rpc_http_url):
            return None

        with patch("pumpfun_bot.strategies.sniper.fetch_top10_concentration_pct", _fake_fetch):
            result = asyncio.run(strategy._holder_concentration_flags_risk("MINT"))
        self.assertFalse(result)


class BundleCheckFilterTests(unittest.TestCase):
    """user-requested, ported from an earlier version of this bot's sniper:
    watches the token's live trade stream for a short window right after
    launch and rejects it if too many buys land in that time - a sign of
    coordinated insider wallets."""

    def test_disabled_by_default_never_opens_a_stream(self):
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(
            client=client, cfg=SniperConfig(enabled=True, enable_bundle_check=False),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNone(result)
        self.assertEqual(client.requested_mints, [])

    def test_flags_a_mint_with_more_buys_than_the_max_within_the_window(self):
        events = [{"txType": "buy"} for _ in range(6)]  # 6 buys, above max of 5
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(
                enabled=True, enable_bundle_check=True,
                bundle_check_window_ms=50, bundle_check_max_buys=5,
            ),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNotNone(result)
        self.assertIn("gebundeld", result)
        self.assertEqual(client.requested_mints, [["MINT"]])

    def test_does_not_flag_a_mint_at_or_below_the_max(self):
        events = [{"txType": "buy"} for _ in range(3)]  # 3 buys, at/below max of 5
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(
                enabled=True, enable_bundle_check=True,
                bundle_check_window_ms=50, bundle_check_max_buys=5,
            ),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNone(result)

    def test_ignores_non_buy_events(self):
        events = [{"txType": "sell"} for _ in range(10)]
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(
                enabled=True, enable_bundle_check=True,
                bundle_check_window_ms=50, bundle_check_max_buys=5,
            ),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNone(result)


class MinBuysInWindowFilterTests(unittest.TestCase):
    """User-requested, real finding 2026-08-23: 57.1% of everything sniper
    bought went stale_price (zero real trade activity) within seconds, and
    min_liquidity_sol can't catch it (every pump.fun launch starts at
    essentially the same bonding-curve liquidity - not a real signal).
    Rejects a candidate with too FEW real buys in the same window
    bundle_check already watches, instead of a second one."""

    def test_disabled_by_default_never_opens_a_stream(self):
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(
            client=client, cfg=SniperConfig(enabled=True, min_buys_in_window=0),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNone(result)
        self.assertEqual(client.requested_mints, [])

    def test_flags_a_mint_with_fewer_buys_than_the_minimum(self):
        events = [{"txType": "buy"}]  # only 1 buy, below minimum of 2
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(enabled=True, bundle_check_window_ms=50, min_buys_in_window=2),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNotNone(result)
        self.assertIn("geen echte koopactiviteit", result)

    def test_flags_a_mint_with_zero_buys(self):
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(enabled=True, bundle_check_window_ms=50, min_buys_in_window=1),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNotNone(result)

    def test_does_not_flag_a_mint_at_or_above_the_minimum(self):
        events = [{"txType": "buy"} for _ in range(2)]
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(enabled=True, bundle_check_window_ms=50, min_buys_in_window=2),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNone(result)

    def test_both_checks_share_a_single_window_not_two(self):
        # 3 buys: passes min_buys_in_window=2, fails bundle_check_max_buys=2 -
        # both evaluated from the SAME stream watch, only one stream request
        events = [{"txType": "buy"} for _ in range(3)]
        client = FakeTokenTradeStreamClient(events=events)
        strategy = _make_strategy(
            client=client,
            cfg=SniperConfig(
                enabled=True, bundle_check_window_ms=50,
                enable_bundle_check=True, bundle_check_max_buys=2,
                min_buys_in_window=2,
            ),
        )
        result = asyncio.run(strategy._pre_buy_activity_check("MINT"))
        self.assertIsNotNone(result)
        self.assertIn("gebundeld", result)  # too many, not too few
        self.assertEqual(len(client.requested_mints), 1)  # one shared window


class DuplicateNameFilterTests(unittest.TestCase):
    """Real finding 2026-08-23: bought "Rogue Rocket (ROGROC)", and later a
    DIFFERENT mint launched under the exact same name and symbol - a real,
    free (no RPC call) scam signal. PERSISTED with no expiry (not just an
    in-memory window) - confirmed live the SAME day: "Rogue Wizard"
    (ROGWIZ) resurfaced a THIRD time ~55 HOURS after the first two, so a
    scam kit clearly reuses a name across days, not just minutes."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "sniper_seen_launch_names.json"
        self._patcher = patch(
            "pumpfun_bot.strategies.sniper.SEEN_LAUNCH_NAMES_PATH", self._path,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_first_sighting_of_a_name_is_not_a_duplicate(self):
        strategy = _make_strategy()
        self.assertFalse(strategy._is_duplicate_name("Rogue Rocket", "ROGROC"))

    def test_the_same_name_and_symbol_seen_again_is_a_duplicate(self):
        strategy = _make_strategy()
        strategy._is_duplicate_name("Rogue Rocket", "ROGROC")
        self.assertTrue(strategy._is_duplicate_name("Rogue Rocket", "ROGROC"))

    def test_matching_is_case_insensitive_and_ignores_surrounding_whitespace(self):
        strategy = _make_strategy()
        strategy._is_duplicate_name("Rogue Rocket", "ROGROC")
        self.assertTrue(strategy._is_duplicate_name("  rogue rocket ", " rogroc "))

    def test_a_different_symbol_with_the_same_name_is_not_a_duplicate(self):
        strategy = _make_strategy()
        strategy._is_duplicate_name("Rogue Rocket", "ROGROC")
        self.assertFalse(strategy._is_duplicate_name("Rogue Rocket", "ROGROC2"))

    def test_placeholder_names_never_count_as_duplicates_of_each_other(self):
        # event.get("name", "?")/event.get("symbol", "?") in run() means
        # every launch missing this field would otherwise collide on the
        # same ("?", "?") key and falsely flag every one after the first
        strategy = _make_strategy()
        strategy._is_duplicate_name("?", "?")
        self.assertFalse(strategy._is_duplicate_name("?", "?"))

    def test_still_a_duplicate_well_past_the_old_short_window(self):
        # real finding: "Rogue Wizard" resurfaced ~55 hours after its first
        # two sightings - a short in-memory window would have missed it,
        # but the current 72h window comfortably covers it
        from pumpfun_bot.strategies import sniper as sniper_module

        strategy = _make_strategy()
        with patch.object(sniper_module.time, "time", return_value=1000.0):
            strategy._is_duplicate_name("Rogue Wizard", "ROGWIZ")
        with patch.object(sniper_module.time, "time", return_value=1000.0 + 55 * 3600):
            self.assertTrue(strategy._is_duplicate_name("Rogue Wizard", "ROGWIZ"))

    def test_expires_after_the_window(self):
        # user-requested: loosened from no-expiry - a permanently-growing
        # store meant a common/generic name reused by unrelated people
        # days later would also get blocked, not just deliberate copycats
        from pumpfun_bot.strategies import sniper as sniper_module

        strategy = _make_strategy()
        with patch.object(sniper_module.time, "time", return_value=1000.0):
            strategy._is_duplicate_name("Rogue Wizard", "ROGWIZ")
        with patch.object(
            sniper_module.time, "time",
            return_value=1000.0 + sniper_module.SEEN_LAUNCH_NAME_WINDOW_SEC + 1,
        ):
            self.assertFalse(strategy._is_duplicate_name("Rogue Wizard", "ROGWIZ"))

    def test_persists_to_disk_across_separate_strategy_instances(self):
        # simulates surviving a bot restart - a fresh SniperStrategy
        # instance must still recognize a name seen by an earlier one
        strategy_a = _make_strategy()
        strategy_a._is_duplicate_name("Rogue Wizard", "ROGWIZ")

        strategy_b = _make_strategy()
        self.assertTrue(strategy_b._is_duplicate_name("Rogue Wizard", "ROGWIZ"))


class ShadowModelScoreLoggingTests(unittest.TestCase):
    """User-requested: sniper_model.py's win-probability score is computed
    and logged for every real buy, but must NEVER affect the buy decision
    itself (already made and executed by the time this runs) - see
    sniper_model.py's module docstring for why this stays shadow-mode
    until there's enough labeled real-trade history to validate it."""

    def test_does_nothing_when_no_model_is_trained_yet(self):
        strategy = _make_strategy()
        with patch("pumpfun_bot.strategies.sniper.sniper_model.load_model", return_value=None):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                import logging
                logging.getLogger("pumpfun_bot.sniper").info("sentinel")  # ensures ctx has ≥1 record
                strategy._log_shadow_model_score(
                    "MINT", "TEST", {"liquidity_sol": 30.0, "creator": "W", "initial_buy_pct": 5.0},
                )
        self.assertFalse(any("model score" in m for m in ctx.output))

    def test_logs_the_score_when_a_model_is_available(self):
        strategy = _make_strategy()
        fake_model = {"weights": [0.0], "bias": 0.0, "means": [0.0], "stds": [1.0], "features": []}
        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.load_model", return_value=fake_model,
        ), patch(
            "pumpfun_bot.strategies.sniper.sniper_model.build_creator_win_rates", return_value={},
        ), patch(
            "pumpfun_bot.strategies.sniper.sniper_model.score", return_value=0.73,
        ):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                strategy._log_shadow_model_score(
                    "MINT", "TEST", {"liquidity_sol": 30.0, "creator": "W", "initial_buy_pct": 5.0},
                )
        self.assertTrue(any("0.73" in m for m in ctx.output))
        self.assertTrue(any("schaduwmodus" in m for m in ctx.output))

    def test_a_score_of_none_logs_nothing(self):
        strategy = _make_strategy()
        fake_model = {"weights": [0.0], "bias": 0.0, "means": [0.0], "stds": [1.0], "features": []}
        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.load_model", return_value=fake_model,
        ), patch(
            "pumpfun_bot.strategies.sniper.sniper_model.build_creator_win_rates", return_value={},
        ), patch(
            "pumpfun_bot.strategies.sniper.sniper_model.score", return_value=None,
        ):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                import logging
                logging.getLogger("pumpfun_bot.sniper").info("sentinel")
                strategy._log_shadow_model_score("MINT", "TEST", {})
        self.assertFalse(any("model score" in m for m in ctx.output))


class PreBuyModelScoreGateTests(unittest.TestCase):
    """User-requested 2026-08-24: promotes sniper_model.py's score from
    shadow-mode logging to an actual pre-buy gate, once corrected real
    trade data showed it beats baseline and that dead-on-arrival tokens
    are the single biggest real loss category - see
    SniperConfig.model_score_min_to_buy's docstring in config.py."""

    def setUp(self):
        # a run()-level test exercises _is_duplicate_name too, which
        # persists to the REAL SEEN_LAUNCH_NAMES_PATH by default (see
        # DuplicateNameFilterTests above) - isolate it so a name used here
        # doesn't get falsely flagged as a duplicate on a later test run
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "sniper_seen_launch_names.json"
        self._patcher = patch(
            "pumpfun_bot.strategies.sniper.SEEN_LAUNCH_NAMES_PATH", self._path,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_returns_none_when_no_model_is_cached_yet(self):
        # cold start: self._model is None until the first background
        # refresh - must fail OPEN (never gate), same as sniper's other
        # checks, not raise or block on an inconclusive read
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, model_score_min_to_buy=0.5))
        self.assertIsNone(strategy._model)
        result = strategy._pre_buy_model_score(
            {"liquidity_sol": 30.0, "creator": "W", "initial_buy_pct": 5.0},
        )
        self.assertIsNone(result)

    def test_uses_the_cached_model_and_win_rates_not_a_fresh_disk_read(self):
        strategy = _make_strategy(cfg=SniperConfig(enabled=True, model_score_min_to_buy=0.5))
        strategy._model = {"weights": [0.0], "bias": 0.0, "means": [0.0], "stds": [1.0], "features": []}
        strategy._creator_win_rates = {"W": 0.9}
        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.load_model",
        ) as mock_load, patch(
            "pumpfun_bot.strategies.sniper.sniper_model.build_creator_win_rates",
        ) as mock_build:
            result = strategy._pre_buy_model_score(
                {"liquidity_sol": 30.0, "creator": "W", "initial_buy_pct": 5.0},
            )
        mock_load.assert_not_called()
        mock_build.assert_not_called()
        self.assertIsNotNone(result)
        self.assertTrue(0.0 <= result <= 1.0)

    def test_run_rejects_a_candidate_scoring_below_the_threshold(self):
        cfg = SniperConfig(enabled=True, model_score_min_to_buy=0.5)
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(cfg=cfg, client=client)
        strategy._model = {"weights": [0.0], "bias": 0.0, "means": [0.0], "stds": [1.0], "features": []}

        async def _fake_stream_new_tokens():
            yield {
                "mint": "MINT", "name": "Low Score", "symbol": "LOW",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR", "initialBuy": 1_000_000,
            }
        client.stream_new_tokens = _fake_stream_new_tokens

        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.score", return_value=0.1,
        ):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                asyncio.run(strategy.run())
        self.assertTrue(any("model score te laag" in m for m in ctx.output))

    def test_run_buys_a_candidate_scoring_at_or_above_the_threshold(self):
        cfg = SniperConfig(enabled=True, model_score_min_to_buy=0.5)
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(cfg=cfg, client=client, alerter=_FakeAlerter())
        strategy._model = {"weights": [0.0], "bias": 0.0, "means": [0.0], "stds": [1.0], "features": []}

        async def _fake_stream_new_tokens():
            yield {
                "mint": "MINT", "name": "High Score", "symbol": "HIGH",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR", "initialBuy": 1_000_000,
            }
        client.stream_new_tokens = _fake_stream_new_tokens

        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.score", return_value=0.9,
        ):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                asyncio.run(strategy.run())
        self.assertFalse(any("model score te laag" in m for m in ctx.output))
        self.assertTrue(any("Zou kopen" in m for m in ctx.output))

    def test_disabled_by_default_never_scores_or_blocks(self):
        cfg = SniperConfig(enabled=True, model_score_min_to_buy=0)
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(cfg=cfg, client=client, alerter=_FakeAlerter())

        async def _fake_stream_new_tokens():
            yield {
                "mint": "MINT", "name": "Any", "symbol": "ANY",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            }
        client.stream_new_tokens = _fake_stream_new_tokens

        with patch("pumpfun_bot.strategies.sniper.sniper_model.score") as mock_score:
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                asyncio.run(strategy.run())
        mock_score.assert_not_called()
        self.assertTrue(any("Zou kopen" in m for m in ctx.output))

    def test_an_exception_is_caught_and_never_propagates(self):
        strategy = _make_strategy()
        with patch(
            "pumpfun_bot.strategies.sniper.sniper_model.load_model", side_effect=RuntimeError("boom"),
        ):
            strategy._log_shadow_model_score("MINT", "TEST", {})  # must not raise


class WalletBlocklistStartupLoadTests(unittest.TestCase):
    """User-requested 2026-08-24 ("check last data... already make
    improvement") - real bug found live: self.blocked_wallets started
    empty on every restart, only ever populated by
    _refresh_background_state_loop which sleeps WALLET_BLOCKLIST_
    REFRESH_SEC (60s) BEFORE its first check. With 168 real wallets
    blocked as of this session, that's a real 60s window on every restart
    where sniper would blindly buy from an already-known-bad launcher."""

    def test_run_loads_the_full_blocklist_before_processing_the_first_candidate(self):
        cfg = SniperConfig(enabled=True)
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(cfg=cfg, client=client, alerter=_FakeAlerter())

        async def _fake_stream_new_tokens():
            yield {
                "mint": "MINT", "name": "Bad Launcher", "symbol": "BAD",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "KNOWN_BAD_WALLET",
            }
        client.stream_new_tokens = _fake_stream_new_tokens

        with patch(
            "pumpfun_bot.strategies.sniper.blocked_wallets", return_value={"KNOWN_BAD_WALLET"},
        ):
            with self.assertLogs("pumpfun_bot.sniper", level="INFO") as ctx:
                asyncio.run(strategy.run())
        # rejected by _passes_filters on the very first (and only) event -
        # never reached "Zou kopen", proving the blocklist was already
        # populated before the buy loop started, not after a 60s wait
        self.assertFalse(any("Zou kopen" in m for m in ctx.output))
        self.assertEqual(strategy.blocked_wallets, {"KNOWN_BAD_WALLET"})

    def test_a_failed_initial_load_is_caught_not_raised(self):
        cfg = SniperConfig(enabled=True)
        client = FakeTokenTradeStreamClient(events=[])
        strategy = _make_strategy(cfg=cfg, client=client, alerter=_FakeAlerter())

        async def _fake_stream_new_tokens():
            return
            yield  # pragma: no cover - makes this an async generator with zero events
        client.stream_new_tokens = _fake_stream_new_tokens

        with patch(
            "pumpfun_bot.strategies.sniper.blocked_wallets", side_effect=RuntimeError("boom"),
        ):
            asyncio.run(strategy.run())  # must not raise
        self.assertEqual(strategy.blocked_wallets, set())


class _FakeOnChainFailClient:
    """A live buy that fails ON-CHAIN (e.g. Custom 6002 TooMuchSolRequired,
    a real, common buy-side slippage failure) - still pays a real priority
    fee, unlike an RPC/network-level failure before the tx ever landed."""

    def __init__(self, events, error):
        self._events = events
        self._error = error
        self.rpc_http_url = "https://example.invalid/rpc"

    async def stream_new_tokens(self):
        for event in self._events:
            yield event

    def stream_token_trades(self, mints):
        async def _stream():
            await asyncio.sleep(3600)
            yield {}  # pragma: no cover - never reached, keeps this an async generator
        return _stream()

    async def build_and_send_trade(self, action, mint, amount_sol, slippage_pct):
        raise self._error


class FailedBuyFeeAccountingTests(unittest.TestCase):
    """User-requested 2026-08-24 ("the actual profit is not right"): a buy
    that fails ON-CHAIN still pays a real priority fee - realized_pnl_sol
    never accounted for this at all before, looking better than reality by
    the sum of every failed buy's fee."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "sniper_seen_launch_names.json"
        self._patcher = patch(
            "pumpfun_bot.strategies.sniper.SEEN_LAUNCH_NAMES_PATH", self._path,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_an_onchain_buy_failure_registers_a_real_fee_loss(self):
        from pumpfun_bot.fees import PRIORITY_FEE_SOL_PER_LEG
        from pumpfun_bot.pumpportal_client import OnChainTransactionError

        error = OnChainTransactionError("Transactie X is gefaald on-chain: ...", custom_error_code=6002)
        client = _FakeOnChainFailClient(
            events=[{
                "mint": "MINT", "name": "Fails", "symbol": "FAIL",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            }],
            error=error,
        )
        strategy = _make_strategy(
            cfg=SniperConfig(enabled=True), client=client, alerter=_FakeAlerter(), dry_run=False,
        )

        asyncio.run(strategy.run())

        self.assertAlmostEqual(strategy.risk.state.realized_pnl_sol, -PRIORITY_FEE_SOL_PER_LEG, places=6)
        self.assertAlmostEqual(strategy.risk.state.open_exposure_sol, 0.0)  # never registered as opened

    def test_a_non_onchain_failure_does_not_register_a_fee_loss(self):
        # e.g. an RPC timeout before the transaction ever landed on-chain -
        # no real fee was paid, nothing to record
        client = _FakeOnChainFailClient(
            events=[{
                "mint": "MINT", "name": "Fails", "symbol": "FAIL",
                "vSolInBondingCurve": 30.0, "traderPublicKey": "CREATOR",
            }],
            error=RuntimeError("simulated network timeout"),
        )
        strategy = _make_strategy(
            cfg=SniperConfig(enabled=True), client=client, alerter=_FakeAlerter(), dry_run=False,
        )

        asyncio.run(strategy.run())

        self.assertAlmostEqual(strategy.risk.state.realized_pnl_sol, 0.0)


if __name__ == "__main__":
    unittest.main()
