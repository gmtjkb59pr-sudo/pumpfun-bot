import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pumpfun_bot.config import RiskConfig, SniperConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.sniper import SniperStrategy


def _make_strategy(*, outcome_tracker=None, cfg=None, client=None):
    risk = RiskManager(RiskConfig())
    return SniperStrategy(
        client=client,
        cfg=cfg if cfg is not None else SniperConfig(enabled=True),
        risk=risk,
        alerter=None,
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=True,
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
    """Real finding 2026-08-23: bought "Rogue Rocket (ROGROC)", and ~35s
    later a DIFFERENT mint launched under the exact same name and symbol -
    a real, free (no RPC call) scam signal: a legitimate project doesn't
    relaunch under its own name minutes later, but a copycat/rug-kit
    reusing a recognizable name to catch bots/humans does."""

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

    def test_expires_after_the_window(self):
        from pumpfun_bot.strategies import sniper as sniper_module

        strategy = _make_strategy()
        with patch.object(sniper_module.time, "time", return_value=1000.0):
            strategy._is_duplicate_name("Rogue Rocket", "ROGROC")
        with patch.object(
            sniper_module.time, "time",
            return_value=1000.0 + sniper_module.DUPLICATE_NAME_WINDOW_SEC + 1,
        ):
            self.assertFalse(strategy._is_duplicate_name("Rogue Rocket", "ROGROC"))


if __name__ == "__main__":
    unittest.main()
