import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.config import RiskConfig, SniperConfig
from pumpfun_bot.outcome_tracker import OutcomeTracker
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.sniper import SniperStrategy


def _make_strategy(*, outcome_tracker=None):
    risk = RiskManager(RiskConfig())
    return SniperStrategy(
        client=None,
        cfg=SniperConfig(enabled=True),
        risk=risk,
        alerter=None,
        trade_size_sol=0.03,
        slippage_pct=10,
        dry_run=True,
        outcome_tracker=outcome_tracker,
    )


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


if __name__ == "__main__":
    unittest.main()
