import asyncio
import time
import unittest
from unittest.mock import patch

from pumpfun_bot.config import RiskConfig, SocialWatchConfig
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.strategies.social_watch import SocialWatchStrategy


class FakeClient:
    def __init__(self, *, trade_events=None, should_fail_buy=False):
        # trade_events: list of events yielded by stream_token_trades, per call
        self._trade_events = trade_events if trade_events is not None else []
        self.should_fail_buy = should_fail_buy
        self.buy_calls = []

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
            "added_ts": time.time() - 5,  # well within the window
        }

        async def _fake_has_socials(uri):
            return True

        with patch(
            "pumpfun_bot.strategies.social_watch.fetch_has_socials", _fake_has_socials,
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
