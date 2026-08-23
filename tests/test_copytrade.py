import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pumpfun_bot.config import CopyTradeConfig
from pumpfun_bot.strategies.copytrade import CopyTradeStrategy


def _event(action, mint, sol=1.0, ts=None):
    return {
        "txType": action,
        "mint": mint,
        "solAmount": sol,
        "timestamp": ts if ts is not None else time.time() * 1000,
        "vSolInBondingCurve": 50.0,
    }


class FakeClient:
    def __init__(self, events):
        self._events = events
        self.build_and_send_trade = AsyncMock(return_value={"signature": "SIG"})

    async def stream_wallet_trades(self, wallets):
        for event in self._events:
            yield event


def _make_strategy(events, dry_run=True, mirror_pct=100):
    client = FakeClient(events)
    cfg = CopyTradeConfig(enabled=True, watched_wallets=["W1"], mirror_pct=mirror_pct, max_copy_delay_ms=3000)
    risk = MagicMock()
    risk.can_trade.return_value = (True, "")
    risk.register_trade_opened = MagicMock()
    alerter = MagicMock()
    alerter.send = AsyncMock()
    strategy = CopyTradeStrategy(
        client=client, cfg=cfg, risk=risk, alerter=alerter,
        max_trade_sol=0.015, slippage_pct=5, dry_run=dry_run,
    )
    return strategy, client, risk


class CopyTradeSellGatingTests(unittest.TestCase):
    def test_sell_signal_for_never_bought_mint_is_ignored(self):
        strategy, client, risk = _make_strategy([_event("sell", "M1")])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        mock_state.log_trade.assert_not_called()
        risk.can_trade.assert_not_called()

    def test_sell_signal_after_a_mirrored_buy_is_mirrored(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("sell", "M1"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        actions = [call.args[1] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(actions, ["buy", "sell"])

    def test_sell_after_the_position_is_already_sold_is_ignored_again(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("sell", "M1"), _event("sell", "M1"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        actions = [call.args[1] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(actions, ["buy", "sell"])

    def test_buy_then_sell_of_different_mints_only_mirrors_the_held_one(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("sell", "M2"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        mints = [call.args[2] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(mints, ["M1"])

    def test_live_mode_only_marks_held_after_a_successful_buy(self):
        strategy, client, risk = _make_strategy(
            [_event("buy", "M1"), _event("sell", "M1")], dry_run=False,
        )
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        self.assertEqual(client.build_and_send_trade.await_count, 2)
        actions = [call.args[1] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(actions, ["buy", "sell"])

    def test_live_mode_failed_buy_does_not_mark_mint_as_held(self):
        client = FakeClient([_event("buy", "M1"), _event("sell", "M1")])
        client.build_and_send_trade = AsyncMock(side_effect=Exception("boom"))
        cfg = CopyTradeConfig(enabled=True, watched_wallets=["W1"], mirror_pct=100, max_copy_delay_ms=3000)
        risk = MagicMock()
        risk.can_trade.return_value = (True, "")
        alerter = MagicMock()
        alerter.send = AsyncMock()
        strategy = CopyTradeStrategy(
            client=client, cfg=cfg, risk=risk, alerter=alerter,
            max_trade_sol=0.015, slippage_pct=5, dry_run=False,
        )
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        # the buy failed, so the sell signal must still be ignored (never
        # actually held) rather than mirrored on the strength of a failed buy
        mock_state.log_trade.assert_not_called()


class CopyTradeBuyDedupTests(unittest.TestCase):
    def test_a_second_buy_signal_for_an_already_held_mint_is_ignored(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("buy", "M1"), _event("buy", "M1"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        self.assertEqual(mock_state.log_trade.call_count, 1)

    def test_a_buy_signal_for_a_different_mint_is_still_mirrored(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("buy", "M2"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        mints = [call.args[2] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(mints, ["M1", "M2"])

    def test_a_buy_after_a_full_round_trip_is_mirrored_again(self):
        strategy, client, risk = _make_strategy([
            _event("buy", "M1"), _event("sell", "M1"), _event("buy", "M1"),
        ])
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        actions = [call.args[1] for call in mock_state.log_trade.call_args_list]
        self.assertEqual(actions, ["buy", "sell", "buy"])

    def test_live_mode_a_second_buy_signal_for_an_already_held_mint_is_ignored(self):
        strategy, client, risk = _make_strategy(
            [_event("buy", "M1"), _event("buy", "M1")], dry_run=False,
        )
        with patch("pumpfun_bot.strategies.copytrade.bot_state") as mock_state:
            asyncio.run(strategy.run())
        self.assertEqual(client.build_and_send_trade.await_count, 1)
        self.assertEqual(mock_state.log_trade.call_count, 1)


if __name__ == "__main__":
    unittest.main()
