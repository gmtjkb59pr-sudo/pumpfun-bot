import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pumpfun_bot.bonding_curve import BondingCurve
from pumpfun_bot.config import MarketMakerConfig
from pumpfun_bot.strategies.market_maker import GridState, MarketMakerStrategy


class FakeClient:
    def __init__(self, events):
        self._events = events
        self.build_and_send_trade = AsyncMock(return_value={"signature": "SIG"})

    async def stream_token_trades(self, mints):
        for event in self._events:
            yield event


def _curve_event(mint, virtual_sol, virtual_token, **extra):
    return {
        "mint": mint,
        "vSolInBondingCurve": virtual_sol,
        "vTokensInBondingCurve": virtual_token,
        **extra,
    }


def _make_strategy(events, dry_run=True):
    client = FakeClient(events)
    cfg = MarketMakerConfig(
        enabled=True, target_tokens=["MINT"],
        grid_levels=5, grid_spacing_pct=2.0, order_size_sol=0.01,
    )
    risk = MagicMock()
    risk.can_trade.return_value = (True, "ok")
    alerter = MagicMock()
    alerter.send = AsyncMock()
    strategy = MarketMakerStrategy(
        client=client, cfg=cfg, risk=risk, alerter=alerter,
        slippage_pct=10, dry_run=dry_run,
    )
    return strategy, client, risk


class MarketMakerBondingCurveGridTests(unittest.TestCase):
    def test_event_without_reserves_does_not_init_a_grid(self):
        # old behavior used price or marketCapSol — that is now a loud skip
        events = [{"mint": "MINT", "price": 1e-8, "marketCapSol": 40.0}]
        strategy, client, risk = _make_strategy(events)
        with patch("pumpfun_bot.strategies.market_maker.bot_state"):
            asyncio.run(strategy.run())
        self.assertEqual(strategy.grids, {})
        risk.can_trade.assert_not_called()

    def test_first_valid_event_initializes_grid_without_trading(self):
        events = [_curve_event("MINT", 30.0, 1_073_000_000.0)]
        strategy, client, risk = _make_strategy(events)
        with patch("pumpfun_bot.strategies.market_maker.bot_state"):
            asyncio.run(strategy.run())
        self.assertIn("MINT", strategy.grids)
        risk.can_trade.assert_not_called()
        client.build_and_send_trade.assert_not_called()

    def test_spot_drop_crosses_a_buy_level(self):
        # first event: init at initial curve. second: drop spot ~5% so the
        # -2% buy level is crossed, the -4% level is not.
        init = BondingCurve.initial()
        dropped = init.at_spot(init.spot_price() * 0.97)
        events = [
            _curve_event("MINT", init.virtual_sol, init.virtual_token),
            _curve_event("MINT", dropped.virtual_sol, dropped.virtual_token),
        ]
        strategy, client, risk = _make_strategy(events, dry_run=True)
        with patch("pumpfun_bot.strategies.market_maker.bot_state") as mock_state:
            asyncio.run(strategy.run())
        self.assertIn(-1, strategy.grids["MINT"].filled_levels)
        self.assertNotIn(-2, strategy.grids["MINT"].filled_levels)
        mock_state.log_trade.assert_called()
        risk.register_trade_opened.assert_called()
        client.build_and_send_trade.assert_not_called()  # dry_run

    def test_live_buy_sends_a_real_trade(self):
        init = BondingCurve.initial()
        dropped = init.at_spot(init.spot_price() * 0.97)
        events = [
            _curve_event("MINT", init.virtual_sol, init.virtual_token),
            _curve_event("MINT", dropped.virtual_sol, dropped.virtual_token),
        ]
        strategy, client, risk = _make_strategy(events, dry_run=False)
        with patch("pumpfun_bot.strategies.market_maker.bot_state"):
            asyncio.run(strategy.run())
        client.build_and_send_trade.assert_called()
        kwargs = client.build_and_send_trade.call_args.kwargs
        self.assertEqual(kwargs["action"], "buy")
        self.assertEqual(kwargs["mint"], "MINT")

    def test_grid_levels_are_curve_spots_not_market_cap(self):
        curve = BondingCurve.initial()
        grid = GridState(curve, levels=5, spacing_pct=2.0, order_size_sol=0.01)
        # old code would have treated marketCapSol (~30) as "price"
        self.assertLess(grid.center_spot, 1e-6)
        self.assertGreater(grid.center_spot, 0)
