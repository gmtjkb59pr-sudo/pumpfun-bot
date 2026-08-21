import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.balance_watch import (
    BalanceFloorReached,
    MaxRealLossReached,
    fetch_sol_balance,
    fetch_sol_usd_price,
    watch_balance_floor,
    watch_max_real_loss,
)
from pumpfun_bot.state import bot_state


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, url, json=None, timeout=None):
        return self._response

    def get(self, url, timeout=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.balance_watch.aiohttp.ClientSession", return_value=_FakeSession(response))


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


class FetchSolBalanceTests(unittest.TestCase):
    def test_returns_balance_in_sol(self):
        response = _FakeResponse({"result": {"value": 250_000_000}})  # 0.25 SOL in lamports
        with _patched(response):
            balance = asyncio.run(fetch_sol_balance("WALLET", "https://example.invalid/rpc"))
        self.assertAlmostEqual(balance, 0.25)

    def test_returns_none_on_rpc_error(self):
        response = _FakeResponse({"error": {"code": -32000, "message": "boom"}})
        with _patched(response):
            balance = asyncio.run(fetch_sol_balance("WALLET", "https://example.invalid/rpc"))
        self.assertIsNone(balance)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.balance_watch.aiohttp.ClientSession", return_value=_RaisingSession()):
            balance = asyncio.run(fetch_sol_balance("WALLET", "https://example.invalid/rpc"))
        self.assertIsNone(balance)


class FetchSolUsdPriceTests(unittest.TestCase):
    def test_returns_the_price(self):
        response = _FakeResponse({"solana": {"usd": 142.5}})
        with _patched(response):
            price = asyncio.run(fetch_sol_usd_price())
        self.assertAlmostEqual(price, 142.5)

    def test_returns_none_on_malformed_response(self):
        response = _FakeResponse({})
        with _patched(response):
            price = asyncio.run(fetch_sol_usd_price())
        self.assertIsNone(price)


class WatchBalanceFloorTests(unittest.TestCase):
    def test_raises_once_confirmed_below_the_floor(self):
        alerter = FakeAlerter()

        async def _fake_balance(wallet_pubkey, rpc_http_url):
            return 0.1  # SOL

        async def _fake_price():
            return 100.0  # $10 total, well under a $40 floor

        async def _drive():
            with patch("pumpfun_bot.balance_watch.fetch_sol_balance", _fake_balance), \
                 patch("pumpfun_bot.balance_watch.fetch_sol_usd_price", _fake_price):
                with self.assertRaises(BalanceFloorReached):
                    await watch_balance_floor(
                        "WALLET", "https://example.invalid/rpc", min_balance_usd=40,
                        alerter=alerter, poll_interval_sec=0,
                    )

        asyncio.run(_drive())
        self.assertTrue(any("$" in m for m in alerter.messages))

    def test_does_not_raise_when_above_the_floor(self):
        async def _fake_balance(wallet_pubkey, rpc_http_url):
            return 1.0  # SOL

        async def _fake_price():
            return 100.0  # $100, well above a $40 floor

        async def _drive():
            with patch("pumpfun_bot.balance_watch.fetch_sol_balance", _fake_balance), \
                 patch("pumpfun_bot.balance_watch.fetch_sol_usd_price", _fake_price):
                try:
                    await asyncio.wait_for(
                        watch_balance_floor(
                            "WALLET", "https://example.invalid/rpc", min_balance_usd=40,
                            poll_interval_sec=0,
                        ),
                        timeout=0.05,
                    )
                except asyncio.TimeoutError:
                    pass  # expected - keeps looping forever while above the floor

        asyncio.run(_drive())  # must not raise BalanceFloorReached

    def test_a_failed_balance_lookup_is_skipped_not_treated_as_below_floor(self):
        async def _fake_balance(wallet_pubkey, rpc_http_url):
            return None  # lookup failed - unknown, not "zero"

        async def _drive():
            with patch("pumpfun_bot.balance_watch.fetch_sol_balance", _fake_balance):
                try:
                    await asyncio.wait_for(
                        watch_balance_floor(
                            "WALLET", "https://example.invalid/rpc", min_balance_usd=40,
                            poll_interval_sec=0,
                        ),
                        timeout=0.05,
                    )
                except asyncio.TimeoutError:
                    pass

        asyncio.run(_drive())  # must not raise BalanceFloorReached

    def test_a_failed_price_lookup_is_skipped_not_treated_as_below_floor(self):
        async def _fake_balance(wallet_pubkey, rpc_http_url):
            return 0.01  # would be below floor at almost any real SOL price

        async def _fake_price():
            return None  # lookup failed - unknown, not "zero"

        async def _drive():
            with patch("pumpfun_bot.balance_watch.fetch_sol_balance", _fake_balance), \
                 patch("pumpfun_bot.balance_watch.fetch_sol_usd_price", _fake_price):
                try:
                    await asyncio.wait_for(
                        watch_balance_floor(
                            "WALLET", "https://example.invalid/rpc", min_balance_usd=40,
                            poll_interval_sec=0,
                        ),
                        timeout=0.05,
                    )
                except asyncio.TimeoutError:
                    pass

        asyncio.run(_drive())  # must not raise BalanceFloorReached


class WatchMaxRealLossTests(unittest.TestCase):
    """Distinct kill-switch from the balance floor: stops the bot once the
    REAL (ground-truth wallet) session loss reaches a dollar amount,
    regardless of the wallet's absolute balance - "stop after losing $X"
    rather than "stop once down to $Y". Reads bot_state.real_pnl_usd, which
    main.py's periodic real-balance tracker keeps up to date elsewhere."""

    def setUp(self):
        # bot_state is a shared singleton - reset the fields this reads/
        # writes so earlier/later tests can't leak into this one
        self._original = (
            bot_state.session_start_balance_sol, bot_state.session_start_balance_usd,
            bot_state.current_balance_sol, bot_state.real_pnl_sol, bot_state.real_pnl_usd,
        )
        bot_state.real_pnl_usd = None

    def tearDown(self):
        (
            bot_state.session_start_balance_sol, bot_state.session_start_balance_usd,
            bot_state.current_balance_sol, bot_state.real_pnl_sol, bot_state.real_pnl_usd,
        ) = self._original

    def test_raises_once_the_real_loss_reaches_the_limit(self):
        bot_state.real_pnl_usd = -10.0
        alerter = FakeAlerter()

        async def _drive():
            with self.assertRaises(MaxRealLossReached):
                await watch_max_real_loss(max_loss_usd=10, alerter=alerter, poll_interval_sec=0)

        asyncio.run(_drive())
        self.assertTrue(any("$" in m for m in alerter.messages))

    def test_does_not_raise_above_the_limit(self):
        bot_state.real_pnl_usd = -3.0

        async def _drive():
            try:
                await asyncio.wait_for(
                    watch_max_real_loss(max_loss_usd=10, poll_interval_sec=0), timeout=0.05,
                )
            except asyncio.TimeoutError:
                pass

        asyncio.run(_drive())  # must not raise MaxRealLossReached

    def test_a_gain_never_triggers(self):
        bot_state.real_pnl_usd = 25.0

        async def _drive():
            try:
                await asyncio.wait_for(
                    watch_max_real_loss(max_loss_usd=10, poll_interval_sec=0), timeout=0.05,
                )
            except asyncio.TimeoutError:
                pass

        asyncio.run(_drive())  # must not raise MaxRealLossReached

    def test_a_missing_real_pnl_is_skipped_not_treated_as_a_loss(self):
        bot_state.real_pnl_usd = None  # no successful balance/price check yet

        async def _drive():
            try:
                await asyncio.wait_for(
                    watch_max_real_loss(max_loss_usd=10, poll_interval_sec=0), timeout=0.05,
                )
            except asyncio.TimeoutError:
                pass

        asyncio.run(_drive())  # must not raise MaxRealLossReached


if __name__ == "__main__":
    unittest.main()
