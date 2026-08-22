import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.dexscreener import fetch_price_change_1h_pct


class _FakeResponse:
    def __init__(self, json_data, status=200):
        self._json_data = json_data
        self.status = status

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, timeout=None, headers=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.dexscreener.aiohttp.ClientSession", return_value=_FakeSession(response))


def _pair(price_change_h1, liquidity_usd):
    return {"priceChange": {"h1": price_change_h1}, "liquidity": {"usd": liquidity_usd}}


class FetchPriceChange1hPctTests(unittest.TestCase):
    def test_returns_the_1h_price_change(self):
        response = _FakeResponse([_pair(12.5, 1000)])
        with _patched(response):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertAlmostEqual(pct, 12.5)

    def test_uses_the_highest_liquidity_pair_when_several_exist(self):
        response = _FakeResponse([_pair(-5.0, 100), _pair(30.0, 999999)])
        with _patched(response):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertAlmostEqual(pct, 30.0)

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse([_pair(12.5, 1000)], status=404)
        with _patched(response):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertIsNone(pct)

    def test_returns_none_when_no_pair_is_indexed_yet(self):
        response = _FakeResponse([])
        with _patched(response):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertIsNone(pct)

    def test_returns_none_when_the_field_is_missing(self):
        response = _FakeResponse([{"liquidity": {"usd": 1000}}])
        with _patched(response):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertIsNone(pct)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, timeout=None, headers=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.dexscreener.aiohttp.ClientSession", return_value=_RaisingSession()):
            pct = asyncio.run(fetch_price_change_1h_pct("MINT"))
        self.assertIsNone(pct)


if __name__ == "__main__":
    unittest.main()
