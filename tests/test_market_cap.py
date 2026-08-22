import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.market_cap import fetch_market_cap_usd


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
    return patch("pumpfun_bot.market_cap.aiohttp.ClientSession", return_value=_FakeSession(response))


class FetchMarketCapUsdTests(unittest.TestCase):
    def test_returns_the_market_cap(self):
        response = _FakeResponse({"usd_market_cap": 12345.67})
        with _patched(response):
            mcap = asyncio.run(fetch_market_cap_usd("MINT"))
        self.assertAlmostEqual(mcap, 12345.67)

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse({"usd_market_cap": 12345.67}, status=404)
        with _patched(response):
            mcap = asyncio.run(fetch_market_cap_usd("MINT"))
        self.assertIsNone(mcap)

    def test_returns_none_when_the_field_is_missing(self):
        response = _FakeResponse({})
        with _patched(response):
            mcap = asyncio.run(fetch_market_cap_usd("MINT"))
        self.assertIsNone(mcap)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, timeout=None, headers=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.market_cap.aiohttp.ClientSession", return_value=_RaisingSession()):
            mcap = asyncio.run(fetch_market_cap_usd("MINT"))
        self.assertIsNone(mcap)


if __name__ == "__main__":
    unittest.main()
