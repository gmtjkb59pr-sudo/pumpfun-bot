import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.dexscreener import fetch_price_changes_pct, fetch_price_usd


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


def _pair(price_change, liquidity_usd, price_usd=None):
    pair = {"priceChange": price_change, "liquidity": {"usd": liquidity_usd}}
    if price_usd is not None:
        pair["priceUsd"] = price_usd
    return pair


class FetchPriceChangesPctTests(unittest.TestCase):
    def test_returns_every_available_window(self):
        response = _FakeResponse([_pair({"m5": 12.5, "h1": 8.0, "h6": -2.0, "h24": 40.0}, 1000)])
        with _patched(response):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertEqual(
            changes, {"m5": 12.5, "h1": 8.0, "h6": -2.0, "h24": 40.0},
        )

    def test_missing_windows_are_none_not_omitted(self):
        response = _FakeResponse([_pair({"m5": 12.5}, 1000)])
        with _patched(response):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertEqual(
            changes, {"m5": 12.5, "h1": None, "h6": None, "h24": None},
        )

    def test_uses_the_highest_liquidity_pair_when_several_exist(self):
        response = _FakeResponse([
            _pair({"m5": -5.0}, 100),
            _pair({"m5": 30.0}, 999999),
        ])
        with _patched(response):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertAlmostEqual(changes["m5"], 30.0)

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse([_pair({"m5": 12.5}, 1000)], status=404)
        with _patched(response):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertIsNone(changes)

    def test_returns_none_when_no_pair_is_indexed_yet(self):
        response = _FakeResponse([])
        with _patched(response):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertIsNone(changes)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, timeout=None, headers=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.dexscreener.aiohttp.ClientSession", return_value=_RaisingSession()):
            changes = asyncio.run(fetch_price_changes_pct("MINT"))
        self.assertIsNone(changes)


class FetchPriceUsdTests(unittest.TestCase):
    def test_returns_the_price_as_a_float(self):
        response = _FakeResponse([_pair({}, 1000, price_usd="0.04384")])
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertAlmostEqual(price, 0.04384)

    def test_uses_the_highest_liquidity_pair_when_several_exist(self):
        response = _FakeResponse([
            _pair({}, 100, price_usd="1.0"),
            _pair({}, 999999, price_usd="2.0"),
        ])
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertAlmostEqual(price, 2.0)

    def test_returns_none_when_no_pair_is_indexed_yet(self):
        response = _FakeResponse([])
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertIsNone(price)

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse([_pair({}, 1000, price_usd="0.04384")], status=404)
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertIsNone(price)

    def test_returns_none_when_price_usd_field_is_missing(self):
        response = _FakeResponse([_pair({}, 1000)])
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertIsNone(price)

    def test_returns_none_on_an_unparseable_price(self):
        response = _FakeResponse([_pair({}, 1000, price_usd="not-a-number")])
        with _patched(response):
            price = asyncio.run(fetch_price_usd("MINT"))
        self.assertIsNone(price)


if __name__ == "__main__":
    unittest.main()
