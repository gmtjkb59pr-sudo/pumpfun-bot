import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.birdeye import fetch_top_traders, fetch_trending_tokens


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
        self.last_call = None

    def get(self, url, headers=None, params=None, timeout=None):
        self.last_call = {"url": url, "headers": headers, "params": params}
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.birdeye.aiohttp.ClientSession", return_value=_FakeSession(response))


def _success_payload(tokens):
    return {"success": True, "data": {"tokens": tokens}}


class FetchTrendingTokensTests(unittest.TestCase):
    def test_returns_the_token_list_on_success(self):
        tokens = [{"address": "MINT1", "symbol": "TEST", "price24hChangePercent": 42.0}]
        response = _FakeResponse(_success_payload(tokens))
        with _patched(response):
            result = asyncio.run(fetch_trending_tokens("fake-key"))
        self.assertEqual(result, tokens)

    def test_sends_the_api_key_header(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.birdeye.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_trending_tokens("my-secret-key"))
        self.assertEqual(session.last_call["headers"]["X-API-KEY"], "my-secret-key")

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse(_success_payload([]), status=401)
        with _patched(response):
            result = asyncio.run(fetch_trending_tokens("fake-key"))
        self.assertIsNone(result)

    def test_returns_none_when_success_is_false(self):
        response = _FakeResponse({"success": False, "message": "invalid key"})
        with _patched(response):
            result = asyncio.run(fetch_trending_tokens("fake-key"))
        self.assertIsNone(result)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, headers=None, params=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.birdeye.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(fetch_trending_tokens("fake-key"))
        self.assertIsNone(result)


def _traders_payload(items):
    return {"success": True, "data": {"items": items}}


class FetchTopTradersTests(unittest.TestCase):
    def test_returns_the_trader_list_on_success(self):
        items = [{"owner": "WALLET1", "totalPnl": 42.0, "tags": ["smart_trader"]}]
        response = _FakeResponse(_traders_payload(items))
        with _patched(response):
            result = asyncio.run(fetch_top_traders("MINT", "fake-key"))
        self.assertEqual(result, items)

    def test_sends_the_mint_address_and_api_key(self):
        session = _FakeSession(_FakeResponse(_traders_payload([])))
        with patch("pumpfun_bot.birdeye.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_top_traders("MINT123", "my-secret-key"))
        self.assertEqual(session.last_call["headers"]["X-API-KEY"], "my-secret-key")
        self.assertEqual(session.last_call["params"]["address"], "MINT123")

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse(_traders_payload([]), status=401)
        with _patched(response):
            result = asyncio.run(fetch_top_traders("MINT", "fake-key"))
        self.assertIsNone(result)

    def test_returns_none_when_success_is_false(self):
        response = _FakeResponse({"success": False, "message": "invalid key"})
        with _patched(response):
            result = asyncio.run(fetch_top_traders("MINT", "fake-key"))
        self.assertIsNone(result)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, headers=None, params=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.birdeye.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(fetch_top_traders("MINT", "fake-key"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
