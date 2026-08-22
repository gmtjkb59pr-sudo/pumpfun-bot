import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.holder_concentration import fetch_top10_concentration_pct


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _supply_result(amount):
    return {"result": {"value": {"amount": str(amount)}}}


def _largest_result(amounts):
    return {"result": {"value": [{"amount": str(a)} for a in amounts]}}


class _FakeSession:
    def __init__(self, supply_response, largest_response):
        self._supply_response = supply_response
        self._largest_response = largest_response

    def post(self, url, json=None, timeout=None):
        if json["method"] == "getTokenSupply":
            return self._supply_response
        return self._largest_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(supply_response, largest_response):
    session = _FakeSession(supply_response, largest_response)
    return patch("pumpfun_bot.holder_concentration.aiohttp.ClientSession", return_value=session)


class FetchTop10ConcentrationPctTests(unittest.TestCase):
    def test_returns_the_top10_concentration_percentage(self):
        supply = _FakeResponse(_supply_result(1000))
        largest = _FakeResponse(_largest_result([100] * 10 + [5] * 10))
        with _patched(supply, largest):
            pct = asyncio.run(fetch_top10_concentration_pct("MINT", "http://rpc"))
        self.assertAlmostEqual(pct, 100.0)

    def test_only_counts_the_top_10_even_if_more_are_returned(self):
        supply = _FakeResponse(_supply_result(1000))
        largest = _FakeResponse(_largest_result([50] * 10 + [50] * 10))
        with _patched(supply, largest):
            pct = asyncio.run(fetch_top10_concentration_pct("MINT", "http://rpc"))
        self.assertAlmostEqual(pct, 50.0)

    def test_returns_none_when_supply_is_zero(self):
        supply = _FakeResponse(_supply_result(0))
        largest = _FakeResponse(_largest_result([]))
        with _patched(supply, largest):
            pct = asyncio.run(fetch_top10_concentration_pct("MINT", "http://rpc"))
        self.assertIsNone(pct)

    def test_returns_none_on_an_rpc_error(self):
        supply = _FakeResponse({"error": {"message": "boom"}})
        largest = _FakeResponse(_largest_result([100]))
        with _patched(supply, largest):
            pct = asyncio.run(fetch_top10_concentration_pct("MINT", "http://rpc"))
        self.assertIsNone(pct)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.holder_concentration.aiohttp.ClientSession", return_value=_RaisingSession()):
            pct = asyncio.run(fetch_top10_concentration_pct("MINT", "http://rpc"))
        self.assertIsNone(pct)


if __name__ == "__main__":
    unittest.main()
