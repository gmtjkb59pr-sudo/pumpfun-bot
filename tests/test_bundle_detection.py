import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.bundle_detection import fetch_launch_slot_clustering


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _signatures_result(slots):
    return {"result": [{"signature": f"sig{i}", "slot": slot} for i, slot in enumerate(slots)]}


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, url, json=None, timeout=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.bundle_detection.aiohttp.ClientSession", return_value=_FakeSession(response))


class FetchLaunchSlotClusteringTests(unittest.TestCase):
    def test_a_single_transaction_reports_no_clustering(self):
        response = _FakeResponse(_signatures_result([100]))
        with _patched(response):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertEqual(result, {"total_txs": 1, "distinct_slots": 1, "max_txs_in_one_slot": 1})

    def test_several_buys_spread_across_distinct_slots_is_not_flagged_as_clustered(self):
        response = _FakeResponse(_signatures_result([100, 101, 102, 103, 104]))
        with _patched(response):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertEqual(result["max_txs_in_one_slot"], 1)
        self.assertEqual(result["distinct_slots"], 5)

    def test_many_buys_landing_in_the_same_slot_is_the_real_signal(self):
        # confirmed live 2026-08-24: this is what a bundled/sniped launch
        # looks like on-chain - several distinct transactions sharing the
        # exact same slot as each other
        response = _FakeResponse(_signatures_result([100, 100, 100, 100, 100, 101]))
        with _patched(response):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertEqual(result, {"total_txs": 6, "distinct_slots": 2, "max_txs_in_one_slot": 5})

    def test_returns_none_when_there_are_no_transactions_yet(self):
        response = _FakeResponse(_signatures_result([]))
        with _patched(response):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertIsNone(result)

    def test_returns_none_on_an_rpc_error(self):
        response = _FakeResponse({"error": {"message": "boom"}})
        with _patched(response):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertIsNone(result)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.bundle_detection.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(fetch_launch_slot_clustering("MINT", "http://rpc"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
