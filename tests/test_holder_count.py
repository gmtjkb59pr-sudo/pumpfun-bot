import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.holder_count import fetch_holder_count, record_holder_count


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.holder_count.aiohttp.ClientSession", return_value=_FakeSession(response))


class FetchHolderCountTests(unittest.TestCase):
    def test_returns_the_number_of_accounts_found(self):
        response = _FakeResponse({"result": [{"pubkey": "A"}, {"pubkey": "B"}, {"pubkey": "C"}]})
        with _patched(response):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertEqual(count, 3)

    def test_returns_zero_for_a_mint_with_no_holder_accounts(self):
        response = _FakeResponse({"result": []})
        with _patched(response):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertEqual(count, 0)

    def test_returns_none_on_rpc_error(self):
        response = _FakeResponse({"error": {"code": -32000, "message": "boom"}})
        with _patched(response):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.holder_count.aiohttp.ClientSession", return_value=_RaisingSession()):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)


class RecordHolderCountTests(unittest.TestCase):
    def setUp(self):
        self._original_path = activity_log.DATA_LOG_PATH
        f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        f.close()
        self._log_path = Path(f.name)
        activity_log.DATA_LOG_PATH = self._log_path

    def tearDown(self):
        activity_log.DATA_LOG_PATH = self._original_path
        self._log_path.unlink(missing_ok=True)

    def _read_log(self):
        with open(self._log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_logs_a_holder_count_record_on_success(self):
        response = _FakeResponse({"result": [{"pubkey": "A"}, {"pubkey": "B"}]})
        with _patched(response):
            asyncio.run(record_holder_count("MINT", "https://example.invalid/rpc", delay_sec=0))

        records = self._read_log()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "holder_count")
        self.assertEqual(records[0]["mint"], "MINT")
        self.assertEqual(records[0]["count"], 2)

    def test_logs_nothing_when_the_lookup_fails(self):
        response = _FakeResponse({"error": {"code": -32000, "message": "boom"}})
        with _patched(response):
            asyncio.run(record_holder_count("MINT", "https://example.invalid/rpc", delay_sec=0))

        self.assertEqual(self._read_log(), [])


if __name__ == "__main__":
    unittest.main()
