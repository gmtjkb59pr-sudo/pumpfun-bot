import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.holder_count import (
    SPL_TOKEN_PROGRAM_ID,
    TOKEN_2022_PROGRAM_ID,
    fetch_holder_count,
    record_holder_count,
)


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _account_info_response(owner):
    return _FakeResponse({"result": {"value": {"owner": owner}}})


def _account_info_missing_response():
    return _FakeResponse({"result": {"value": None}})


def _program_accounts_response(pubkeys):
    return _FakeResponse({"result": [{"pubkey": p} for p in pubkeys]})


def _largest_accounts_response(accounts):
    return _FakeResponse({"result": {"value": accounts}})


def _error_response(message="boom"):
    return _FakeResponse({"error": {"code": -32000, "message": message}})


class _RoutingSession:
    """Routes each POST to the response matching its JSON-RPC method -
    fetch_holder_count now makes up to two sequential calls (which program
    owns the mint, then the matching holder-count query for that program)."""

    def __init__(self, *, account_info=None, program_accounts=None, largest_accounts=None):
        self._responses = {
            "getAccountInfo": account_info,
            "getProgramAccounts": program_accounts,
            "getTokenLargestAccounts": largest_accounts,
        }
        self.calls = []

    def post(self, url, json=None, timeout=None):
        method = json["method"]
        self.calls.append(method)
        response = self._responses.get(method)
        if response is None:
            raise AssertionError(f"unexpected call to {method}")
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(session):
    return patch("pumpfun_bot.holder_count.aiohttp.ClientSession", return_value=session)


class FetchHolderCountTests(unittest.TestCase):
    def test_returns_the_number_of_accounts_for_a_token2022_mint(self):
        session = _RoutingSession(
            account_info=_account_info_response(TOKEN_2022_PROGRAM_ID),
            program_accounts=_program_accounts_response(["A", "B", "C"]),
        )
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertEqual(count, 3)

    def test_returns_zero_for_a_token2022_mint_with_no_holder_accounts(self):
        session = _RoutingSession(
            account_info=_account_info_response(TOKEN_2022_PROGRAM_ID),
            program_accounts=_program_accounts_response([]),
        )
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertEqual(count, 0)

    def test_falls_back_to_largest_accounts_for_a_classic_spl_mint(self):
        # confirmed live: getProgramAccounts filtered by mint on the classic
        # SPL Token program is rejected outright by Helius (spans every
        # classic-SPL account on Solana, not just this mint's) - this is
        # the working fallback for exactly that case
        session = _RoutingSession(
            account_info=_account_info_response(SPL_TOKEN_PROGRAM_ID),
            largest_accounts=_largest_accounts_response(
                [{"address": f"ACC{i}", "amount": "1000"} for i in range(20)]
            ),
        )
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertEqual(count, 20)
        self.assertEqual(session.calls, ["getAccountInfo", "getTokenLargestAccounts"])

    def test_returns_none_when_the_mint_owner_is_unknown(self):
        session = _RoutingSession(account_info=_account_info_missing_response())
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)

    def test_returns_none_when_the_owner_program_is_neither_known_program(self):
        session = _RoutingSession(account_info=_account_info_response("SomeOtherProgram1111111111111111111111111"))
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)

    def test_returns_none_when_the_account_info_lookup_errors(self):
        session = _RoutingSession(account_info=_error_response())
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)

    def test_returns_none_on_a_token2022_program_accounts_error(self):
        session = _RoutingSession(
            account_info=_account_info_response(TOKEN_2022_PROGRAM_ID),
            program_accounts=_error_response(),
        )
        with _patched(session):
            count = asyncio.run(fetch_holder_count("MINT", "https://example.invalid/rpc"))
        self.assertIsNone(count)

    def test_returns_none_on_a_classic_spl_largest_accounts_error(self):
        session = _RoutingSession(
            account_info=_account_info_response(SPL_TOKEN_PROGRAM_ID),
            largest_accounts=_error_response(),
        )
        with _patched(session):
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
        session = _RoutingSession(
            account_info=_account_info_response(TOKEN_2022_PROGRAM_ID),
            program_accounts=_program_accounts_response(["A", "B"]),
        )
        with _patched(session):
            asyncio.run(record_holder_count("MINT", "https://example.invalid/rpc", delay_sec=0))

        records = self._read_log()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "holder_count")
        self.assertEqual(records[0]["mint"], "MINT")
        self.assertEqual(records[0]["count"], 2)

    def test_logs_nothing_when_the_lookup_fails(self):
        session = _RoutingSession(account_info=_error_response())
        with _patched(session):
            asyncio.run(record_holder_count("MINT", "https://example.invalid/rpc", delay_sec=0))

        self.assertEqual(self._read_log(), [])


if __name__ == "__main__":
    unittest.main()
