import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.wallet_reconciliation import fetch_wallet_token_mints, find_untracked_wallet_holdings


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
    return patch(
        "pumpfun_bot.wallet_reconciliation.aiohttp.ClientSession", return_value=_FakeSession(response),
    )


def _account(mint: str, ui_amount: float) -> dict:
    return {
        "account": {"data": {"parsed": {"info": {
            "mint": mint, "tokenAmount": {"uiAmount": ui_amount},
        }}}},
    }


class _FakeSessionByProgram:
    """Routes each post() call to a different canned response based on the
    programId in the request payload - used to prove both token programs
    (see TOKEN_PROGRAM_IDS) are actually queried and unioned, not just one."""

    def __init__(self, response_by_program_id):
        self._response_by_program_id = response_by_program_id

    def post(self, url, json=None, timeout=None):
        program_id = json["params"][1]["programId"]
        return self._response_by_program_id[program_id]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FetchWalletTokenMintsTests(unittest.TestCase):
    def test_returns_mints_with_nonzero_balance(self):
        response = _FakeResponse({"result": {"value": [
            _account("MINT_A", 100.0), _account("MINT_B", 0), _account("MINT_C", 5.5),
        ]}})
        with _patched(response):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertEqual(mints, {"MINT_A", "MINT_C"})

    def test_returns_none_on_rpc_error(self):
        response = _FakeResponse({"error": {"code": -32000, "message": "boom"}})
        with _patched(response):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertIsNone(mints)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.wallet_reconciliation.aiohttp.ClientSession", return_value=_RaisingSession()):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertIsNone(mints)

    def test_empty_wallet_returns_empty_set(self):
        response = _FakeResponse({"result": {"value": []}})
        with _patched(response):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertEqual(mints, set())

    def test_unions_holdings_from_both_token_programs(self):
        # real bug found live: a mint held under the classic SPL Token
        # program (confirmed directly on-chain) never showed up when only
        # Token-2022 was queried - a pump.fun mint can use either program,
        # not consistently one or the other, so both must be queried
        from pumpfun_bot.wallet_reconciliation import SPL_TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID

        session = _FakeSessionByProgram({
            SPL_TOKEN_PROGRAM_ID: _FakeResponse({"result": {"value": [_account("CLASSIC_MINT", 10.0)]}}),
            TOKEN_2022_PROGRAM_ID: _FakeResponse({"result": {"value": [_account("TOKEN2022_MINT", 5.0)]}}),
        })
        with patch("pumpfun_bot.wallet_reconciliation.aiohttp.ClientSession", return_value=session):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertEqual(mints, {"CLASSIC_MINT", "TOKEN2022_MINT"})

    def test_none_if_either_program_query_fails(self):
        # a combined result missing one program's accounts is incomplete by
        # definition - must report "unknown", never a confidently-wrong
        # partial set that silently drops real holdings
        from pumpfun_bot.wallet_reconciliation import SPL_TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID

        session = _FakeSessionByProgram({
            SPL_TOKEN_PROGRAM_ID: _FakeResponse({"result": {"value": [_account("CLASSIC_MINT", 10.0)]}}),
            TOKEN_2022_PROGRAM_ID: _FakeResponse({"error": {"code": -32000, "message": "boom"}}),
        })
        with patch("pumpfun_bot.wallet_reconciliation.aiohttp.ClientSession", return_value=session):
            mints = asyncio.run(fetch_wallet_token_mints("WALLET", "https://example.invalid/rpc"))
        self.assertIsNone(mints)


class FindUntrackedWalletHoldingsTests(unittest.TestCase):
    def test_finds_wallet_holdings_the_bot_is_not_tracking(self):
        # the actual bug: wallet holds a leftover from an earlier session
        # that this session's OutcomeTracker never learned about
        response = _FakeResponse({"result": {"value": [
            _account("TRACKED_MINT", 10.0), _account("LEFTOVER_MINT", 5.0),
        ]}})
        with _patched(response):
            untracked = asyncio.run(find_untracked_wallet_holdings(
                "WALLET", "https://example.invalid/rpc", tracked_mints={"TRACKED_MINT"},
            ))
        self.assertEqual(untracked, {"LEFTOVER_MINT"})

    def test_empty_set_when_everything_held_is_tracked(self):
        response = _FakeResponse({"result": {"value": [_account("TRACKED_MINT", 10.0)]}})
        with _patched(response):
            untracked = asyncio.run(find_untracked_wallet_holdings(
                "WALLET", "https://example.invalid/rpc", tracked_mints={"TRACKED_MINT"},
            ))
        self.assertEqual(untracked, set())

    def test_returns_none_when_the_lookup_fails_rather_than_a_false_all_clear(self):
        response = _FakeResponse({"error": {"code": -32000, "message": "boom"}})
        with _patched(response):
            untracked = asyncio.run(find_untracked_wallet_holdings(
                "WALLET", "https://example.invalid/rpc", tracked_mints={"TRACKED_MINT"},
            ))
        self.assertIsNone(untracked)


if __name__ == "__main__":
    unittest.main()
