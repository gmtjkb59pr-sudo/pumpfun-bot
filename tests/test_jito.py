import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.jito import get_bundle_status, poll_bundle_until_landed, send_bundle


class _FakeResponse:
    def __init__(self, json_data=None, status=200, text=""):
        self._json_data = json_data
        self.status = status
        self._text = text

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Scripts a sequence of responses - each call to post() returns the
    next one, repeating the last once exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.post_calls = []

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[idx]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(*responses):
    return patch("pumpfun_bot.jito.aiohttp.ClientSession", return_value=_FakeSession(list(responses)))


async def _fast_sleep(_seconds):
    return None


class SendBundleTests(unittest.TestCase):
    def test_returns_the_bundle_id_on_success(self):
        response = _FakeResponse(json_data={"jsonrpc": "2.0", "result": "BUNDLE_ID_ABC", "id": 1})
        with _patched(response):
            result = asyncio.run(send_bundle([b"signed_tx_bytes"]))
        self.assertEqual(result, "BUNDLE_ID_ABC")

    def test_sends_base64_encoded_transactions_with_the_correct_method(self):
        response = _FakeResponse(json_data={"result": "BUNDLE_ID"})
        with _patched(response) as mock_session_cls:
            asyncio.run(send_bundle([b"raw_tx_1", b"raw_tx_2"]))
        session = mock_session_cls.return_value
        _args, kwargs = session.post_calls[0]
        payload = kwargs["json"]
        self.assertEqual(payload["method"], "sendBundle")
        self.assertEqual(payload["params"][1], {"encoding": "base64"})
        self.assertEqual(len(payload["params"][0]), 2)
        # base64 strings, not raw bytes - real transactions decode cleanly
        import base64
        self.assertEqual(base64.b64decode(payload["params"][0][0]), b"raw_tx_1")

    def test_returns_none_on_non_200_status(self):
        response = _FakeResponse(status=500, text="internal error")
        with _patched(response):
            result = asyncio.run(send_bundle([b"tx"]))
        self.assertIsNone(result)

    def test_returns_none_on_a_jsonrpc_error(self):
        response = _FakeResponse(json_data={"jsonrpc": "2.0", "error": {"message": "boom"}, "id": 1})
        with _patched(response):
            result = asyncio.run(send_bundle([b"tx"]))
        self.assertIsNone(result)

    def test_returns_none_on_a_connection_failure(self):
        class _RaisingSession:
            def post(self, *a, **kw):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.jito.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(send_bundle([b"tx"]))
        self.assertIsNone(result)

    def test_uses_the_given_block_engine_url(self):
        response = _FakeResponse(json_data={"result": "BUNDLE_ID"})
        with _patched(response) as mock_session_cls:
            asyncio.run(send_bundle([b"tx"], block_engine_url="https://frankfurt.mainnet.block-engine.jito.wtf"))
        session = mock_session_cls.return_value
        args, _kwargs = session.post_calls[0]
        self.assertEqual(args[0], "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/bundles")


class GetBundleStatusTests(unittest.TestCase):
    def test_returns_the_status_entry_when_present(self):
        response = _FakeResponse(json_data={
            "jsonrpc": "2.0",
            "result": {"context": {"slot": 1}, "value": [
                {"bundle_id": "BID", "confirmation_status": "confirmed", "err": {"Ok": None}},
            ]},
        })
        with _patched(response):
            result = asyncio.run(get_bundle_status("BID"))
        self.assertEqual(result["confirmation_status"], "confirmed")

    def test_returns_none_when_the_bundle_is_not_found_yet(self):
        response = _FakeResponse(json_data={
            "jsonrpc": "2.0", "result": {"context": {"slot": 1}, "value": []},
        })
        with _patched(response):
            result = asyncio.run(get_bundle_status("BID"))
        self.assertIsNone(result)

    def test_returns_none_on_non_200_status(self):
        response = _FakeResponse(status=500)
        with _patched(response):
            result = asyncio.run(get_bundle_status("BID"))
        self.assertIsNone(result)

    def test_returns_none_on_a_connection_failure(self):
        class _RaisingSession:
            def post(self, *a, **kw):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.jito.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(get_bundle_status("BID"))
        self.assertIsNone(result)


class PollBundleUntilLandedTests(unittest.TestCase):
    def test_returns_immediately_once_confirmed(self):
        response = _FakeResponse(json_data={
            "result": {"value": [{"bundle_id": "BID", "confirmation_status": "confirmed", "err": None}]},
        })
        with _patched(response), patch("pumpfun_bot.jito.asyncio.sleep", _fast_sleep):
            result = asyncio.run(poll_bundle_until_landed("BID", poll_interval_sec=0.01, timeout_sec=1))
        self.assertEqual(result["confirmation_status"], "confirmed")

    def test_keeps_polling_while_the_bundle_has_not_landed_yet(self):
        not_found = _FakeResponse(json_data={"result": {"value": []}})
        confirmed = _FakeResponse(json_data={
            "result": {"value": [{"bundle_id": "BID", "confirmation_status": "finalized", "err": None}]},
        })
        with _patched(not_found, not_found, confirmed), patch("pumpfun_bot.jito.asyncio.sleep", _fast_sleep):
            result = asyncio.run(poll_bundle_until_landed("BID", poll_interval_sec=0.01, timeout_sec=1))
        self.assertEqual(result["confirmation_status"], "finalized")

    def test_returns_none_after_timeout_with_no_result(self):
        never_found = _FakeResponse(json_data={"result": {"value": []}})
        with _patched(never_found), patch("pumpfun_bot.jito.asyncio.sleep", _fast_sleep):
            result = asyncio.run(poll_bundle_until_landed("BID", poll_interval_sec=0.01, timeout_sec=0.03))
        self.assertIsNone(result)

    def test_a_processed_but_not_yet_confirmed_status_keeps_polling(self):
        processed = _FakeResponse(json_data={
            "result": {"value": [{"bundle_id": "BID", "confirmation_status": "processed", "err": None}]},
        })
        confirmed = _FakeResponse(json_data={
            "result": {"value": [{"bundle_id": "BID", "confirmation_status": "confirmed", "err": None}]},
        })
        with _patched(processed, confirmed), patch("pumpfun_bot.jito.asyncio.sleep", _fast_sleep):
            result = asyncio.run(poll_bundle_until_landed("BID", poll_interval_sec=0.01, timeout_sec=1))
        self.assertEqual(result["confirmation_status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
