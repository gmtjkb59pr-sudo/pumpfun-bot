import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.okx_client import (
    TRACKER_TYPE_KOL,
    TRACKER_TYPE_MULTI_ADDRESS,
    TRACKER_TYPE_SMART_MONEY,
    _auth_headers,
    _sign,
    fetch_address_tracker_trades,
)


class SignTests(unittest.TestCase):
    """_sign must implement OKX's documented v5 signature scheme exactly -
    base64(HMAC-SHA256(secret, timestamp+method+requestPath+body)). A
    subtly wrong signature fails auth with no useful error beyond
    "unauthorized", so this is worth pinning to a known-correct value."""

    def test_matches_a_known_worked_example(self):
        sig = _sign(
            "test-secret", "2023-10-18T12:21:41.274Z", "GET",
            "/api/v6/dex/market/address-tracker/trades?trackerType=1",
        )
        self.assertEqual(sig, "zrC5WFBkVRxUAAgSA5h8cr+nNulOYNQjn80sejE2tjY=")

    def test_is_deterministic_for_the_same_inputs(self):
        sig1 = _sign("secret", "2023-01-01T00:00:00.000Z", "GET", "/path")
        sig2 = _sign("secret", "2023-01-01T00:00:00.000Z", "GET", "/path")
        self.assertEqual(sig1, sig2)

    def test_changes_when_the_secret_changes(self):
        sig1 = _sign("secret-a", "2023-01-01T00:00:00.000Z", "GET", "/path")
        sig2 = _sign("secret-b", "2023-01-01T00:00:00.000Z", "GET", "/path")
        self.assertNotEqual(sig1, sig2)

    def test_changes_when_the_request_path_changes(self):
        sig1 = _sign("secret", "2023-01-01T00:00:00.000Z", "GET", "/path-a")
        sig2 = _sign("secret", "2023-01-01T00:00:00.000Z", "GET", "/path-b")
        self.assertNotEqual(sig1, sig2)

    def test_lowercases_method_do_not_change_the_signature(self):
        # OKX's docs show uppercase methods - confirm we normalize rather
        # than silently producing a different (wrong) signature for "get"
        sig_upper = _sign("secret", "2023-01-01T00:00:00.000Z", "GET", "/path")
        sig_lower = _sign("secret", "2023-01-01T00:00:00.000Z", "get", "/path")
        self.assertEqual(sig_upper, sig_lower)


class AuthHeadersTests(unittest.TestCase):
    def test_includes_all_four_required_headers(self):
        headers = _auth_headers("my-key", "my-secret", "my-passphrase", "GET", "/some/path")
        self.assertEqual(headers["OK-ACCESS-KEY"], "my-key")
        self.assertEqual(headers["OK-ACCESS-PASSPHRASE"], "my-passphrase")
        self.assertIn("OK-ACCESS-SIGN", headers)
        self.assertIn("OK-ACCESS-TIMESTAMP", headers)

    def test_timestamp_is_iso8601_with_milliseconds_and_z_suffix(self):
        headers = _auth_headers("k", "s", "p", "GET", "/path")
        ts = headers["OK-ACCESS-TIMESTAMP"]
        self.assertTrue(ts.endswith("Z"))
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


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

    def get(self, url, headers=None, timeout=None):
        self.last_call = {"url": url, "headers": headers}
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=_FakeSession(response))


def _success_payload(trades):
    return {"code": "0", "data": {"trades": trades}, "msg": ""}


class FetchAddressTrackerTradesTests(unittest.TestCase):
    def test_returns_the_trade_list_on_success(self):
        trades = [{"walletAddress": "W1", "tokenSymbol": "TEST", "tradeType": "1"}]
        response = _FakeResponse(_success_payload(trades))
        with _patched(response):
            result = asyncio.run(fetch_address_tracker_trades("key", "secret", "pass"))
        self.assertEqual(result, trades)

    def test_defaults_to_smart_money_tracker_type(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_address_tracker_trades("key", "secret", "pass"))
        self.assertIn(f"trackerType={TRACKER_TYPE_SMART_MONEY}", session.last_call["url"])

    def test_multi_address_tracker_includes_the_wallet_addresses(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_address_tracker_trades(
                "key", "secret", "pass",
                tracker_type=TRACKER_TYPE_MULTI_ADDRESS, wallet_address="WALLET1,WALLET2",
            ))
        self.assertIn("walletAddress=WALLET1%2CWALLET2", session.last_call["url"])

    def test_kol_tracker_type_is_accepted(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_address_tracker_trades("key", "secret", "pass", tracker_type=TRACKER_TYPE_KOL))
        self.assertIn(f"trackerType={TRACKER_TYPE_KOL}", session.last_call["url"])

    def test_optional_filters_are_included_when_given(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_address_tracker_trades(
                "key", "secret", "pass", min_market_cap=1000, max_market_cap=20_000_000,
            ))
        self.assertIn("minMarketCap=1000", session.last_call["url"])
        self.assertIn("maxMarketCap=20000000", session.last_call["url"])

    def test_sends_the_auth_headers(self):
        session = _FakeSession(_FakeResponse(_success_payload([])))
        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(fetch_address_tracker_trades("my-key", "secret", "my-pass"))
        self.assertEqual(session.last_call["headers"]["OK-ACCESS-KEY"], "my-key")
        self.assertEqual(session.last_call["headers"]["OK-ACCESS-PASSPHRASE"], "my-pass")

    def test_returns_none_on_a_non_200_status(self):
        response = _FakeResponse(_success_payload([]), status=401)
        with _patched(response):
            result = asyncio.run(fetch_address_tracker_trades("key", "secret", "pass"))
        self.assertIsNone(result)

    def test_returns_none_when_code_is_not_zero(self):
        response = _FakeResponse({"code": "50011", "msg": "Invalid signature"})
        with _patched(response):
            result = asyncio.run(fetch_address_tracker_trades("key", "secret", "pass"))
        self.assertIsNone(result)

    def test_returns_none_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, headers=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.okx_client.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(fetch_address_tracker_trades("key", "secret", "pass"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
