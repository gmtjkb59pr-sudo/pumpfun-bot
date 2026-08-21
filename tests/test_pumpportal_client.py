import asyncio
import unittest
from unittest.mock import patch

from solders.keypair import Keypair

from pumpfun_bot.pumpportal_client import PumpPortalClient, authenticated_ws_url


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
    """Scripts a sequence of getSignatureStatuses responses - each call to
    post() returns the next one, repeating the last once exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def post(self, url, json=None, timeout=None):
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return _FakeResponse(self._responses[idx])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _status_response(*, err=None, confirmation_status="confirmed"):
    return {"result": {"value": [{"err": err, "confirmationStatus": confirmation_status}]}}


def _not_found_response():
    return {"result": {"value": [None]}}


class AuthenticatedWsUrlTests(unittest.TestCase):
    def test_no_api_key_returns_url_unchanged(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data", ""),
            "wss://pumpportal.fun/api/data",
        )

    def test_appends_api_key_as_query_param(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data", "abc123"),
            "wss://pumpportal.fun/api/data?api-key=abc123",
        )

    def test_appends_with_ampersand_if_url_already_has_query_params(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data?foo=bar", "abc123"),
            "wss://pumpportal.fun/api/data?foo=bar&api-key=abc123",
        )


class TradeRequestBodyTests(unittest.TestCase):
    """Verifies the exact request bodies sent to PumpPortal's trade-local
    endpoint - this is safety-critical for real-money trades, particularly
    that a full exit really does say "sell everything" (amount: "100%",
    denominatedInSol: "false") rather than a partial SOL-denominated sell.
    Confirmed against https://pumpportal.fun/local-trading-api/trading-api."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_full_sell_uses_100_percent_denominated_in_tokens(self):
        client = self._make_client()
        captured = {}

        async def fake_sign_and_send(body):
            captured.update(body)
            return "fake_signature"

        client._sign_and_send = fake_sign_and_send
        result = asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))

        self.assertEqual(captured["action"], "sell")
        self.assertEqual(captured["amount"], "100%")
        self.assertEqual(captured["denominatedInSol"], "false")
        self.assertEqual(captured["mint"], "MINT123")
        self.assertEqual(result["signature"], "fake_signature")

    def test_buy_is_still_sol_denominated(self):
        # regression guard: build_and_send_trade (used for buys, and for
        # copytrade's partial sells) must keep its existing SOL-denominated
        # behavior - only the new full-sell path uses percentage/token amounts
        client = self._make_client()
        captured = {}

        async def fake_sign_and_send(body):
            captured.update(body)
            return "fake_signature"

        client._sign_and_send = fake_sign_and_send
        asyncio.run(client.build_and_send_trade(
            action="buy", mint="MINT123", amount_sol=0.05, slippage_pct=10
        ))

        self.assertEqual(captured["action"], "buy")
        self.assertEqual(captured["amount"], 0.05)
        self.assertEqual(captured["denominatedInSol"], "true")


class ConfirmTransactionTests(unittest.TestCase):
    """Regression coverage for a real bug: with skipPreflight on,
    sendTransaction returning a signature only means the tx was ACCEPTED for
    processing, not that it executed. A transaction that reverts on-chain
    (e.g. slippage exceeded) must not be silently treated as a successful
    sell - that's exactly what happened live before this check existed."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_succeeds_when_confirmed_with_no_error(self):
        client = self._make_client()
        session = _FakeSession([_status_response(err=None, confirmation_status="confirmed")])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(client._confirm_transaction("SIG", timeout_sec=1, poll_interval_sec=0.01))
        # no exception raised = success

    def test_raises_when_transaction_reverted_on_chain(self):
        client = self._make_client()
        session = _FakeSession([
            _status_response(err={"InstructionError": [3, {"Custom": 6005}]}, confirmation_status="processed")
        ])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                asyncio.run(client._confirm_transaction("SIG", timeout_sec=1, poll_interval_sec=0.01))

    def test_raises_if_never_confirmed_within_timeout(self):
        client = self._make_client()
        session = _FakeSession([_not_found_response()])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                asyncio.run(client._confirm_transaction("SIG", timeout_sec=0.05, poll_interval_sec=0.01))

    def test_keeps_polling_through_processed_until_confirmed(self):
        client = self._make_client()
        session = _FakeSession([
            _status_response(err=None, confirmation_status="processed"),
            _status_response(err=None, confirmation_status="processed"),
            _status_response(err=None, confirmation_status="confirmed"),
        ])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(client._confirm_transaction("SIG", timeout_sec=1, poll_interval_sec=0.01))
        self.assertEqual(session.call_count, 3)


if __name__ == "__main__":
    unittest.main()
