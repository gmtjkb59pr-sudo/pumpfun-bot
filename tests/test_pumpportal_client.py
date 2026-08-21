import asyncio
import unittest

from solders.keypair import Keypair

from pumpfun_bot.pumpportal_client import PumpPortalClient, authenticated_ws_url


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


if __name__ == "__main__":
    unittest.main()
