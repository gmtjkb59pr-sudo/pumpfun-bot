import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.genuine_interest import fetch_genuine_interest_stats


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _signatures_result(sigs):
    return {"result": [{"signature": s} for s in sigs]}


def _tx_result(*, wallet, action, unrelated=False):
    """action: "buy", "sell", or None (neither instruction found)."""
    if unrelated:
        logs = ["Program log: Instruction: SomethingElse"]
    elif action == "buy":
        logs = ["Program log: Instruction: Buy"]
    elif action == "sell":
        logs = ["Program log: Instruction: Sell"]
    else:
        logs = []
    return {
        "result": {
            "meta": {"logMessages": logs},
            "transaction": {"message": {"accountKeys": [{"pubkey": wallet, "signer": True}]}},
        },
    }


class _RoutingSession:
    """Routes each POST by JSON-RPC method - getTransaction additionally
    routed by the signature in params[0], since different transactions in
    the same test need different classifications."""

    def __init__(self, *, signatures_response=None, tx_responses=None):
        self._signatures_response = signatures_response
        self._tx_responses = tx_responses or {}

    def post(self, url, json=None, timeout=None):
        method = json["method"]
        if method == "getSignaturesForAddress":
            return _FakeResponse(self._signatures_response)
        if method == "getTransaction":
            sig = json["params"][0]
            return _FakeResponse(self._tx_responses[sig])
        raise AssertionError(f"unexpected method: {method}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(session):
    return patch("pumpfun_bot.genuine_interest.aiohttp.ClientSession", return_value=session)


class FetchGenuineInterestStatsTests(unittest.TestCase):
    def test_returns_none_when_signature_lookup_fails(self):
        session = _RoutingSession(signatures_response={"error": {"message": "boom"}})
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertIsNone(result)

    def test_zero_transactions_returns_none_ratios_not_zero(self):
        session = _RoutingSession(signatures_response=_signatures_result([]))
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertEqual(result["total_classified"], 0)
        self.assertIsNone(result["unique_buyer_ratio"])
        self.assertIsNone(result["wash_ratio"])

    def test_every_buy_from_a_different_wallet_is_a_ratio_of_1(self):
        sigs = ["SIG1", "SIG2", "SIG3"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": _tx_result(wallet="WALLET_B", action="buy"),
                "SIG3": _tx_result(wallet="WALLET_C", action="buy"),
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertEqual(result["total_classified"], 3)
        self.assertAlmostEqual(result["unique_buyer_ratio"], 1.0)

    def test_the_same_wallet_buying_repeatedly_lowers_the_ratio(self):
        sigs = ["SIG1", "SIG2", "SIG3", "SIG4"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG3": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG4": _tx_result(wallet="WALLET_B", action="buy"),
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        # 2 unique wallets / 4 buy transactions
        self.assertAlmostEqual(result["unique_buyer_ratio"], 0.5)

    def test_a_wallet_that_buys_and_sells_within_the_window_counts_as_wash(self):
        sigs = ["SIG1", "SIG2", "SIG3"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": _tx_result(wallet="WALLET_A", action="sell"),
                "SIG3": _tx_result(wallet="WALLET_B", action="buy"),
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        # wallets seen: {A, B} - only A round-tripped -> 1/2 = 0.5
        self.assertAlmostEqual(result["wash_ratio"], 0.5)

    def test_no_round_tripping_wallets_gives_a_zero_wash_ratio(self):
        sigs = ["SIG1", "SIG2"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": _tx_result(wallet="WALLET_B", action="buy"),
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertAlmostEqual(result["wash_ratio"], 0.0)

    def test_unrelated_transactions_are_excluded_from_classification(self):
        sigs = ["SIG1", "SIG2"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": _tx_result(wallet="WALLET_B", action=None, unrelated=True),
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertEqual(result["total_classified"], 1)
        self.assertAlmostEqual(result["unique_buyer_ratio"], 1.0)

    def test_a_failed_transaction_fetch_is_excluded_not_fatal(self):
        sigs = ["SIG1", "SIG2"]
        session = _RoutingSession(
            signatures_response=_signatures_result(sigs),
            tx_responses={
                "SIG1": _tx_result(wallet="WALLET_A", action="buy"),
                "SIG2": {"error": {"message": "boom"}},
            },
        )
        with _patched(session):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertEqual(result["total_classified"], 1)

    def test_a_connection_failure_on_signatures_returns_none(self):
        class _RaisingSession:
            def post(self, *a, **kw):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.genuine_interest.aiohttp.ClientSession", return_value=_RaisingSession()):
            result = asyncio.run(fetch_genuine_interest_stats("MINT", "http://rpc"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
