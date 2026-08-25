import asyncio
import unittest
from unittest.mock import patch

import base58
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

from pumpfun_bot.pumpportal_client import (
    OnChainTransactionError,
    PumpPortalClient,
    authenticated_ws_url,
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


class _FakeSession:
    """Scripts a sequence of getSignatureStatuses responses - each call to
    post() returns the next one, repeating the last once exhausted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.post_calls = []  # (args, kwargs) for every post() call, in order

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        idx = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return _FakeResponse(self._responses[idx])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


async def _fast_sleep(_seconds):
    """Drop-in for asyncio.sleep in retry-loop tests - avoids real delays."""
    return None


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

    def test_amount_pct_overrides_the_default_100_percent(self):
        # user-requested: 99% fallback for a real deterministic on-chain
        # Overflow (Custom 6024) confirmed on 3 stuck positions, all
        # failing at exactly 100% liquidation - see outcome_tracker.py's
        # _exit()
        client = self._make_client()
        captured = {}

        async def fake_sign_and_send(body):
            captured.update(body)
            return "fake_signature"

        client._sign_and_send = fake_sign_and_send
        result = asyncio.run(
            client.build_and_send_full_sell(mint="MINT123", slippage_pct=10, amount_pct=99)
        )

        self.assertEqual(captured["amount"], "99%")
        self.assertEqual(captured["denominatedInSol"], "false")
        self.assertEqual(result["amount"], "99%")

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

    def test_buy_defaults_to_pool_auto_not_bonding_curve_only(self):
        # confirmed live: a candidate that isn't a real pump.fun mint fails
        # identically under every pool value, but a REAL pump.fun position
        # can still migrate to PumpSwap while held - "pump" alone can only
        # ever route to the bonding curve, "auto" routes to wherever the
        # token actually trades right now
        client = self._make_client()
        captured = {}

        async def fake_sign_and_send(body):
            captured.update(body)
            return "fake_signature"

        client._sign_and_send = fake_sign_and_send
        asyncio.run(client.build_and_send_trade(
            action="buy", mint="MINT123", amount_sol=0.05, slippage_pct=10
        ))

        self.assertEqual(captured["pool"], "auto")

    def test_full_sell_defaults_to_pool_auto_not_bonding_curve_only(self):
        client = self._make_client()
        captured = {}

        async def fake_sign_and_send(body):
            captured.update(body)
            return "fake_signature"

        client._sign_and_send = fake_sign_and_send
        asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))

        self.assertEqual(captured["pool"], "auto")


class FetchRealSolDeltaTests(unittest.TestCase):
    """User-requested, real finding 2026-08-23: the flat fee-model pnl
    estimate was systematically too optimistic (real wallet balance drop
    was ~4x the bot's own tracked pnl over one session) - this fetches the
    actual on-chain SOL delta for a confirmed transaction so pnl on a sell
    can be computed from ground truth instead."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_computes_the_wallets_own_balance_delta(self):
        client = self._make_client()
        our_key = str(client.keypair.pubkey())
        response = {
            "result": {
                "transaction": {"message": {"accountKeys": ["OTHER_KEY", our_key]}},
                "meta": {"preBalances": [0, 1_000_000_000], "postBalances": [0, 1_050_000_000]},
            },
        }
        session = _FakeSession([response])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG"))
        self.assertAlmostEqual(delta, 0.05)

    def test_a_negative_delta_for_a_buy_style_transaction(self):
        client = self._make_client()
        our_key = str(client.keypair.pubkey())
        response = {
            "result": {
                "transaction": {"message": {"accountKeys": [our_key]}},
                "meta": {"preBalances": [1_000_000_000], "postBalances": [950_000_000]},
            },
        }
        session = _FakeSession([response])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG"))
        self.assertAlmostEqual(delta, -0.05)

    def test_requests_confirmed_commitment_not_the_finalized_default(self):
        # Real bug found live 2026-08-24 (user-reported: "catecoin this is
        # a false log", "wojakius also not correct"): no commitment meant
        # the RPC default of "finalized" - far stricter/slower than the
        # "confirmed" status _confirm_transaction() already waited for, so
        # called seconds later this almost always silently returned None,
        # falling back to a stale pre-sell price estimate (confirmed live:
        # two real trades logged as +40.2% and +101.9% "wins" that were
        # actually real on-chain losses). Must request "confirmed" to
        # match what's already been waited for.
        client = self._make_client()
        our_key = str(client.keypair.pubkey())
        response = {
            "result": {
                "transaction": {"message": {"accountKeys": [our_key]}},
                "meta": {"preBalances": [1_000_000_000], "postBalances": [1_050_000_000]},
            },
        }
        session = _FakeSession([response])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            asyncio.run(client._fetch_real_sol_delta("SIG"))
        sent_payload = session.post_calls[0][1]["json"]
        self.assertEqual(sent_payload["params"][1]["commitment"], "confirmed")

    def test_retries_when_not_yet_visible_then_succeeds(self):
        # the transaction can land a beat before getTransaction can see it
        # even at "confirmed" commitment (different RPC replica) - a short
        # retry recovers the real delta instead of silently giving up on
        # the first null result.
        client = self._make_client()
        our_key = str(client.keypair.pubkey())
        not_found = {"result": None}
        found = {
            "result": {
                "transaction": {"message": {"accountKeys": [our_key]}},
                "meta": {"preBalances": [1_000_000_000], "postBalances": [1_050_000_000]},
            },
        }
        session = _FakeSession([not_found, found])
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch("pumpfun_bot.pumpportal_client.asyncio.sleep", new=_fast_sleep):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG", retry_delay_sec=0))
        self.assertAlmostEqual(delta, 0.05)
        self.assertEqual(session.call_count, 2)

    def test_returns_none_when_the_transaction_is_not_found(self):
        client = self._make_client()
        session = _FakeSession([{"result": None}])
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch("pumpfun_bot.pumpportal_client.asyncio.sleep", new=_fast_sleep):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG", retry_delay_sec=0))
        self.assertIsNone(delta)
        self.assertEqual(session.call_count, 4)  # exhausted every retry, gave up

    def test_returns_none_on_fetch_exception(self):
        client = self._make_client()

        class _RaisingSession:
            def post(self, url, json=None, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=_RaisingSession()):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG"))
        self.assertIsNone(delta)

    def test_returns_none_when_our_key_is_not_in_account_keys(self):
        client = self._make_client()
        response = {
            "result": {
                "transaction": {"message": {"accountKeys": ["SOME_OTHER_KEY"]}},
                "meta": {"preBalances": [1_000_000_000], "postBalances": [950_000_000]},
            },
        }
        session = _FakeSession([response])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            delta = asyncio.run(client._fetch_real_sol_delta("SIG"))
        self.assertIsNone(delta)


class RealSolDeltaWiringTests(unittest.TestCase):
    """A sell's result must carry the real delta (used by outcome_tracker.py
    for ground-truth pnl); a buy must NOT trigger the extra lookup at all -
    buys are speed-critical (sniper's whole edge depends on it)."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_full_sell_result_includes_the_real_delta(self):
        client = self._make_client()

        async def fake_sign_and_send(body):
            return "fake_signature"

        async def fake_fetch_delta(sig):
            return 0.0423

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = fake_fetch_delta
        result = asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))

        self.assertAlmostEqual(result["real_sol_delta"], 0.0423)

    def test_sell_via_build_and_send_trade_includes_the_real_delta(self):
        client = self._make_client()

        async def fake_sign_and_send(body):
            return "fake_signature"

        async def fake_fetch_delta(sig):
            return 0.0111

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = fake_fetch_delta
        result = asyncio.run(client.build_and_send_trade(
            action="sell", mint="MINT123", amount_sol=0.02, slippage_pct=10,
        ))

        self.assertAlmostEqual(result["real_sol_delta"], 0.0111)

    def test_a_buy_never_triggers_the_real_delta_lookup(self):
        client = self._make_client()

        async def fake_sign_and_send(body):
            return "fake_signature"

        called = []

        async def fake_fetch_delta(sig):
            called.append(sig)
            return 0.01

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = fake_fetch_delta
        result = asyncio.run(client.build_and_send_trade(
            action="buy", mint="MINT123", amount_sol=0.05, slippage_pct=10,
        ))

        self.assertEqual(called, [])
        self.assertIsNone(result["real_sol_delta"])


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

    def test_the_raised_error_carries_the_parsed_custom_error_code(self):
        client = self._make_client()
        session = _FakeSession([
            _status_response(err={"InstructionError": [3, {"Custom": 6005}]}, confirmation_status="processed")
        ])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(OnChainTransactionError) as ctx:
                asyncio.run(client._confirm_transaction("SIG", timeout_sec=1, poll_interval_sec=0.01))
        self.assertEqual(ctx.exception.custom_error_code, 6005)

    def test_a_non_instruction_error_has_no_custom_error_code(self):
        client = self._make_client()
        session = _FakeSession([
            _status_response(err="SomeOtherKindOfError", confirmation_status="processed")
        ])
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(OnChainTransactionError) as ctx:
                asyncio.run(client._confirm_transaction("SIG", timeout_sec=1, poll_interval_sec=0.01))
        self.assertIsNone(ctx.exception.custom_error_code)

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


class MigrationRetryTests(unittest.TestCase):
    """Real finding 2026-08-23: a sell failed with AnchorError
    BondingCurveComplete (Custom 6005) even with pool="auto" - a real
    position pumped hard enough to migrate to Raydium/PumpSwap WHILE this
    bot held it, and "auto" didn't catch the fresh migration. One retry,
    forcing pool="pump-amm" explicitly since the error itself confirms the
    bonding curve is done."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_full_sell_retries_with_pump_amm_on_bonding_curve_complete(self):
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            if len(calls) == 1:
                raise OnChainTransactionError("boom", custom_error_code=6005)
            return "retry_signature"

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = lambda sig: _async_none()
        result = asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))

        self.assertEqual(result["signature"], "retry_signature")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["pool"], "auto")
        self.assertEqual(calls[1]["pool"], "pump-amm")

    def test_sell_via_build_and_send_trade_also_retries(self):
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            if len(calls) == 1:
                raise OnChainTransactionError("boom", custom_error_code=6005)
            return "retry_signature"

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = lambda sig: _async_none()
        result = asyncio.run(client.build_and_send_trade(
            action="sell", mint="MINT123", amount_sol=0.02, slippage_pct=10,
        ))

        self.assertEqual(result["signature"], "retry_signature")
        self.assertEqual(calls[1]["pool"], "pump-amm")

    def test_a_buy_never_retries_on_bonding_curve_complete(self):
        # a buy hitting an already-migrated token is a different, pre-buy
        # problem - "auto" already claims to handle routing a fresh buy
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            raise OnChainTransactionError("boom", custom_error_code=6005)

        client._sign_and_send = fake_sign_and_send
        with self.assertRaises(OnChainTransactionError):
            asyncio.run(client.build_and_send_trade(
                action="buy", mint="MINT123", amount_sol=0.05, slippage_pct=10,
            ))
        self.assertEqual(len(calls), 1)  # no retry attempted

    def test_does_not_retry_a_different_error_code(self):
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            raise OnChainTransactionError("boom", custom_error_code=6022)

        client._sign_and_send = fake_sign_and_send
        with self.assertRaises(OnChainTransactionError):
            asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))
        self.assertEqual(len(calls), 1)

    def test_does_not_retry_again_if_already_on_pump_amm(self):
        # avoids a pointless second failure/fee if pump-amm itself is
        # somehow still wrong - one retry, not an infinite loop
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            raise OnChainTransactionError("boom", custom_error_code=6005)

        client._sign_and_send = fake_sign_and_send
        with self.assertRaises(OnChainTransactionError):
            asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10, pool="pump-amm"))
        self.assertEqual(len(calls), 1)

    def test_a_successful_first_attempt_never_retries(self):
        client = self._make_client()
        calls = []

        async def fake_sign_and_send(body):
            calls.append(dict(body))
            return "sig"

        client._sign_and_send = fake_sign_and_send
        client._fetch_real_sol_delta = lambda sig: _async_none()
        asyncio.run(client.build_and_send_full_sell(mint="MINT123", slippage_pct=10))
        self.assertEqual(len(calls), 1)


async def _async_none():
    return None


def _fake_unsigned_tx_base58(signer_pubkey) -> str:
    """A genuinely valid, empty (zero-instruction) unsigned VersionedTransaction,
    base58-encoded - matches PumpPortal's real Jito-bundle-mode response
    shape closely enough to exercise the real decode -> sign round trip,
    not just a mocked call."""
    from solders.hash import Hash
    from solders.message import MessageV0

    msg = MessageV0.try_compile(
        payer=signer_pubkey, instructions=[], address_lookup_table_accounts=[],
        recent_blockhash=Hash.default(),
    )
    unsigned_tx = VersionedTransaction.populate(msg, [])
    return base58.b58encode(bytes(unsigned_tx)).decode("utf-8")


class _FakeJitoResponse:
    def __init__(self, status=200, json_data=None, text=""):
        self.status = status
        self._json_data = json_data
        self._text = text

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeJitoSession:
    def __init__(self, response):
        self._response = response
        self.post_calls = []

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class BuildAndSendFullSellViaJitoBundleTests(unittest.TestCase):
    """User-requested 2026-08-24 ("yes build if you think it will make the
    bot better") - submits a real sell as a Jito bundle instead of a
    normal sendTransaction, closing the trigger-to-landed-fill gap this
    session's own real data showed dominates real trading costs. Confirmed
    against PumpPortal's own docs: array-mode POST to the same trade-local
    endpoint, response is base58-encoded unsigned transactions (not raw
    bytes like single-object mode)."""

    def _make_client(self):
        return PumpPortalClient(
            ws_url="wss://example.invalid",
            trade_api_url="https://example.invalid/trade-local",
            rpc_http_url="https://example.invalid/rpc",
            keypair=Keypair(),
        )

    def test_posts_the_request_as_an_array_not_a_single_object(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        trade_local_session = _FakeJitoSession(
            _FakeJitoResponse(status=200, json_data=[encoded])
        )
        client._fetch_real_sol_delta = lambda sig: _async_none()

        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=trade_local_session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value="BUNDLE_ID",
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.poll_bundle_until_landed",
            return_value={"confirmation_status": "confirmed", "err": None},
        ):
            asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))

        args, kwargs = trade_local_session.post_calls[0]
        posted_body = kwargs["json"]
        self.assertIsInstance(posted_body, list)
        self.assertEqual(len(posted_body), 1)
        self.assertEqual(posted_body[0]["action"], "sell")
        self.assertEqual(posted_body[0]["amount"], "100%")

    def test_returns_the_signature_and_bundle_id_on_success(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        trade_local_session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[encoded]))
        client._fetch_real_sol_delta = lambda sig: _async_ret(0.5)

        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=trade_local_session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value="BUNDLE_ID",
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.poll_bundle_until_landed",
            return_value={"confirmation_status": "finalized", "err": None},
        ):
            result = asyncio.run(
                client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10)
            )

        self.assertEqual(result["bundle_id"], "BUNDLE_ID")
        self.assertEqual(result["real_sol_delta"], 0.5)
        self.assertTrue(len(result["signature"]) > 0)

    def test_raises_on_a_non_200_from_pumpportal(self):
        client = self._make_client()
        session = _FakeJitoSession(_FakeJitoResponse(status=500, text="server error"))
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))

    def test_raises_on_an_empty_transaction_array(self):
        client = self._make_client()
        session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[]))
        with patch("pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))

    def test_raises_when_the_bundle_never_gets_a_bundle_id(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[encoded]))
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))

    def test_raises_when_the_bundle_never_lands_within_the_timeout(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[encoded]))
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value="BUNDLE_ID",
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.poll_bundle_until_landed", return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))

    def test_raises_on_chain_transaction_error_when_the_bundle_lands_with_an_error(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[encoded]))
        landed_with_error = {
            "confirmation_status": "confirmed",
            "err": {"InstructionError": [4, {"Custom": 6024}]},
        }
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value="BUNDLE_ID",
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.poll_bundle_until_landed",
            return_value=landed_with_error,
        ):
            with self.assertRaises(OnChainTransactionError) as ctx:
                asyncio.run(client.build_and_send_full_sell_via_jito_bundle(mint="MINT123", slippage_pct=10))
        self.assertEqual(ctx.exception.custom_error_code, 6024)

    def test_amount_pct_override_is_used_in_the_request(self):
        client = self._make_client()
        encoded = _fake_unsigned_tx_base58(client.keypair.pubkey())
        session = _FakeJitoSession(_FakeJitoResponse(status=200, json_data=[encoded]))
        client._fetch_real_sol_delta = lambda sig: _async_none()
        with patch(
            "pumpfun_bot.pumpportal_client.aiohttp.ClientSession", return_value=session,
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.send_bundle", return_value="BUNDLE_ID",
        ), patch(
            "pumpfun_bot.pumpportal_client.jito.poll_bundle_until_landed",
            return_value={"confirmation_status": "confirmed", "err": None},
        ):
            result = asyncio.run(client.build_and_send_full_sell_via_jito_bundle(
                mint="MINT123", slippage_pct=10, amount_pct=99,
            ))
        args, kwargs = session.post_calls[0]
        self.assertEqual(kwargs["json"][0]["amount"], "99%")
        self.assertEqual(result["amount"], "99%")


async def _async_ret(value):
    return value


if __name__ == "__main__":
    unittest.main()
