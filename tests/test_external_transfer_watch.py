import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pumpfun_bot.external_transfer_watch import (
    _load_own_signatures,
    check_for_external_transfers,
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


class _FakeRpcSession:
    """Dispatches by RPC method, since one poll cycle calls
    getSignaturesForAddress once and getTransaction potentially several
    times (once per new external signature) - unlike PumpPortalClient's
    tests, which only ever call ONE method per test."""

    def __init__(self, *, signatures_result, tx_results_by_sig):
        self._signatures_result = signatures_result
        self._tx_results_by_sig = tx_results_by_sig
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        method = json["method"]
        if method == "getSignaturesForAddress":
            return _FakeResponse({"result": self._signatures_result})
        if method == "getTransaction":
            sig = json["params"][0]
            return _FakeResponse({"result": self._tx_results_by_sig.get(sig)})
        raise AssertionError(f"unexpected RPC method: {method}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _tx_result(wallet, delta_sol):
    lamports = int(delta_sol * 1_000_000_000)
    return {
        "transaction": {"message": {"accountKeys": [wallet, "OTHER"]}},
        "meta": {"preBalances": [1_000_000_000, 0], "postBalances": [1_000_000_000 + lamports, 0]},
    }


class LoadOwnSignaturesTests(unittest.TestCase):
    def test_collects_signatures_from_both_trade_and_exit_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activity_log.jsonl"
            with path.open("w") as f:
                f.write(json.dumps({"ts": 100.0, "tx_signature": "BUY_SIG"}) + "\n")
                f.write(json.dumps({"ts": 101.0, "tx_signature": "EXIT_SIG"}) + "\n")

            signatures = _load_own_signatures(path, since_ts=0.0)
        self.assertEqual(signatures, {"BUY_SIG", "EXIT_SIG"})

    def test_ignores_records_before_since_ts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activity_log.jsonl"
            with path.open("w") as f:
                f.write(json.dumps({"ts": 50.0, "tx_signature": "OLD_SIG"}) + "\n")
                f.write(json.dumps({"ts": 150.0, "tx_signature": "NEW_SIG"}) + "\n")

            signatures = _load_own_signatures(path, since_ts=100.0)
        self.assertEqual(signatures, {"NEW_SIG"})

    def test_ignores_records_without_a_signature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activity_log.jsonl"
            with path.open("w") as f:
                f.write(json.dumps({"ts": 100.0, "type": "alert"}) + "\n")

            signatures = _load_own_signatures(path, since_ts=0.0)
        self.assertEqual(signatures, set())

    def test_missing_file_returns_empty_set(self):
        signatures = _load_own_signatures(Path("/nonexistent/activity_log.jsonl"), since_ts=0.0)
        self.assertEqual(signatures, set())


class CheckForExternalTransfersTests(unittest.TestCase):
    """User-requested 2026-08-24 ("the actual profit is not right because
    i topped up the bot with 10 dollar") - a deposit landing mid-session
    must be detected and excluded from real_pnl_sol, not counted as
    trading profit."""

    WALLET = "WALLET_PUBKEY"

    def _own_signatures_log(self, tmpdir, own_sigs):
        path = Path(tmpdir) / "activity_log.jsonl"
        with path.open("w") as f:
            for sig in own_sigs:
                f.write(json.dumps({"ts": 200.0, "tx_signature": sig}) + "\n")
        return path

    def test_a_deposit_not_in_our_own_signatures_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=["OUR_BUY_SIG"])
            session = _FakeRpcSession(
                signatures_result=[{"signature": "DEPOSIT_SIG", "blockTime": 300, "err": None}],
                tx_results_by_sig={"DEPOSIT_SIG": _tx_result(self.WALLET, 0.10634)},
            )
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                new_cursor = asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature=None, session=session,
                ))
        mock_state.add_external_transfer.assert_called_once()
        self.assertAlmostEqual(mock_state.add_external_transfer.call_args[0][0], 0.10634, places=6)
        self.assertEqual(new_cursor, "DEPOSIT_SIG")

    def test_our_own_trade_signature_is_not_treated_as_external(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=["OUR_BUY_SIG"])
            session = _FakeRpcSession(
                signatures_result=[{"signature": "OUR_BUY_SIG", "blockTime": 300, "err": None}],
                tx_results_by_sig={"OUR_BUY_SIG": _tx_result(self.WALLET, -0.012)},
            )
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature=None, session=session,
                ))
        mock_state.add_external_transfer.assert_not_called()

    def test_a_signature_from_before_session_start_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=[])
            session = _FakeRpcSession(
                signatures_result=[{"signature": "OLD_SIG", "blockTime": 50, "err": None}],
                tx_results_by_sig={"OLD_SIG": _tx_result(self.WALLET, 1.0)},
            )
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature=None, session=session,
                ))
        mock_state.add_external_transfer.assert_not_called()

    def test_a_failed_transaction_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=[])
            session = _FakeRpcSession(
                signatures_result=[{"signature": "FAILED_SIG", "blockTime": 300, "err": {"InstructionError": []}}],
                tx_results_by_sig={},
            )
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature=None, session=session,
                ))
        mock_state.add_external_transfer.assert_not_called()

    def test_no_new_signatures_returns_the_cursor_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=[])
            session = _FakeRpcSession(signatures_result=[], tx_results_by_sig={})
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                new_cursor = asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature="PREVIOUS_CURSOR", session=session,
                ))
        mock_state.add_external_transfer.assert_not_called()
        self.assertEqual(new_cursor, "PREVIOUS_CURSOR")

    def test_a_withdrawal_reports_a_negative_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = self._own_signatures_log(tmpdir, own_sigs=[])
            session = _FakeRpcSession(
                signatures_result=[{"signature": "WITHDRAW_SIG", "blockTime": 300, "err": None}],
                tx_results_by_sig={"WITHDRAW_SIG": _tx_result(self.WALLET, -0.4)},
            )
            with patch("pumpfun_bot.external_transfer_watch.DATA_LOG_PATH", log_path), patch(
                "pumpfun_bot.external_transfer_watch.bot_state",
            ) as mock_state:
                asyncio.run(check_for_external_transfers(
                    self.WALLET, "https://example.invalid/rpc", session_start_ts=100.0,
                    last_seen_signature=None, session=session,
                ))
        self.assertAlmostEqual(mock_state.add_external_transfer.call_args[0][0], -0.4, places=6)


if __name__ == "__main__":
    unittest.main()
