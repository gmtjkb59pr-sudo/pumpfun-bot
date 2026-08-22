import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.smart_money_wallets import (
    extract_profitable_wallets,
    fetch_smart_money_wallets,
)


def _trade(wallet, pnl):
    return {"walletAddress": wallet, "realizedPnlUsd": pnl}


class ExtractProfitableWalletsTests(unittest.TestCase):
    def test_keeps_a_wallet_with_positive_pnl(self):
        result = extract_profitable_wallets([_trade("W1", "42.5")])
        self.assertEqual(result, ["W1"])

    def test_drops_a_wallet_with_negative_pnl(self):
        result = extract_profitable_wallets([_trade("W1", "-10.0")])
        self.assertEqual(result, [])

    def test_drops_a_wallet_with_zero_pnl(self):
        result = extract_profitable_wallets([_trade("W1", "0")])
        self.assertEqual(result, [])

    def test_dedupes_a_wallet_appearing_in_multiple_trades(self):
        result = extract_profitable_wallets([_trade("W1", "10"), _trade("W1", "20")])
        self.assertEqual(result, ["W1"])

    def test_drops_a_trade_with_no_wallet_address(self):
        result = extract_profitable_wallets([{"realizedPnlUsd": "10"}])
        self.assertEqual(result, [])

    def test_drops_a_trade_with_an_unparseable_pnl(self):
        result = extract_profitable_wallets([_trade("W1", "not-a-number")])
        self.assertEqual(result, [])

    def test_drops_a_trade_with_a_missing_pnl(self):
        result = extract_profitable_wallets([{"walletAddress": "W1"}])
        self.assertEqual(result, [])

    def test_preserves_first_seen_order(self):
        result = extract_profitable_wallets([_trade("W2", "5"), _trade("W1", "10")])
        self.assertEqual(result, ["W2", "W1"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(extract_profitable_wallets([]), [])


class FetchSmartMoneyWalletsTests(unittest.TestCase):
    def test_returns_wallets_from_a_successful_fetch(self):
        async def _fake_fetch(api_key, secret_key, passphrase, tracker_type=None, chain_index=None):
            return [_trade("W1", "10"), _trade("W2", "-5")]

        with patch("pumpfun_bot.smart_money_wallets.fetch_address_tracker_trades", _fake_fetch):
            wallets = asyncio.run(fetch_smart_money_wallets("key", "secret", "pass"))

        self.assertEqual(wallets, ["W1"])

    def test_returns_none_when_the_underlying_fetch_fails(self):
        async def _failing_fetch(api_key, secret_key, passphrase, tracker_type=None, chain_index=None):
            return None

        with patch("pumpfun_bot.smart_money_wallets.fetch_address_tracker_trades", _failing_fetch):
            wallets = asyncio.run(fetch_smart_money_wallets("key", "secret", "pass"))

        self.assertIsNone(wallets)


if __name__ == "__main__":
    unittest.main()
