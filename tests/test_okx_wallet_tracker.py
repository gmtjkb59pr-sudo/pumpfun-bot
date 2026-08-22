import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.okx_wallet_tracker as okx_wallet_tracker
from pumpfun_bot.okx_wallet_tracker import (
    aggregate_wallet_stats,
    load_cached_trades,
    record_snapshot,
    top_wallets,
)

_ORIGINAL_CACHE_PATH = okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH


class OkxWalletTrackerTestCase(unittest.TestCase):
    """Base class: redirects OKX_WALLET_TRACKER_CACHE_PATH to a temp file
    so these tests never touch the real data/okx_wallet_tracker_cache.jsonl."""

    def setUp(self):
        f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        f.close()
        self._cache_path = Path(f.name)
        self._cache_path.unlink()
        okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH = self._cache_path

    def tearDown(self):
        okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH = _ORIGINAL_CACHE_PATH
        self._cache_path.unlink(missing_ok=True)


def _trade(tx_hash, wallet, pnl):
    return {"txHash": tx_hash, "walletAddress": wallet, "realizedPnlUsd": pnl}


class LoadCachedTradesTests(OkxWalletTrackerTestCase):
    def test_returns_empty_dict_when_no_cache_file_exists(self):
        self.assertEqual(load_cached_trades(), {})

    def test_reads_back_appended_trades(self):
        okx_wallet_tracker._append_trades([_trade("TX1", "W1", "10")])
        trades = load_cached_trades()
        self.assertEqual(set(trades.keys()), {"TX1"})

    def test_skips_malformed_lines(self):
        okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps(_trade("TX1", "W1", "10")) + "\n")
        trades = load_cached_trades()
        self.assertEqual(set(trades.keys()), {"TX1"})

    def test_skips_a_trade_with_no_txhash(self):
        okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(okx_wallet_tracker.OKX_WALLET_TRACKER_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps({"walletAddress": "W1", "realizedPnlUsd": "10"}) + "\n")
        self.assertEqual(load_cached_trades(), {})


class RecordSnapshotTests(OkxWalletTrackerTestCase):
    def test_appends_only_new_trades_by_txhash(self):
        okx_wallet_tracker._append_trades([_trade("TX1", "W1", "10")])

        async def _fake_fetch(api_key, secret_key, passphrase, tracker_type=None, chain_index=None):
            return [_trade("TX1", "W1", "10"), _trade("TX2", "W2", "20")]

        with patch("pumpfun_bot.okx_wallet_tracker.fetch_address_tracker_trades", _fake_fetch):
            new_count = asyncio.run(record_snapshot("k", "s", "p"))

        self.assertEqual(new_count, 1)
        self.assertEqual(set(load_cached_trades().keys()), {"TX1", "TX2"})

    def test_returns_zero_when_the_underlying_fetch_fails(self):
        async def _failing_fetch(api_key, secret_key, passphrase, tracker_type=None, chain_index=None):
            return None

        with patch("pumpfun_bot.okx_wallet_tracker.fetch_address_tracker_trades", _failing_fetch):
            new_count = asyncio.run(record_snapshot("k", "s", "p"))

        self.assertEqual(new_count, 0)
        self.assertEqual(load_cached_trades(), {})

    def test_skips_trades_with_no_txhash_when_recording(self):
        async def _fake_fetch(api_key, secret_key, passphrase, tracker_type=None, chain_index=None):
            return [{"walletAddress": "W1", "realizedPnlUsd": "10"}]

        with patch("pumpfun_bot.okx_wallet_tracker.fetch_address_tracker_trades", _fake_fetch):
            new_count = asyncio.run(record_snapshot("k", "s", "p"))

        self.assertEqual(new_count, 0)


class AggregateWalletStatsTests(unittest.TestCase):
    def test_computes_win_rate_and_total_pnl(self):
        trades = {
            "TX1": _trade("TX1", "W1", "100"),
            "TX2": _trade("TX2", "W1", "-40"),
            "TX3": _trade("TX3", "W1", "20"),
        }
        stats = aggregate_wallet_stats(trades)
        self.assertEqual(stats["W1"]["trade_count"], 3)
        self.assertEqual(stats["W1"]["scored_count"], 3)
        self.assertEqual(stats["W1"]["win_count"], 2)
        self.assertAlmostEqual(stats["W1"]["total_pnl_usd"], 80.0)
        self.assertAlmostEqual(stats["W1"]["win_rate_pct"], 66.7)

    def test_ignores_a_trade_with_no_wallet_address(self):
        trades = {"TX1": {"txHash": "TX1", "realizedPnlUsd": "10"}}
        stats = aggregate_wallet_stats(trades)
        self.assertEqual(stats, {})

    def test_unparseable_pnl_counts_toward_trade_count_but_not_scored(self):
        trades = {"TX1": _trade("TX1", "W1", "not-a-number")}
        stats = aggregate_wallet_stats(trades)
        self.assertEqual(stats["W1"]["trade_count"], 1)
        self.assertEqual(stats["W1"]["scored_count"], 0)
        self.assertIsNone(stats["W1"]["win_rate_pct"])

    def test_separates_stats_per_wallet(self):
        trades = {
            "TX1": _trade("TX1", "W1", "10"),
            "TX2": _trade("TX2", "W2", "-5"),
        }
        stats = aggregate_wallet_stats(trades)
        self.assertEqual(set(stats.keys()), {"W1", "W2"})
        self.assertEqual(stats["W1"]["win_count"], 1)
        self.assertEqual(stats["W2"]["win_count"], 0)


class TopWalletsTests(unittest.TestCase):
    def test_excludes_wallets_below_min_trades(self):
        by_wallet = {"W1": {"scored_count": 2, "win_rate_pct": 100.0, "total_pnl_usd": 50.0}}
        self.assertEqual(top_wallets(by_wallet, min_trades=3), [])

    def test_includes_wallets_meeting_min_trades(self):
        by_wallet = {"W1": {"scored_count": 3, "win_rate_pct": 66.7, "total_pnl_usd": 80.0}}
        ranked = top_wallets(by_wallet, min_trades=3)
        self.assertEqual([w for w, _ in ranked], ["W1"])

    def test_sorts_by_win_rate_then_total_pnl_descending(self):
        by_wallet = {
            "HIGH_WIN_LOW_PNL": {"scored_count": 5, "win_rate_pct": 80.0, "total_pnl_usd": 10.0},
            "LOW_WIN_HIGH_PNL": {"scored_count": 5, "win_rate_pct": 40.0, "total_pnl_usd": 1000.0},
            "HIGH_WIN_HIGH_PNL": {"scored_count": 5, "win_rate_pct": 80.0, "total_pnl_usd": 500.0},
        }
        ranked = top_wallets(by_wallet, min_trades=3)
        self.assertEqual(
            [w for w, _ in ranked],
            ["HIGH_WIN_HIGH_PNL", "HIGH_WIN_LOW_PNL", "LOW_WIN_HIGH_PNL"],
        )

    def test_a_wallet_with_no_scored_trades_and_none_win_rate_is_never_ranked(self):
        by_wallet = {"W1": {"scored_count": 0, "win_rate_pct": None, "total_pnl_usd": 0.0}}
        self.assertEqual(top_wallets(by_wallet, min_trades=0), [])


if __name__ == "__main__":
    unittest.main()
