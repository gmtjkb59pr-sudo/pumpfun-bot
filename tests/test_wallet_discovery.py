import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.wallet_discovery as wallet_discovery
from pumpfun_bot.wallet_discovery import (
    aggregate_wallet_performance,
    fetch_and_cache_new_mints,
    is_candidate_wallet,
    load_cached_snapshots,
    repeat_candidates,
)

_ORIGINAL_CACHE_PATH = wallet_discovery.WALLET_DISCOVERY_CACHE_PATH


class WalletDiscoveryTestCase(unittest.TestCase):
    """Base class: redirects WALLET_DISCOVERY_CACHE_PATH to a temp file so
    these tests never touch the real data/wallet_discovery_cache.jsonl."""

    def setUp(self):
        f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        f.close()
        self._cache_path = Path(f.name)
        self._cache_path.unlink()  # start from "doesn't exist yet"
        wallet_discovery.WALLET_DISCOVERY_CACHE_PATH = self._cache_path

    def tearDown(self):
        wallet_discovery.WALLET_DISCOVERY_CACHE_PATH = _ORIGINAL_CACHE_PATH
        self._cache_path.unlink(missing_ok=True)


def _trader(owner, pnl, tags=None):
    return {"owner": owner, "totalPnl": pnl, "tags": tags or []}


class IsCandidateWalletTests(unittest.TestCase):
    def test_accepts_a_wallet_with_no_tags(self):
        self.assertTrue(is_candidate_wallet(_trader("W1", 10.0)))

    def test_accepts_a_wallet_tagged_smart_trader_only(self):
        self.assertTrue(is_candidate_wallet(_trader("W1", 10.0, tags=["smart_trader", "fresh"])))

    def test_rejects_a_wallet_tagged_sniper(self):
        self.assertFalse(is_candidate_wallet(_trader("W1", 10.0, tags=["sniper"])))

    def test_rejects_a_wallet_tagged_bundler(self):
        self.assertFalse(is_candidate_wallet(_trader("W1", 10.0, tags=["bundler"])))

    def test_rejects_a_wallet_tagged_dev(self):
        self.assertFalse(is_candidate_wallet(_trader("W1", 10.0, tags=["dev"])))

    def test_rejects_a_wallet_with_a_mix_of_good_and_excluded_tags(self):
        # smart_trader alone isn't enough to override a real red flag
        self.assertFalse(is_candidate_wallet(_trader("W1", 10.0, tags=["smart_trader", "bundler"])))


class AggregateWalletPerformanceTests(unittest.TestCase):
    def test_aggregates_pnl_and_appearances_across_mints(self):
        snapshots = {
            "MINT1": {"traders": [_trader("WALLET_A", 100.0), _trader("WALLET_B", 5.0)]},
            "MINT2": {"traders": [_trader("WALLET_A", 50.0)]},
        }
        result = aggregate_wallet_performance(snapshots)
        self.assertEqual(len(result["WALLET_A"]["appearances"]), 2)
        self.assertAlmostEqual(result["WALLET_A"]["total_pnl"], 150.0)
        self.assertEqual(len(result["WALLET_B"]["appearances"]), 1)

    def test_excludes_sniper_and_bundler_tagged_traders(self):
        snapshots = {
            "MINT1": {"traders": [
                _trader("SNIPER_WALLET", 999999.0, tags=["sniper"]),
                _trader("CLEAN_WALLET", 10.0),
            ]},
        }
        result = aggregate_wallet_performance(snapshots)
        self.assertNotIn("SNIPER_WALLET", result)
        self.assertIn("CLEAN_WALLET", result)

    def test_ignores_a_mint_with_no_traders_data(self):
        snapshots = {"MINT1": {"traders": None}}
        result = aggregate_wallet_performance(snapshots)
        self.assertEqual(result, {})

    def test_ignores_a_trader_entry_with_no_owner(self):
        snapshots = {"MINT1": {"traders": [{"totalPnl": 10.0, "tags": []}]}}
        result = aggregate_wallet_performance(snapshots)
        self.assertEqual(result, {})


class RepeatCandidatesTests(unittest.TestCase):
    def test_excludes_a_wallet_with_only_one_appearance(self):
        by_wallet = {"W1": {"appearances": [{"mint": "M1", "pnl": 100.0}], "total_pnl": 100.0}}
        self.assertEqual(repeat_candidates(by_wallet), [])

    def test_includes_a_wallet_with_two_or_more_appearances(self):
        by_wallet = {
            "W1": {
                "appearances": [{"mint": "M1", "pnl": 100.0}, {"mint": "M2", "pnl": 50.0}],
                "total_pnl": 150.0,
            },
        }
        candidates = repeat_candidates(by_wallet)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "W1")

    def test_sorts_by_total_pnl_descending(self):
        by_wallet = {
            "LOW": {"appearances": [{"mint": "M1", "pnl": 1}, {"mint": "M2", "pnl": 1}], "total_pnl": 2.0},
            "HIGH": {"appearances": [{"mint": "M1", "pnl": 500}, {"mint": "M2", "pnl": 500}], "total_pnl": 1000.0},
        }
        candidates = repeat_candidates(by_wallet)
        self.assertEqual([w for w, _ in candidates], ["HIGH", "LOW"])

    def test_respects_a_custom_min_appearances(self):
        by_wallet = {
            "W1": {
                "appearances": [{"mint": "M1", "pnl": 1}, {"mint": "M2", "pnl": 1}],
                "total_pnl": 2.0,
            },
        }
        self.assertEqual(repeat_candidates(by_wallet, min_appearances=3), [])


class LoadCachedSnapshotsTests(WalletDiscoveryTestCase):
    def test_returns_empty_dict_when_no_cache_file_exists(self):
        self.assertEqual(load_cached_snapshots(), {})

    def test_reads_back_appended_snapshots(self):
        wallet_discovery._append_snapshot("MINT1", [_trader("W1", 10.0)])
        wallet_discovery._append_snapshot("MINT2", None)

        snapshots = load_cached_snapshots()

        self.assertEqual(set(snapshots.keys()), {"MINT1", "MINT2"})
        self.assertEqual(snapshots["MINT1"]["traders"], [_trader("W1", 10.0)])
        self.assertIsNone(snapshots["MINT2"]["traders"])

    def test_a_later_record_for_the_same_mint_overwrites_an_earlier_one(self):
        wallet_discovery._append_snapshot("MINT1", [_trader("OLD", 1.0)])
        wallet_discovery._append_snapshot("MINT1", [_trader("NEW", 2.0)])

        snapshots = load_cached_snapshots()

        self.assertEqual(snapshots["MINT1"]["traders"], [_trader("NEW", 2.0)])

    def test_skips_malformed_lines(self):
        wallet_discovery.WALLET_DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(wallet_discovery.WALLET_DISCOVERY_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps({"mint": "MINT1", "traders": []}) + "\n")

        snapshots = load_cached_snapshots()

        self.assertEqual(set(snapshots.keys()), {"MINT1"})


class FetchAndCacheNewMintsTests(WalletDiscoveryTestCase):
    def test_fetches_only_uncached_mints(self):
        wallet_discovery._append_snapshot("ALREADY_CACHED", [_trader("W1", 1.0)])
        calls = []

        async def _fake_fetch(mint, api_key):
            calls.append(mint)
            return [_trader("W2", 2.0)]

        with patch("pumpfun_bot.wallet_discovery.fetch_top_traders", _fake_fetch):
            fetched = asyncio.run(fetch_and_cache_new_mints(
                ["ALREADY_CACHED", "NEW_MINT"], "fake-key", delay_sec=0,
            ))

        self.assertEqual(fetched, 1)
        self.assertEqual(calls, ["NEW_MINT"])

    def test_respects_the_calls_per_run_cap(self):
        async def _fake_fetch(mint, api_key):
            return [_trader("W1", 1.0)]

        with patch("pumpfun_bot.wallet_discovery.fetch_top_traders", _fake_fetch):
            fetched = asyncio.run(fetch_and_cache_new_mints(
                ["M1", "M2", "M3"], "fake-key", calls_per_run=2, delay_sec=0,
            ))

        self.assertEqual(fetched, 2)
        self.assertEqual(len(load_cached_snapshots()), 2)

    def test_caches_a_failed_lookup_too_so_it_is_not_retried_forever(self):
        async def _failing_fetch(mint, api_key):
            return None

        with patch("pumpfun_bot.wallet_discovery.fetch_top_traders", _failing_fetch):
            asyncio.run(fetch_and_cache_new_mints(["MINT1"], "fake-key", delay_sec=0))

        snapshots = load_cached_snapshots()
        self.assertIn("MINT1", snapshots)
        self.assertIsNone(snapshots["MINT1"]["traders"])


if __name__ == "__main__":
    unittest.main()
