import json
import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.stats import compute_stats


def write_log(lines: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for line in lines:
        tmp.write(json.dumps(line) + "\n")
    tmp.close()
    return Path(tmp.name)


class ComputeStatsTests(unittest.TestCase):
    def test_missing_file_returns_empty_stats(self):
        stats = compute_stats("/nonexistent/activity_log.jsonl")
        self.assertEqual(stats["total_trades"], 0)
        self.assertEqual(stats["total_outcomes"], 0)

    def test_aggregates_win_rate_and_avg_pct_change(self):
        log_path = write_log([
            {
                "type": "trade", "mint": "A", "dry_run": True,
                "meta": {"liquidity_sol": 10, "has_socials": True},
            },
            {
                "type": "trade", "mint": "B", "dry_run": True,
                "meta": {"liquidity_sol": 2, "has_socials": False},
            },
            {"type": "outcome", "mint": "A", "checkpoint_sec": 60, "pct_change": 20.0},
            {"type": "outcome", "mint": "B", "checkpoint_sec": 60, "pct_change": -10.0},
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["total_outcomes"], 2)

        cp60 = stats["by_checkpoint"]["60"]
        self.assertEqual(cp60["overall"]["count"], 2)
        self.assertEqual(cp60["overall"]["avg_pct_change"], 5.0)
        self.assertEqual(cp60["overall"]["win_rate_pct"], 50.0)

        self.assertEqual(cp60["by_socials"]["true"]["count"], 1)
        self.assertEqual(cp60["by_socials"]["true"]["avg_pct_change"], 20.0)
        self.assertEqual(cp60["by_socials"]["false"]["count"], 1)
        self.assertEqual(cp60["by_socials"]["false"]["avg_pct_change"], -10.0)

        self.assertEqual(cp60["by_liquidity"]["5-20 SOL"]["count"], 1)
        self.assertEqual(cp60["by_liquidity"]["<5 SOL"]["count"], 1)

    def test_unmeasured_outcomes_excluded_from_stats_but_counted(self):
        log_path = write_log([
            {
                "type": "trade", "mint": "A", "dry_run": True,
                "meta": {"liquidity_sol": 10, "has_socials": True},
            },
            {
                "type": "trade", "mint": "B", "dry_run": True,
                "meta": {"liquidity_sol": 10, "has_socials": True},
            },
            # A got real trade data, B never did (e.g. PumpPortal rejected the
            # subscription for lack of a funded API key) - B must NOT count as
            # a measured "0% change" outcome
            {"type": "outcome", "mint": "A", "checkpoint_sec": 60, "pct_change": 15.0, "measured": True},
            {"type": "outcome", "mint": "B", "checkpoint_sec": 60, "pct_change": None, "measured": False},
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        self.assertEqual(stats["total_outcomes"], 1)
        self.assertEqual(stats["total_unmeasured"], 1)
        cp60 = stats["by_checkpoint"]["60"]
        self.assertEqual(cp60["overall"]["count"], 1)
        self.assertEqual(cp60["overall"]["avg_pct_change"], 15.0)

    def test_ignores_live_trades_for_meta_lookup(self):
        log_path = write_log([
            {"type": "trade", "mint": "A", "dry_run": False, "meta": {}},
            {"type": "outcome", "mint": "A", "checkpoint_sec": 60, "pct_change": 5.0},
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        self.assertEqual(stats["total_trades"], 0)
        cp60 = stats["by_checkpoint"]["60"]
        self.assertEqual(cp60["overall"]["count"], 1)
        self.assertEqual(cp60["by_socials"]["false"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
