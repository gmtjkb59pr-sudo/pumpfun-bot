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
        self.assertEqual(stats["exits"]["total"], 0)

    def test_aggregates_win_rate_mean_and_median(self):
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
        self.assertEqual(cp60["overall"]["mean_pct_change"], 5.0)
        self.assertEqual(cp60["overall"]["median_pct_change"], 5.0)
        self.assertEqual(cp60["overall"]["win_rate_pct"], 50.0)

        self.assertEqual(cp60["by_socials"]["true"]["count"], 1)
        self.assertEqual(cp60["by_socials"]["true"]["median_pct_change"], 20.0)
        self.assertEqual(cp60["by_socials"]["false"]["count"], 1)
        self.assertEqual(cp60["by_socials"]["false"]["median_pct_change"], -10.0)

        self.assertEqual(cp60["by_liquidity"]["5-20 SOL"]["count"], 1)
        self.assertEqual(cp60["by_liquidity"]["<5 SOL"]["count"], 1)

    def test_median_is_not_skewed_by_a_single_outlier(self):
        # a couple of values near zero plus one enormous outlier should barely
        # move the median even though it wrecks the mean
        log_path = write_log([
            {"type": "outcome", "mint": "A", "checkpoint_sec": 60, "pct_change": 1.0},
            {"type": "outcome", "mint": "B", "checkpoint_sec": 60, "pct_change": -2.0},
            {"type": "outcome", "mint": "C", "checkpoint_sec": 60, "pct_change": 3.0},
            {"type": "outcome", "mint": "D", "checkpoint_sec": 60, "pct_change": 200000.0},
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        cp60 = stats["by_checkpoint"]["60"]["overall"]
        self.assertGreater(cp60["mean_pct_change"], 1000)
        self.assertLess(cp60["median_pct_change"], 5)

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
        self.assertEqual(cp60["overall"]["median_pct_change"], 15.0)

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

    def test_aggregates_exits_by_reason_and_realized_pnl(self):
        log_path = write_log([
            {"type": "exit", "mint": "A", "reason": "take_profit", "pct_change": 50.0, "trade_size_sol": 0.05},
            {"type": "exit", "mint": "B", "reason": "stop_loss", "pct_change": -25.0, "trade_size_sol": 0.05},
            {"type": "exit", "mint": "C", "reason": "take_profit", "pct_change": 60.0, "trade_size_sol": 0.05},
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        self.assertEqual(stats["exits"]["total"], 3)
        self.assertAlmostEqual(
            stats["exits"]["total_realized_pnl_sol"],
            0.05 * 0.5 + 0.05 * -0.25 + 0.05 * 0.6,
            places=6,
        )
        self.assertEqual(stats["exits"]["by_reason"]["take_profit"]["count"], 2)
        self.assertEqual(stats["exits"]["by_reason"]["take_profit"]["win_rate_pct"], 100.0)
        self.assertEqual(stats["exits"]["by_reason"]["stop_loss"]["count"], 1)
        self.assertEqual(stats["exits"]["by_reason"]["stop_loss"]["win_rate_pct"], 0.0)

    def test_aggregates_counterfactual_hold_by_reason_and_checkpoint(self):
        log_path = write_log([
            # holding would have beaten the take-profit exit by 30pp
            {
                "type": "post_exit_check", "mint": "A", "exit_reason": "take_profit",
                "checkpoint_sec_after_exit": 900, "vs_realized_pct": 30.0,
            },
            # the stop-loss exit was the right call - holding would have been worse
            {
                "type": "post_exit_check", "mint": "B", "exit_reason": "stop_loss",
                "checkpoint_sec_after_exit": 900, "vs_realized_pct": -15.0,
            },
            # unmeasured post-exit checks must not count as a 0pp data point
            {
                "type": "post_exit_check", "mint": "C", "exit_reason": "take_profit",
                "checkpoint_sec_after_exit": 900, "vs_realized_pct": None,
            },
        ])
        try:
            stats = compute_stats(log_path)
        finally:
            log_path.unlink()

        cf900 = stats["counterfactual_hold"]["900"]
        self.assertEqual(cf900["take_profit"]["count"], 1)
        self.assertEqual(cf900["take_profit"]["median_pct_change"], 30.0)
        self.assertEqual(cf900["stop_loss"]["count"], 1)
        self.assertEqual(cf900["stop_loss"]["median_pct_change"], -15.0)


if __name__ == "__main__":
    unittest.main()
