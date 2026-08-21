import json
import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.fees import net_pct_change_after_fees
from pumpfun_bot.wallet_reputation import blocked_wallets, compute_wallet_stats


def _write_log(lines: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for line in lines:
        f.write(json.dumps(line) + "\n")
    f.close()
    return f.name


class ComputeWalletStatsTests(unittest.TestCase):
    def test_joins_buy_creator_with_exit_outcome_by_mint(self):
        path = _write_log([
            {"type": "trade", "action": "buy", "mint": "M1", "meta": {"creator": "WALLET_A"}},
            {"type": "exit", "mint": "M1", "pct_change": -50.0},
        ])
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        stats = compute_wallet_stats(path)

        self.assertIn("WALLET_A", stats)
        self.assertEqual(stats["WALLET_A"]["count"], 1)
        self.assertAlmostEqual(
            stats["WALLET_A"]["median_pct_change"], net_pct_change_after_fees(-50.0)
        )

    def test_ignores_buys_without_a_creator(self):
        path = _write_log([
            {"type": "trade", "action": "buy", "mint": "M1", "meta": {}},
            {"type": "exit", "mint": "M1", "pct_change": -50.0},
        ])
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_wallet_stats(path), {})

    def test_ignores_still_open_positions_with_no_exit_yet(self):
        path = _write_log([
            {"type": "trade", "action": "buy", "mint": "M1", "meta": {"creator": "WALLET_A"}},
        ])
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_wallet_stats(path), {})

    def test_missing_file_returns_empty(self):
        self.assertEqual(compute_wallet_stats("/no/such/file.jsonl"), {})


class BlockedWalletsTests(unittest.TestCase):
    def _log_with_repeated_losses(self, wallet: str, count: int, pct_change: float) -> str:
        lines = []
        for i in range(count):
            mint = f"M{i}"
            lines.append({"type": "trade", "action": "buy", "mint": mint, "meta": {"creator": wallet}})
            lines.append({"type": "exit", "mint": mint, "pct_change": pct_change})
        return _write_log(lines)

    def test_blocks_wallet_with_enough_samples_and_negative_median(self):
        path = self._log_with_repeated_losses("SERIAL_RUGGER", count=3, pct_change=-80.0)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(blocked_wallets(path, min_samples=3), {"SERIAL_RUGGER"})

    def test_does_not_block_below_min_sample_threshold(self):
        path = self._log_with_repeated_losses("NEW_WALLET", count=2, pct_change=-80.0)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(blocked_wallets(path, min_samples=3), set())

    def test_does_not_block_a_profitable_wallet(self):
        path = self._log_with_repeated_losses("GOOD_WALLET", count=5, pct_change=+80.0)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(blocked_wallets(path, min_samples=3), set())


if __name__ == "__main__":
    unittest.main()
