import json
import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.fees import net_pct_change_after_fees
from pumpfun_bot.holder_count_tuning import compute_holder_count_outcomes, decide_min_holder_count


def _write_log(lines: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for line in lines:
        f.write(json.dumps(line) + "\n")
    f.close()
    return f.name


def _round_trip(mint: str, holder_count: int, pct_change: float, strategy: str = "social_watch") -> list[dict]:
    return [
        {
            "type": "trade", "action": "buy", "strategy": strategy, "mint": mint,
            "meta": {"holder_count": holder_count},
        },
        {"type": "exit", "mint": mint, "pct_change": pct_change},
    ]


_social_watch_round_trip = _round_trip


class ComputeHolderCountOutcomesTests(unittest.TestCase):
    def test_pairs_up_holder_count_with_net_outcome(self):
        path = _write_log(_social_watch_round_trip("M1", 20, 50.0))
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        pairs = compute_holder_count_outcomes(path)

        self.assertEqual(pairs, [(20, net_pct_change_after_fees(50.0))])

    def test_ignores_sniper_buys(self):
        lines = [
            {
                "type": "trade", "action": "buy", "strategy": "sniper", "mint": "M1",
                "meta": {"holder_count": 20},
            },
            {"type": "exit", "mint": "M1", "pct_change": 50.0},
        ]
        path = _write_log(lines)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_holder_count_outcomes(path), [])

    def test_ignores_buys_with_no_holder_count(self):
        lines = [
            {"type": "trade", "action": "buy", "strategy": "social_watch", "mint": "M1", "meta": {}},
            {"type": "exit", "mint": "M1", "pct_change": 50.0},
        ]
        path = _write_log(lines)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_holder_count_outcomes(path), [])

    def test_ignores_still_open_positions(self):
        lines = [{
            "type": "trade", "action": "buy", "strategy": "social_watch", "mint": "M1",
            "meta": {"holder_count": 20},
        }]
        path = _write_log(lines)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_holder_count_outcomes(path), [])

    def test_missing_file_returns_empty(self):
        self.assertEqual(compute_holder_count_outcomes("/no/such/file.jsonl"), [])

    def test_default_strategy_excludes_birdeye_movers_buys(self):
        path = _write_log(_round_trip("M1", 20, 50.0, strategy="birdeye_movers"))
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        self.assertEqual(compute_holder_count_outcomes(path), [])

    def test_can_be_scoped_to_birdeye_movers_instead(self):
        lines = _round_trip("M1", 20, 50.0, strategy="birdeye_movers") + _round_trip("M2", 30, 10.0)
        path = _write_log(lines)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        pairs = compute_holder_count_outcomes(path, strategy="birdeye_movers")

        self.assertEqual(pairs, [(20, net_pct_change_after_fees(50.0))])


class DecideMinHolderCountTests(unittest.TestCase):
    def _log_with_bucket(self, *, low_count: int, low_pct: float, high_count: int, high_pct: float,
                          n_low: int, n_high: int) -> str:
        lines = []
        for i in range(n_low):
            lines += _social_watch_round_trip(f"LOW{i}", low_count, low_pct)
        for i in range(n_high):
            lines += _social_watch_round_trip(f"HIGH{i}", high_count, high_pct)
        return _write_log(lines)

    def test_recommends_raising_when_high_bucket_clearly_outperforms(self):
        # low group sits under every bucket threshold, high group clears all
        # of them - with only two distinct holder_count values, every
        # qualifying bucket ends up with an identical median (the high
        # group), so the lowest threshold that clears the bar wins: it
        # achieves the same measured improvement while filtering out fewer
        # candidates than necessary
        path = self._log_with_bucket(
            low_count=5, low_pct=0.0, high_count=30, high_pct=30.0, n_low=15, n_high=15,
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        changes = decide_min_holder_count(path, current_min_holder_count=0, min_samples=15, margin_pct=10.0)

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "min_holder_count")
        self.assertEqual(changes[0]["from"], 0)
        self.assertEqual(changes[0]["to"], 10)

    def test_no_change_below_sample_size(self):
        path = self._log_with_bucket(
            low_count=5, low_pct=0.0, high_count=20, high_pct=30.0, n_low=3, n_high=3,
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        changes = decide_min_holder_count(path, current_min_holder_count=0, min_samples=15, margin_pct=10.0)
        self.assertEqual(changes, [])

    def test_no_change_when_margin_not_cleared(self):
        path = self._log_with_bucket(
            low_count=5, low_pct=10.0, high_count=20, high_pct=12.0, n_low=15, n_high=15,
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        changes = decide_min_holder_count(path, current_min_holder_count=0, min_samples=15, margin_pct=10.0)
        self.assertEqual(changes, [])

    def test_no_change_when_already_at_or_above_every_bucket(self):
        path = self._log_with_bucket(
            low_count=5, low_pct=0.0, high_count=20, high_pct=30.0, n_low=15, n_high=15,
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        # already at 25 - the highest defined bucket - nothing left to tighten to
        changes = decide_min_holder_count(path, current_min_holder_count=25, min_samples=15, margin_pct=10.0)
        self.assertEqual(changes, [])

    def test_empty_log_yields_no_change(self):
        path = _write_log([])
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.assertEqual(decide_min_holder_count(path), [])

    def test_field_name_override_is_used_in_the_returned_change(self):
        path = self._log_with_bucket(
            low_count=5, low_pct=0.0, high_count=30, high_pct=30.0, n_low=15, n_high=15,
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        changes = decide_min_holder_count(
            path, current_min_holder_count=0, min_samples=15, margin_pct=10.0,
            field_name="birdeye_movers_min_holder_count",
        )

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "birdeye_movers_min_holder_count")

    def test_scopes_to_the_given_strategy_only(self):
        lines = []
        for i in range(15):
            lines += _round_trip(f"LOW{i}", 5, 0.0, strategy="birdeye_movers")
            lines += _round_trip(f"HIGH{i}", 30, 30.0, strategy="birdeye_movers")
            # same shape, but the wrong strategy - must not contribute
            lines += _round_trip(f"OTHER{i}", 30, 30.0, strategy="social_watch")
        path = _write_log(lines)
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        changes = decide_min_holder_count(
            path, current_min_holder_count=0, min_samples=15, margin_pct=10.0,
            strategy="birdeye_movers",
        )

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["to"], 10)


if __name__ == "__main__":
    unittest.main()
