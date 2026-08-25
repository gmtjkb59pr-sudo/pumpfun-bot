import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pumpfun_bot.social_watch_model import (
    DEFAULT_LAUNCH_MAX_TXS_IN_ONE_SLOT,
    DEFAULT_MARKET_CAP_USD,
    DEFAULT_TOP10_CONCENTRATION_PCT,
    MIN_TRAINING_ROWS,
    extract_features,
    load_corrections,
    load_labeled_dataset,
    retrain_and_save,
    score,
)


class ExtractFeaturesTests(unittest.TestCase):
    def test_extracts_the_four_features_in_order(self):
        meta = {
            "holder_count": 50, "top10_concentration_pct": 20.0,
            "market_cap_usd": 100_000.0, "launch_max_txs_in_one_slot": 3,
        }
        self.assertEqual(extract_features(meta), [50.0, 20.0, 100_000.0, 3.0])

    def test_missing_holder_count_returns_none(self):
        meta = {"top10_concentration_pct": 20.0, "market_cap_usd": 100_000.0}
        self.assertIsNone(extract_features(meta))

    def test_missing_top10_concentration_falls_back_to_the_sentinel(self):
        # real for rows logged before social_watch.py started saving this
        # field (2026-08-24)
        meta = {"holder_count": 50}
        features = extract_features(meta)
        self.assertEqual(features[1], DEFAULT_TOP10_CONCENTRATION_PCT)

    def test_missing_market_cap_falls_back_to_the_sentinel(self):
        meta = {"holder_count": 50}
        features = extract_features(meta)
        self.assertEqual(features[2], DEFAULT_MARKET_CAP_USD)

    def test_missing_launch_max_txs_falls_back_to_the_sentinel(self):
        meta = {"holder_count": 50}
        features = extract_features(meta)
        self.assertEqual(features[3], DEFAULT_LAUNCH_MAX_TXS_IN_ONE_SLOT)

    def test_a_genuinely_observed_zero_concentration_is_not_confused_with_the_sentinel(self):
        meta = {"holder_count": 50, "top10_concentration_pct": 0.0}
        features = extract_features(meta)
        self.assertEqual(features[1], 0.0)


class LoadCorrectionsTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "does_not_exist.json"
            self.assertEqual(load_corrections(path), {})

    def test_returns_none_on_a_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrections.json"
            path.write_text("not json")
            self.assertEqual(load_corrections(path), {})


class LoadLabeledDatasetTests(unittest.TestCase):
    def _write_log(self, tmpdir, records):
        path = Path(tmpdir) / "activity_log.jsonl"
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_ignores_dry_run_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": True,
                 "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50}},
                {"type": "exit", "dry_run": True, "mint": "MINT_A", "pct_change": 50.0},
            ]
            path = self._write_log(tmpdir, records)
            features, labels = load_labeled_dataset(path)
        self.assertEqual(features, [])
        self.assertEqual(labels, [])

    def test_ignores_non_social_watch_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                 "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50}},
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 50.0},
            ]
            path = self._write_log(tmpdir, records)
            features, labels = load_labeled_dataset(path)
        self.assertEqual(features, [])

    def test_ignores_a_buy_with_no_matching_exit_yet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                 "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50}},
            ]
            path = self._write_log(tmpdir, records)
            features, labels = load_labeled_dataset(path)
        self.assertEqual(features, [])

    def test_skips_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activity_log.jsonl"
            with path.open("w") as f:
                f.write("not valid json\n")
                f.write(json.dumps({
                    "type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                    "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50},
                }) + "\n")
                f.write(json.dumps(
                    {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 50.0}
                ) + "\n")
            features, labels = load_labeled_dataset(path)  # must not raise
        self.assertEqual(labels, [1])

    def test_uses_the_corrected_pct_change_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                 "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50}},
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 101.9,
                 "tx_signature": "SIG_A"},
            ]
            path = self._write_log(tmpdir, records)
            corrections = {"SIG_A": {"corrected_pct_change": -46.0}}
            features, labels = load_labeled_dataset(path, corrections)
        self.assertEqual(labels, [0])  # corrected -46% is a loss, not the original "+101.9%" win

    def test_falls_back_to_the_original_pct_change_when_uncorrected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                 "mint": "MINT_A", "ts": 1000.0, "meta": {"holder_count": 50}},
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 50.0,
                 "tx_signature": "SIG_A"},
            ]
            path = self._write_log(tmpdir, records)
            features, labels = load_labeled_dataset(path, corrections={})
        self.assertEqual(labels, [1])


class ScoreTests(unittest.TestCase):
    def test_returns_none_when_no_model_is_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model_path = Path(tmpdir) / "no_model.json"
            with patch("pumpfun_bot.social_watch_model.MODEL_PATH", missing_model_path):
                result = score({"holder_count": 50})
        self.assertIsNone(result)

    def test_returns_none_when_holder_count_is_missing(self):
        model = {
            "weights": [0.1, 0.1, 0.1, 0.1], "bias": 0.0,
            "means": [0, 0, 0, 0], "stds": [1, 1, 1, 1], "features": [],
        }
        result = score({"top10_concentration_pct": 20.0}, model=model)
        self.assertIsNone(result)

    def test_returns_a_probability_when_everything_is_available(self):
        model = {
            "weights": [0.1, 0.1, 0.1, 0.1], "bias": 0.0,
            "means": [0, 0, 0, 0], "stds": [1, 1, 1, 1], "features": [],
        }
        result = score(
            {
                "holder_count": 50, "top10_concentration_pct": 20.0,
                "market_cap_usd": 100_000.0, "launch_max_txs_in_one_slot": 2,
            },
            model=model,
        )
        self.assertIsNotNone(result)
        self.assertTrue(0.0 <= result <= 1.0)


def _write_perfectly_separable_dataset(tmpdir, n_per_class=20):
    """n_per_class wins (low top10_concentration_pct, real gain) vs
    n_per_class losses (high top10_concentration_pct, real loss) - mirrors
    sniper_model's test file's identically-purposed helper."""
    path = Path(tmpdir) / "activity_log.jsonl"
    records = []
    for i in range(n_per_class):
        for is_win, concentration, pct_change in ((True, 5.0, 50.0), (False, 80.0, -50.0)):
            mint = f"MINT_{'WIN' if is_win else 'LOSS'}_{i}"
            ts = i * 2 + (0 if is_win else 1)
            records.append({
                "type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                "mint": mint, "ts": float(ts),
                "meta": {
                    "holder_count": 50, "top10_concentration_pct": concentration,
                    "market_cap_usd": 100_000.0, "launch_max_txs_in_one_slot": 1,
                },
            })
            records.append({
                "type": "exit", "dry_run": False, "mint": mint,
                "pct_change": pct_change, "tx_signature": f"SIG_{mint}",
            })
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


class RetrainAndSaveTests(unittest.TestCase):
    def test_returns_none_below_min_training_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_perfectly_separable_dataset(tmpdir, n_per_class=1)  # only 2 rows
            self.assertLess(2, MIN_TRAINING_ROWS)
            result = retrain_and_save(activity_log_path=path, model_path=Path(tmpdir) / "model.json")
        self.assertIsNone(result)

    def test_returns_none_when_the_activity_log_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "does_not_exist.jsonl"
            result = retrain_and_save(activity_log_path=missing, model_path=Path(tmpdir) / "model.json")
        self.assertIsNone(result)

    def test_trains_and_saves_a_model_on_enough_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_perfectly_separable_dataset(tmpdir, n_per_class=20)  # 40 rows
            model_path = Path(tmpdir) / "model.json"
            result = retrain_and_save(activity_log_path=path, model_path=model_path)

            self.assertIsNotNone(result)
            self.assertEqual(result["n"], 40)
            self.assertEqual(result["train_n"] + result["holdout_n"], 40)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertTrue(result["beats_baseline"])
            self.assertTrue(model_path.exists())
            saved = json.loads(model_path.read_text())
            self.assertIn("weights", saved)

    def test_uses_corrections_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_perfectly_separable_dataset(tmpdir, n_per_class=20)
            corrections_path = Path(tmpdir) / "corrections.json"
            corrections = {
                f"SIG_MINT_WIN_{i}": {"corrected_pct_change": -50.0} for i in range(20)
            }
            corrections_path.write_text(json.dumps(corrections))
            model_path = Path(tmpdir) / "model.json"

            result = retrain_and_save(
                activity_log_path=path, corrections_path=corrections_path, model_path=model_path,
            )
        self.assertIsNotNone(result)
        self.assertFalse(result["beats_baseline"])


if __name__ == "__main__":
    unittest.main()
