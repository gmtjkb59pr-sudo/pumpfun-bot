import json
import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.sniper_model import (
    DEFAULT_CREATOR_WIN_RATE,
    WIN_MARGIN_PCT,
    build_creator_win_rates,
    extract_features,
    load_model,
    save_model,
    score,
    score_with_model,
    train_logistic_regression,
)


class ExtractFeaturesTests(unittest.TestCase):
    def test_extracts_the_three_features_in_order(self):
        meta = {"initial_buy_pct": 5.0, "liquidity_sol": 30.0, "creator": "WALLET_A"}
        features = extract_features(meta, {"WALLET_A": 0.7})
        self.assertEqual(features, [5.0, 30.0, 0.7])

    def test_missing_creator_falls_back_to_the_default_prior(self):
        meta = {"initial_buy_pct": 5.0, "liquidity_sol": 30.0}
        features = extract_features(meta, {})
        self.assertEqual(features[2], DEFAULT_CREATOR_WIN_RATE)

    def test_unknown_creator_falls_back_to_the_default_prior(self):
        meta = {"initial_buy_pct": 5.0, "liquidity_sol": 30.0, "creator": "NEVER_SEEN"}
        features = extract_features(meta, {"SOME_OTHER_WALLET": 0.9})
        self.assertEqual(features[2], DEFAULT_CREATOR_WIN_RATE)

    def test_returns_none_when_initial_buy_pct_is_missing(self):
        meta = {"liquidity_sol": 30.0, "creator": "WALLET_A"}
        self.assertIsNone(extract_features(meta, {}))

    def test_returns_none_when_liquidity_sol_is_missing(self):
        meta = {"initial_buy_pct": 5.0, "creator": "WALLET_A"}
        self.assertIsNone(extract_features(meta, {}))


class TrainLogisticRegressionTests(unittest.TestCase):
    def test_learns_a_perfectly_separable_toy_dataset(self):
        # feature 0 alone perfectly separates the classes - a converged
        # model should classify all of these correctly
        features = [[10.0, 30.0, 0.5], [12.0, 31.0, 0.5], [1.0, 30.0, 0.5], [2.0, 29.0, 0.5]]
        labels = [0, 0, 1, 1]

        model = train_logistic_regression(features, labels, epochs=500, lr=0.5)

        predictions = [1 if score_with_model(model, f) >= 0.5 else 0 for f in features]
        self.assertEqual(predictions, labels)

    def test_raises_on_empty_training_data(self):
        with self.assertRaises(ValueError):
            train_logistic_regression([], [])

    def test_handles_a_constant_feature_without_dividing_by_zero(self):
        # liquidity_sol-style feature with zero variance across every row
        features = [[1.0, 30.0], [2.0, 30.0], [3.0, 30.0]]
        labels = [0, 1, 0]
        model = train_logistic_regression(features, labels, epochs=10)
        # must not raise, and must produce a usable score
        result = score_with_model(model, [1.0, 30.0])
        self.assertTrue(0.0 <= result <= 1.0)


class SaveLoadModelTests(unittest.TestCase):
    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model = {"weights": [0.1, 0.2], "bias": 0.05, "means": [1.0, 2.0], "stds": [1.0, 1.0], "features": ["a", "b"]}
            save_model(model, path)
            loaded = load_model(path)
        self.assertEqual(loaded, model)

    def test_returns_none_when_no_model_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "does_not_exist.json"
            self.assertIsNone(load_model(path))

    def test_returns_none_on_a_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            path.write_text("not valid json{{{")
            self.assertIsNone(load_model(path))


class BuildCreatorWinRatesTests(unittest.TestCase):
    def _write_log(self, tmpdir, records):
        path = Path(tmpdir) / "activity_log.jsonl"
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_computes_the_real_win_rate_per_creator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                 "mint": "MINT_A", "meta": {"creator": "WALLET_A"}},
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 20.0},
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                 "mint": "MINT_B", "meta": {"creator": "WALLET_A"}},
                {"type": "exit", "dry_run": False, "mint": "MINT_B", "pct_change": -30.0},
            ]
            path = self._write_log(tmpdir, records)
            rates = build_creator_win_rates(path)
        self.assertAlmostEqual(rates["WALLET_A"], 0.5)

    def test_a_win_must_clear_the_fee_margin_not_just_be_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                 "mint": "MINT_A", "meta": {"creator": "WALLET_A"}},
                # a small positive move that doesn't clear WIN_MARGIN_PCT after fees
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": WIN_MARGIN_PCT - 1},
            ]
            path = self._write_log(tmpdir, records)
            rates = build_creator_win_rates(path)
        self.assertAlmostEqual(rates["WALLET_A"], 0.0)

    def test_ignores_dry_run_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": True,
                 "mint": "MINT_A", "meta": {"creator": "WALLET_A"}},
                {"type": "exit", "dry_run": True, "mint": "MINT_A", "pct_change": 50.0},
            ]
            path = self._write_log(tmpdir, records)
            rates = build_creator_win_rates(path)
        self.assertEqual(rates, {})

    def test_ignores_non_sniper_strategies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "social_watch", "dry_run": False,
                 "mint": "MINT_A", "meta": {"creator": "WALLET_A"}},
                {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 50.0},
            ]
            path = self._write_log(tmpdir, records)
            rates = build_creator_win_rates(path)
        self.assertEqual(rates, {})

    def test_ignores_a_buy_with_no_matching_exit_yet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                 "mint": "MINT_A", "meta": {"creator": "WALLET_A"}},
            ]
            path = self._write_log(tmpdir, records)
            rates = build_creator_win_rates(path)
        self.assertEqual(rates, {})

    def test_skips_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activity_log.jsonl"
            with path.open("w") as f:
                f.write("not valid json\n")
                f.write(json.dumps({
                    "type": "trade", "action": "buy", "strategy": "sniper", "dry_run": False,
                    "mint": "MINT_A", "meta": {"creator": "WALLET_A"},
                }) + "\n")
                f.write(json.dumps(
                    {"type": "exit", "dry_run": False, "mint": "MINT_A", "pct_change": 50.0}
                ) + "\n")
            rates = build_creator_win_rates(path)  # must not raise
        self.assertAlmostEqual(rates["WALLET_A"], 1.0)


class ScoreTests(unittest.TestCase):
    def test_returns_none_when_no_model_is_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model_path = Path(tmpdir) / "no_model.json"
            result = score({"initial_buy_pct": 5.0, "liquidity_sol": 30.0}, {}, model=load_model(missing_model_path))
        self.assertIsNone(result)

    def test_returns_none_when_features_are_incomplete(self):
        model = {"weights": [0.1, 0.1, 0.1], "bias": 0.0, "means": [0, 0, 0], "stds": [1, 1, 1], "features": []}
        result = score({"liquidity_sol": 30.0}, {}, model=model)  # missing initial_buy_pct
        self.assertIsNone(result)

    def test_returns_a_probability_when_everything_is_available(self):
        model = {"weights": [0.1, 0.1, 0.1], "bias": 0.0, "means": [0, 0, 0], "stds": [1, 1, 1], "features": []}
        result = score({"initial_buy_pct": 5.0, "liquidity_sol": 30.0, "creator": "W"}, {"W": 0.6}, model=model)
        self.assertIsNotNone(result)
        self.assertTrue(0.0 <= result <= 1.0)


if __name__ == "__main__":
    unittest.main()
