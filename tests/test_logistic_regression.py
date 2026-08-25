import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.logistic_regression import (
    load_model,
    save_model,
    score_with_model,
    train_logistic_regression,
)


class TrainLogisticRegressionTests(unittest.TestCase):
    def test_learns_a_clean_linear_separation(self):
        features = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]
        labels = [0, 0, 0, 1, 1, 1]
        model = train_logistic_regression(features, labels, ["x"], epochs=500, lr=0.5)
        predictions = [1 if score_with_model(model, f) >= 0.5 else 0 for f in features]
        self.assertEqual(predictions, labels)

    def test_raises_on_empty_dataset(self):
        with self.assertRaises(ValueError):
            train_logistic_regression([], [], ["x"])

    def test_records_the_given_feature_names(self):
        model = train_logistic_regression([[1.0, 2.0]], [1], ["a", "b"], epochs=1)
        self.assertEqual(model["features"], ["a", "b"])

    def test_score_is_a_probability_between_zero_and_one(self):
        model = train_logistic_regression([[1.0, 30.0], [5.0, 32.0]], [0, 1], ["a", "b"], epochs=10)
        result = score_with_model(model, [1.0, 30.0])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class SaveLoadModelTests(unittest.TestCase):
    def test_round_trips_through_disk(self):
        model = train_logistic_regression([[1.0], [5.0]], [0, 1], ["x"], epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "model.json"
            save_model(model, path)
            loaded = load_model(path)
        self.assertEqual(loaded["features"], ["x"])
        self.assertAlmostEqual(loaded["bias"], model["bias"])

    def test_returns_none_when_the_file_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertIsNone(load_model(path))

    def test_returns_none_on_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json")
            self.assertIsNone(load_model(path))


if __name__ == "__main__":
    unittest.main()
