import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.model_retrain import retrain_loop


class FakeAlerter:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


def _drive(coro, timeout=0.05):
    async def _run():
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pass  # expected - the loop runs forever until the timeout cuts it off

    asyncio.run(_run())


def _patch_social_watch_model_as_never_enough_data():
    """Every test in this file is about sniper_model's OWN retrain
    behavior (or, in MultiModelIndependenceTests, explicitly about both) -
    this keeps social_watch_model's real functions from ever running
    unmocked against real files/state during a sniper-focused test.
    Real bug this guards against: without mocking BOTH models, a sniper-
    focused test would silently also retrain the REAL social_watch_model
    against whatever's actually on disk, a genuine test-isolation leak
    confirmed live when this file's own social_watch_model support was
    first added 2026-08-24."""
    return patch(
        "pumpfun_bot.model_retrain.social_watch_model.load_labeled_dataset",
        return_value=([], []),
    )


class RetrainLoopTests(unittest.TestCase):
    """User-requested 2026-08-24: "build an AI in the bot so it learns more
    automatically" - periodically retrains sniper_model.py from real trade
    history without a manual script run. Same conservative philosophy as
    auto_tuner.py: only ever retrains on MORE real data than last time, and
    only alerts on something actually worth surfacing (a first model, or
    its beats-baseline status changing), not every quiet tick."""

    def test_skips_below_min_training_rows_never_calls_retrain(self):
        alerter = FakeAlerter()
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset",
            return_value=([[1.0]] * 5, [0] * 5),
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save",
        ) as mock_retrain:
            _drive(retrain_loop(alerter, interval_sec=0))
        mock_retrain.assert_not_called()
        self.assertEqual(alerter.messages, [])

    def test_skips_a_retrain_when_the_row_count_has_not_grown(self):
        alerter = FakeAlerter()
        fake_result = {
            "n": 40, "train_n": 32, "holdout_n": 8,
            "accuracy": 0.7, "baseline": 0.6, "beats_baseline": True,
        }
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset",
            return_value=([[1.0]] * 40, [0] * 40),
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", return_value=fake_result,
        ) as mock_retrain:
            _drive(retrain_loop(alerter, interval_sec=0))
        # the dataset never changes size across iterations, so after the
        # first successful retrain every later tick must skip before ever
        # calling retrain_and_save again
        self.assertEqual(mock_retrain.call_count, 1)
        self.assertEqual(len(alerter.messages), 1)

    def test_alerts_on_the_first_successful_model(self):
        alerter = FakeAlerter()
        fake_result = {
            "n": 40, "train_n": 32, "holdout_n": 8,
            "accuracy": 0.7, "baseline": 0.6, "beats_baseline": True,
        }
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset",
            return_value=([[1.0]] * 40, [0] * 40),
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", return_value=fake_result,
        ):
            _drive(retrain_loop(alerter, interval_sec=0))
        self.assertEqual(len(alerter.messages), 1)
        self.assertIn("hertraind", alerter.messages[0])
        self.assertIn("70.0%", alerter.messages[0])

    def test_stays_quiet_on_a_later_retrain_with_the_same_beats_baseline_status(self):
        alerter = FakeAlerter()
        call_count = {"n": 0}

        def fake_dataset(corrections=None):
            call_count["n"] += 1
            size = 40 if call_count["n"] == 1 else 41
            return ([[1.0]] * size, [0] * size)

        results = iter([
            {"n": 40, "train_n": 32, "holdout_n": 8, "accuracy": 0.70, "baseline": 0.6, "beats_baseline": True},
            {"n": 41, "train_n": 33, "holdout_n": 8, "accuracy": 0.72, "baseline": 0.6, "beats_baseline": True},
        ])
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset", side_effect=fake_dataset,
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", side_effect=lambda: next(results),
        ):
            _drive(retrain_loop(alerter, interval_sec=0))
        # first model alerts; the second retrain still beats baseline (no
        # status change) so it logs quietly, not a second chat message
        self.assertEqual(len(alerter.messages), 1)

    def test_alerts_again_when_the_beats_baseline_status_flips(self):
        alerter = FakeAlerter()
        call_count = {"n": 0}

        def fake_dataset(corrections=None):
            call_count["n"] += 1
            size = 40 if call_count["n"] == 1 else 41
            return ([[1.0]] * size, [0] * size)

        results = iter([
            {"n": 40, "train_n": 32, "holdout_n": 8, "accuracy": 0.70, "baseline": 0.6, "beats_baseline": True},
            {"n": 41, "train_n": 33, "holdout_n": 8, "accuracy": 0.55, "baseline": 0.6, "beats_baseline": False},
        ])
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset", side_effect=fake_dataset,
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", side_effect=lambda: next(results),
        ):
            _drive(retrain_loop(alerter, interval_sec=0))
        self.assertEqual(len(alerter.messages), 2)
        self.assertIn("NIET beter dan", alerter.messages[1])

    def test_a_missing_activity_log_is_skipped_not_raised(self):
        alerter = FakeAlerter()
        with _patch_social_watch_model_as_never_enough_data(), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset",
            side_effect=FileNotFoundError,
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save",
        ) as mock_retrain:
            _drive(retrain_loop(alerter, interval_sec=0))  # must not raise
        mock_retrain.assert_not_called()
        self.assertEqual(alerter.messages, [])


class MultiModelIndependenceTests(unittest.TestCase):
    """User-requested 2026-08-24 ("build" a win-probability model for
    social_watch): each model retrains on its OWN schedule, independent
    growth, independent beats_baseline history - one model having new data
    (or flipping status) must never affect the other's state or alerts."""

    def test_both_models_retrain_and_alert_independently_on_their_first_success(self):
        alerter = FakeAlerter()
        sniper_result = {
            "n": 40, "train_n": 32, "holdout_n": 8,
            "accuracy": 0.70, "baseline": 0.60, "beats_baseline": True,
        }
        social_watch_result = {
            "n": 38, "train_n": 31, "holdout_n": 7,
            "accuracy": 0.60, "baseline": 0.70, "beats_baseline": False,
        }
        with patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset",
            return_value=([[1.0]] * 40, [0] * 40),
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", return_value=sniper_result,
        ), patch(
            "pumpfun_bot.model_retrain.social_watch_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.social_watch_model.load_labeled_dataset",
            return_value=([[1.0]] * 38, [0] * 38),
        ), patch(
            "pumpfun_bot.model_retrain.social_watch_model.retrain_and_save",
            return_value=social_watch_result,
        ):
            _drive(retrain_loop(alerter, interval_sec=0))
        self.assertEqual(len(alerter.messages), 2)
        sniper_msgs = [m for m in alerter.messages if "Sniper-model" in m]
        social_msgs = [m for m in alerter.messages if "Social-watch-model" in m]
        self.assertEqual(len(sniper_msgs), 1)
        self.assertEqual(len(social_msgs), 1)
        self.assertIn("beter dan", sniper_msgs[0])
        self.assertIn("NIET beter dan", social_msgs[0])

    def test_one_models_growth_does_not_trigger_a_retrain_of_the_other(self):
        alerter = FakeAlerter()
        call_count = {"n": 0}

        def growing_sniper_dataset(corrections=None):
            call_count["n"] += 1
            size = 40 if call_count["n"] == 1 else 41
            return ([[1.0]] * size, [0] * size)

        sniper_results = iter([
            {"n": 40, "train_n": 32, "holdout_n": 8, "accuracy": 0.70, "baseline": 0.6, "beats_baseline": True},
            {"n": 41, "train_n": 33, "holdout_n": 8, "accuracy": 0.72, "baseline": 0.6, "beats_baseline": True},
        ])
        with patch(
            "pumpfun_bot.model_retrain.sniper_model.load_corrections", return_value={},
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.load_labeled_dataset", side_effect=growing_sniper_dataset,
        ), patch(
            "pumpfun_bot.model_retrain.sniper_model.retrain_and_save", side_effect=lambda: next(sniper_results),
        ), _patch_social_watch_model_as_never_enough_data() as social_dataset_mock:
            _drive(retrain_loop(alerter, interval_sec=0))
        # sniper retrained twice (grew 40 -> 41 across ticks), social_watch
        # never had enough rows - only sniper's own alert fires, and
        # social_watch's dataset loader was still checked every tick
        # (independent scheduling, not skipped entirely)
        self.assertTrue(social_dataset_mock.call_count >= 1)
        self.assertTrue(all("Sniper-model" in m for m in alerter.messages))


if __name__ == "__main__":
    unittest.main()
