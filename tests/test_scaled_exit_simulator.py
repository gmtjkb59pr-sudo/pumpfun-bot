import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.activity_log as activity_log
from pumpfun_bot.scaled_exit_simulator import ScaledExitSimulator

_ORIGINAL_DATA_LOG_PATH = activity_log.DATA_LOG_PATH


class ScaledExitSimulatorTestCase(unittest.TestCase):
    """Base class: redirects activity_log.DATA_LOG_PATH to a temp file so
    these tests never write into the real, live activity_log.jsonl."""

    def setUp(self):
        self._log_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self._log_file.close()
        activity_log.DATA_LOG_PATH = Path(self._log_file.name)

    def tearDown(self):
        activity_log.DATA_LOG_PATH = _ORIGINAL_DATA_LOG_PATH
        Path(self._log_file.name).unlink(missing_ok=True)

    def _read_logged_records(self):
        with open(activity_log.DATA_LOG_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TrackAndUpdateTests(ScaledExitSimulatorTestCase):
    def test_ignores_a_price_update_for_an_untracked_mint(self):
        sim = ScaledExitSimulator()
        asyncio.run(sim.on_price_update("UNTRACKED", 100.0))
        self.assertEqual(self._read_logged_records(), [])

    def test_stop_loss_resolves_before_any_partial_take(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)

        asyncio.run(sim.on_price_update("MINT", 49.0))  # -51%, past the -50% stop

        records = self._read_logged_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "stop_loss")
        self.assertFalse(records[0]["partial_taken"])
        self.assertAlmostEqual(records[0]["blended_pct_change"], -51.0)
        self.assertNotIn("MINT", sim._positions)

    def test_does_not_take_partial_below_the_100pct_threshold(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)

        asyncio.run(sim.on_price_update("MINT", 150.0))  # only +50%

        self.assertEqual(self._read_logged_records(), [])
        self.assertFalse(sim._positions["MINT"]["partial_taken"])

    def test_takes_partial_at_the_100pct_threshold_without_resolving(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)

        asyncio.run(sim.on_price_update("MINT", 200.0))  # +100%

        self.assertEqual(self._read_logged_records(), [])  # not resolved yet
        pos = sim._positions["MINT"]
        self.assertTrue(pos["partial_taken"])
        self.assertAlmostEqual(pos["partial_pct_change"], 100.0)
        self.assertAlmostEqual(pos["peak_ref"], 200.0)  # remainder trails from here

    def test_remainder_trails_and_resolves_on_a_25pct_drawdown_from_its_own_peak(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)
        asyncio.run(sim.on_price_update("MINT", 200.0))  # +100%, partial taken, peak=200
        asyncio.run(sim.on_price_update("MINT", 300.0))  # keeps climbing, peak=300

        asyncio.run(sim.on_price_update("MINT", 224.0))  # 300 -> 224 is -25.33%, past the 25% trail

        records = self._read_logged_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "partial_then_trailing_stop")
        self.assertTrue(records[0]["partial_taken"])
        self.assertAlmostEqual(records[0]["partial_pct_change"], 100.0)
        # remainder pct_change measured from ORIGINAL entry (100), not from
        # the post-partial peak
        remainder_pct = ((224.0 - 100.0) / 100.0) * 100
        self.assertAlmostEqual(records[0]["remainder_pct_change"], remainder_pct)
        expected_blended = 0.5 * 100.0 + 0.5 * remainder_pct
        self.assertAlmostEqual(records[0]["blended_pct_change"], expected_blended)

    def test_does_not_resolve_the_remainder_on_a_small_pullback(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)
        asyncio.run(sim.on_price_update("MINT", 200.0))

        asyncio.run(sim.on_price_update("MINT", 190.0))  # only a 5% pullback from peak

        self.assertEqual(self._read_logged_records(), [])


class CheckTimeoutsTests(ScaledExitSimulatorTestCase):
    def test_resolves_with_timeout_after_max_hold_sec(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)
        sim._positions["MINT"]["start_ts"] = time.time() - 99999
        sim._positions["MINT"]["last_ref"] = 130.0

        asyncio.run(sim.check_timeouts())

        records = self._read_logged_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "timeout")
        self.assertAlmostEqual(records[0]["blended_pct_change"], 30.0)

    def test_resolves_with_stale_price_after_no_updates(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)
        sim._positions["MINT"]["last_update_ts"] = time.time() - 9999
        sim._positions["MINT"]["last_ref"] = 80.0

        asyncio.run(sim.check_timeouts())

        records = self._read_logged_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "stale_price")
        self.assertAlmostEqual(records[0]["blended_pct_change"], -20.0)

    def test_does_not_resolve_a_fresh_position(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)

        asyncio.run(sim.check_timeouts())

        self.assertEqual(self._read_logged_records(), [])
        self.assertIn("MINT", sim._positions)

    def test_timeout_after_a_partial_take_blends_correctly(self):
        sim = ScaledExitSimulator()
        sim.track("MINT", entry_ref=100.0, trade_size_sol=0.03)
        asyncio.run(sim.on_price_update("MINT", 200.0))  # +100% partial take
        sim._positions["MINT"]["start_ts"] = time.time() - 99999
        sim._positions["MINT"]["last_ref"] = 250.0  # +150% at time of timeout

        asyncio.run(sim.check_timeouts())

        records = self._read_logged_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "partial_then_timeout")
        expected_blended = 0.5 * 100.0 + 0.5 * 150.0
        self.assertAlmostEqual(records[0]["blended_pct_change"], expected_blended)


if __name__ == "__main__":
    unittest.main()
