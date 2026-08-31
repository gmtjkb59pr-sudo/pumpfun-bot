import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.risk_store as risk_store
from pumpfun_bot.config import RiskConfig
from pumpfun_bot.risk import RiskManager
from pumpfun_bot.risk_store import load, path_for_mode, save


class PathForModeTests(unittest.TestCase):
    def test_dry_run_and_live_resolve_to_different_paths(self):
        self.assertNotEqual(path_for_mode(dry_run=True), path_for_mode(dry_run=False))

    def test_reads_default_store_path_fresh(self):
        original = risk_store.DEFAULT_STORE_PATH
        try:
            risk_store.DEFAULT_STORE_PATH = Path("/tmp/redirected/risk.json")
            live_path = path_for_mode(dry_run=False)
            dry_path = path_for_mode(dry_run=True)
        finally:
            risk_store.DEFAULT_STORE_PATH = original
        self.assertIn("redirected", str(live_path))
        self.assertNotEqual(live_path, dry_path)


class RiskStoreRoundtripTests(unittest.TestCase):
    def _temp_path(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        path = Path(f.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_missing_file_returns_empty_dict(self):
        path = self._temp_path()
        path.unlink()
        self.assertEqual(load(path), {})

    def test_corrupt_file_returns_empty_dict(self):
        path = self._temp_path()
        path.write_text("not json{{{")
        self.assertEqual(load(path), {})

    def test_roundtrip(self):
        path = self._temp_path()
        save({"realized_pnl_sol": -0.12, "trade_timestamps": [1.0, 2.0]}, path)
        loaded = load(path)
        self.assertEqual(loaded["realized_pnl_sol"], -0.12)
        self.assertEqual(loaded["trade_timestamps"], [1.0, 2.0])


class RiskManagerPersistenceTests(unittest.TestCase):
    def _path(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        path = Path(f.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_without_store_path_does_not_write_a_file(self):
        # existing tests construct RiskManager() with no path — must stay
        # in-memory only so they don't leak into data/
        risk = RiskManager(RiskConfig(dry_run=False))
        risk.register_trade_opened(0.05)
        self.assertIsNone(risk.store_path)

    def test_daily_loss_and_hourly_trades_survive_a_restart(self):
        path = self._path()
        cfg = RiskConfig(dry_run=False, max_daily_loss_sol=0.2, max_trades_per_hour=10)
        risk = RiskManager(cfg, store_path=path)
        risk.register_trade_opened(0.05)
        risk.register_trade_closed(0.05, pnl_sol=-0.08)
        risk.register_trade_opened(0.03)

        restarted = RiskManager(cfg, store_path=path)
        self.assertAlmostEqual(restarted.state.realized_pnl_sol, -0.08)
        self.assertEqual(len(restarted.state.trade_timestamps), 2)
        ok, reason = restarted.can_trade(0.01)
        self.assertTrue(ok, reason)

    def test_daily_loss_limit_still_blocks_after_restart(self):
        path = self._path()
        cfg = RiskConfig(dry_run=False, max_daily_loss_sol=0.1)
        risk = RiskManager(cfg, store_path=path)
        risk.register_trade_closed(0.05, pnl_sol=-0.1)

        restarted = RiskManager(cfg, store_path=path)
        ok, reason = restarted.can_trade(0.01)
        self.assertFalse(ok)
        self.assertIn("verlieslimiet", reason)

    def test_hourly_trade_cap_still_blocks_after_restart(self):
        path = self._path()
        cfg = RiskConfig(dry_run=False, max_trades_per_hour=2)
        risk = RiskManager(cfg, store_path=path)
        risk.register_trade_opened(0.01)
        risk.register_trade_opened(0.01)

        restarted = RiskManager(cfg, store_path=path)
        ok, reason = restarted.can_trade(0.01)
        self.assertFalse(ok)
        self.assertIn("trades/uur", reason)

    def test_stale_hourly_timestamps_are_pruned_on_load(self):
        path = self._path()
        cfg = RiskConfig(dry_run=False, max_trades_per_hour=2)
        risk = RiskManager(cfg, store_path=path)
        risk.state.trade_timestamps = [time.time() - 7200, time.time() - 7200]
        risk._persist()

        restarted = RiskManager(cfg, store_path=path)
        ok, _ = restarted.can_trade(0.01)
        self.assertTrue(ok)
        self.assertEqual(restarted._trades_in_last_hour(), 0)

    def test_day_rollover_resets_pnl_on_load(self):
        path = self._path()
        cfg = RiskConfig(dry_run=False, max_daily_loss_sol=0.1)
        risk = RiskManager(cfg, store_path=path)
        risk.state.realized_pnl_sol = -0.1
        risk.state.day_start_ts = time.time() - 90000
        risk._persist()

        restarted = RiskManager(cfg, store_path=path)
        self.assertAlmostEqual(restarted.state.realized_pnl_sol, 0.0)
        ok, _ = restarted.can_trade(0.01)
        self.assertTrue(ok)

    def test_exposure_is_reconstructed_from_positions_not_the_raw_snapshot(self):
        path = self._path()
        cfg = RiskConfig(
            dry_run=False, max_sol_total_exposure=0.1, max_sol_per_trade=0.5,
        )
        risk = RiskManager(cfg, store_path=path)
        risk.state.open_exposure_sol = 9.99  # stale snapshot
        risk._persist()

        restarted = RiskManager(cfg, store_path=path)
        restarted.sync_open_exposure_from_positions({
            "M1": {"trade_size_sol": 0.05, "remaining_fraction": 0.7},
            "M2": {"trade_size_sol": 0.02, "remaining_fraction": 1.0},
            # exposure already released for this stuck position
            "M3": {
                "trade_size_sol": 0.5, "remaining_fraction": 1.0,
                "sell_paused": True, "reputation_logged": True,
            },
        })
        self.assertAlmostEqual(restarted.state.open_exposure_sol, 0.05 * 0.7 + 0.02)
        # 0.055 + 0.05 = 0.105 > 0.1 cap
        ok, reason = restarted.can_trade(0.05)
        self.assertFalse(ok)
        self.assertIn("totale exposure", reason)

    def test_dry_run_and_live_files_do_not_cross_contaminate(self):
        with patch.object(
            risk_store, "DEFAULT_STORE_PATH", Path(tempfile.mktemp(suffix=".json")),
        ):
            dry_path = path_for_mode(True)
            live_path = path_for_mode(False)
            self.addCleanup(lambda: dry_path.unlink(missing_ok=True))
            self.addCleanup(lambda: live_path.unlink(missing_ok=True))
            dry = RiskManager(RiskConfig(dry_run=True), store_path=dry_path)
            dry.register_trade_closed(0.05, pnl_sol=-0.2)
            live = RiskManager(RiskConfig(dry_run=False), store_path=live_path)
            self.assertAlmostEqual(live.state.realized_pnl_sol, 0.0)


