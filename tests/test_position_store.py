import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pumpfun_bot.position_store as position_store
from pumpfun_bot.position_store import load, path_for_mode, save


class PositionStoreTests(unittest.TestCase):
    def _temp_path(self) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        f.close()
        path = Path(f.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_missing_file_returns_empty_dict(self):
        path = self._temp_path()
        path.unlink()  # doesn't exist at all
        self.assertEqual(load(path), {})

    def test_roundtrips_a_position_including_the_hit_set(self):
        path = self._temp_path()
        positions = {
            "MINT1": {
                "entry_ts": 1000.0,
                "entry_ref": 100.0,
                "last_ref": 120.0,
                "peak_ref": 120.0,
                "name": "Test Token",
                "symbol": "TEST",
                "trade_size_sol": 0.03,
                "hit": {60, 300},
                "has_real_update": True,
                "last_update_ts": 1010.0,
            }
        }
        save(positions, path)
        loaded = load(path)

        self.assertEqual(set(loaded.keys()), {"MINT1"})
        self.assertEqual(loaded["MINT1"]["entry_ref"], 100.0)
        self.assertEqual(loaded["MINT1"]["hit"], {60, 300})
        self.assertIsInstance(loaded["MINT1"]["hit"], set)

    def test_save_is_atomic_no_leftover_tmp_file_after_success(self):
        path = self._temp_path()
        save({"MINT1": {"hit": set()}}, path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.assertFalse(tmp_path.exists())

    def test_corrupt_file_returns_empty_dict_instead_of_raising(self):
        path = self._temp_path()
        path.write_text("not valid json{{{")
        self.assertEqual(load(path), {})


class PathForModeTests(unittest.TestCase):
    """Real bug this guards against: starting a live session loaded 23
    leftover positions from an earlier dry-run farming session (never real
    purchases) and tried to actually sell them - each attempt burned a real
    transaction fee failing on-chain, and the phantom positions occupied
    slots against max_open_positions, crowding out real trading. Dry-run
    and live must never be able to load each other's positions."""

    def test_dry_run_and_live_resolve_to_different_paths(self):
        self.assertNotEqual(path_for_mode(dry_run=True), path_for_mode(dry_run=False))

    def test_dry_run_path_is_stable_across_calls(self):
        self.assertEqual(path_for_mode(dry_run=True), path_for_mode(dry_run=True))

    def test_reads_default_store_path_fresh_not_bound_at_import(self):
        # setUpModule-style test redirection (patch DEFAULT_STORE_PATH, then
        # construct things that use it) must still separate live/dry-run
        original = position_store.DEFAULT_STORE_PATH
        try:
            position_store.DEFAULT_STORE_PATH = Path("/tmp/some/redirected/store.json")
            live_path = path_for_mode(dry_run=False)
            dry_run_path = path_for_mode(dry_run=True)
        finally:
            position_store.DEFAULT_STORE_PATH = original

        self.assertIn("redirected", str(live_path))
        self.assertNotEqual(live_path, dry_run_path)

    def test_a_live_session_never_loads_a_dry_run_session_s_positions(self):
        with patch.object(position_store, "DEFAULT_STORE_PATH", Path(tempfile.mktemp(suffix=".json"))):
            dry_run_path = path_for_mode(dry_run=True)
            live_path = path_for_mode(dry_run=False)
            self.addCleanup(lambda: dry_run_path.unlink(missing_ok=True))
            self.addCleanup(lambda: live_path.unlink(missing_ok=True))

            # a dry-run session persists a (simulated, never-bought) position
            save({"PHANTOM_MINT": {"hit": set()}}, dry_run_path)

            # a live session starting up must see nothing at its own path
            self.assertEqual(load(live_path), {})


if __name__ == "__main__":
    unittest.main()
