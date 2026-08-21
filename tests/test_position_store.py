import tempfile
import unittest
from pathlib import Path

from pumpfun_bot.position_store import load, save


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


if __name__ == "__main__":
    unittest.main()
