"""
Persists currently-open positions to disk so a restart can resume tracking
them instead of losing them from memory.

This was a real gap: OutcomeTracker._pending only ever lived in-process
memory. Every restart - and there were many today, each one shipping a fix -
silently abandoned whatever was open at that exact moment. Nothing sold them
wrong; nothing ever looked at them again, by design, because nothing knew
they existed anymore.

Written on every change to _pending (opened, updated, closed) so a kill at
any point leaves the file consistent with the last known state. Uses an
atomic write (temp file + rename) so a kill mid-write can't corrupt it.

path_for_mode() gives dry-run and live sessions separate files. Found live:
starting a real-money session loaded 23 leftover positions from an earlier
dry-run farming session (never real purchases) and tried to actually sell
them - each one burned a real transaction fee failing on-chain (nothing was
ever bought), and worse, they occupied slots against max_open_positions,
crowding out real trading capacity with phantom positions. A single shared
file can't tell dry-run and live apart; two files can't cross-contaminate
by construction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_STORE_PATH = Path("data/open_positions.json")


def path_for_mode(dry_run: bool) -> Path:
    """dry-run and live sessions must never load each other's positions -
    a dry-run position is simulated (nothing was ever bought) and a live
    position is real money, so treating one as the other in either
    direction is wrong. Reads DEFAULT_STORE_PATH fresh (not as a bound
    default) so tests can redirect it and still get separated paths."""
    suffix = "dry_run" if dry_run else "live"
    base = DEFAULT_STORE_PATH
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def load(path: str | Path = DEFAULT_STORE_PATH) -> dict[str, dict]:
    path = Path(path)
    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    positions = {}
    for mint, info in raw.items():
        info = dict(info)
        info["hit"] = set(info.get("hit", []))
        positions[mint] = info
    return positions


def save(positions: dict[str, dict], path: str | Path = DEFAULT_STORE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for mint, info in positions.items():
        info = dict(info)
        info["hit"] = sorted(info.get("hit", set()))
        serializable[mint] = info

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(serializable, f)
    os.replace(tmp_path, path)
