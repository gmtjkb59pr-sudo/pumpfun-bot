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
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_STORE_PATH = Path("data/open_positions.json")


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
