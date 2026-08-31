"""
Persists RiskManager counters (daily P&L, trades-per-hour timestamps, and
a snapshot of open exposure) so a restart does not silently reset the
circuit-breakers that exist specifically to stop a drain.

Positions already persist separately (position_store.py). Open exposure is
RECONSTRUCTED from those positions on startup (see
RiskManager.sync_open_exposure_from_positions) so it stays consistent with
wallet-reconciled holdings — remaining_fraction after a ladder rung, and
sell_paused+reputation_logged positions whose exposure was already released,
must not be counted twice. The exposure field in this file is only a
snapshot of last known state, overwritten by that reconstruction.

path_for_mode() gives dry-run and live sessions separate files, same reason
as position_store: a dry-run farming session's daily-loss must never block
a later live session (or vice versa).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_STORE_PATH = Path("data/risk_state.json")


def path_for_mode(dry_run: bool) -> Path:
    suffix = "dry_run" if dry_run else "live"
    base = DEFAULT_STORE_PATH
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def load(path: str | Path = DEFAULT_STORE_PATH) -> dict:
    path = Path(path)
    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def save(state: dict, path: str | Path = DEFAULT_STORE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)
