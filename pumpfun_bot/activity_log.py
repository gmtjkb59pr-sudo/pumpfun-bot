"""Append-only JSONL log of trades/alerts/outcomes, kept on disk (unlike the
capped in-memory deques in state.py) so nothing is lost across restarts and
there's a full history available to analyze later."""
from __future__ import annotations

import json
from pathlib import Path

DATA_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "activity_log.jsonl"


def append_jsonl(record: dict) -> None:
    DATA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
