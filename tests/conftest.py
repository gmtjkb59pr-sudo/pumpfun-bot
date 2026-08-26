"""Shared pytest fixtures for the whole test suite.

Real bug found live 2026-08-26: append_jsonl() (pumpfun_bot/activity_log.py)
- and everything built on it: bot_state.log_trade()/log_alert(),
AutoTuner._apply(), OutcomeTracker's exit logging, etc. - always writes to
the real, module-level DATA_LOG_PATH, with no per-call override. Any test
that exercises one of those code paths without individually patching
DATA_LOG_PATH first silently wrote a real record into the ACTUAL
production data/activity_log.jsonl.

Confirmed live: this had been happening for a long time, on a large scale.
tests/test_auto_tuner.py alone had appended 194 fake "reason: test"
autotune_change records into the real file (found and fixed separately,
see that file's ApplyTests docstring) - and while investigating an
unrelated real-money question, obviously-fake placeholder trade records
(mint="MINT", creator="CREATOR", tx_signature="sig") turned up too, from
tests/test_sniper.py exercising real buy/exit code paths with no isolation
at all.

This single autouse fixture redirects EVERY test's activity-log writes to
a fresh per-test temp file instead, with no per-test-file changes needed
anywhere else - the entire class of bug (past, present, and future test
files) is closed in one place. Individual tests that already patch
DATA_LOG_PATH themselves to their own specific temp path (e.g.
test_activity_log.py) are unaffected - patches compose, theirs just wins
for the duration of their own `with` block.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import pumpfun_bot.activity_log as activity_log


@pytest.fixture(autouse=True)
def _isolate_activity_log(tmp_path):
    fake_path = tmp_path / "activity_log.jsonl"
    with patch.object(activity_log, "DATA_LOG_PATH", fake_path):
        yield fake_path
