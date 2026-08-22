"""
Simulates a "scaled exit" strategy in parallel with the real exit logic in
outcome_tracker.py, on the exact same live price ticks for the exact same
real positions - a true apples-to-apples comparison, not a different subset
of coins the way running a second bot instance would produce (same
reasoning as the momentum-window comparison logged from social_watch.py).

User-requested after the current all-or-nothing exit produced far more
stop-losses than take-profits: instead of closing the whole position at a
single take-profit level, take part of it off the table at
PARTIAL_TAKE_PCT (locking in real profit), then let the remainder ride
with its own trailing stop from whatever peak it reaches afterward - only
the stop-loss protecting the position BEFORE that partial take is much
wider (STOP_LOSS_PCT) than the live strategy's, since holding longer for a
bigger move is the whole point of this variant.

Purely observational - never places a real (or dry-run) trade, never
touches the actual open position or its exit decision. Logs a
"scaled_exit_counterfactual" record once a simulated position resolves,
so it can be compared against the real "exit" records for the same mints
later.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .activity_log import append_jsonl

logger = logging.getLogger("pumpfun_bot.scaled_exit_simulator")

# user-requested exact values, see the AskUserQuestion exchange this was
# built from: take half off at +100%, trail the remainder 25% off its
# peak, and only stop the whole thing out (before any partial take) at -50%
PARTIAL_TAKE_PCT = 100.0
PARTIAL_TAKE_FRACTION = 0.5
TRAIL_PCT_AFTER_PARTIAL = 25.0
STOP_LOSS_PCT = 50.0

# matches outcome_tracker.py's MAX_HOLD_SEC/STALE_PRICE_TIMEOUT_SEC - same
# real-world reasoning (don't hold a dead/rugged token forever waiting for
# a price tick that will never come)
MAX_HOLD_SEC = 900
STALE_PRICE_TIMEOUT_SEC = 10
TIMEOUT_CHECK_INTERVAL_SEC = 5


class ScaledExitSimulator:
    def __init__(self):
        self._positions: dict[str, dict] = {}

    def track(self, mint: str, entry_ref: float, trade_size_sol: float) -> None:
        now = time.time()
        self._positions[mint] = {
            "entry_ref": entry_ref,
            "trade_size_sol": trade_size_sol,
            "peak_ref": entry_ref,
            "last_ref": entry_ref,
            "partial_taken": False,
            "partial_pct_change": None,
            "start_ts": now,
            "last_update_ts": now,
        }

    async def on_price_update(self, mint: str, ref: float) -> None:
        pos = self._positions.get(mint)
        if pos is None:
            return
        pos["last_ref"] = ref
        pos["last_update_ts"] = time.time()
        if ref > pos["peak_ref"]:
            pos["peak_ref"] = ref

        entry_ref = pos["entry_ref"]
        pct_change = ((ref - entry_ref) / entry_ref) * 100

        if not pos["partial_taken"]:
            if pct_change <= -STOP_LOSS_PCT:
                self._resolve(mint, pos, "stop_loss", pct_change)
                return
            if pct_change >= PARTIAL_TAKE_PCT:
                pos["partial_taken"] = True
                pos["partial_pct_change"] = pct_change
                pos["peak_ref"] = ref  # remainder trails from here, not from the original entry
                logger.debug(
                    "Scaled-exit sim: %s nam %.0f%% initials bij +%.1f%%.",
                    mint, PARTIAL_TAKE_FRACTION * 100, pct_change,
                )
            return

        drawdown_from_peak_pct = ((ref - pos["peak_ref"]) / pos["peak_ref"]) * 100
        if drawdown_from_peak_pct <= -TRAIL_PCT_AFTER_PARTIAL:
            self._resolve(mint, pos, "trailing_stop", pct_change)

    async def check_timeouts(self) -> None:
        now = time.time()
        for mint, pos in list(self._positions.items()):
            age = now - pos["start_ts"]
            stale = now - pos["last_update_ts"] >= STALE_PRICE_TIMEOUT_SEC
            if age >= MAX_HOLD_SEC:
                entry_ref = pos["entry_ref"]
                pct_change = ((pos["last_ref"] - entry_ref) / entry_ref) * 100
                self._resolve(mint, pos, "timeout", pct_change)
            elif stale:
                entry_ref = pos["entry_ref"]
                pct_change = ((pos["last_ref"] - entry_ref) / entry_ref) * 100
                self._resolve(mint, pos, "stale_price", pct_change)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(TIMEOUT_CHECK_INTERVAL_SEC)
            await self.check_timeouts()

    def _resolve(self, mint: str, pos: dict, remainder_reason: str, remainder_pct_change: float) -> None:
        if pos["partial_taken"]:
            blended_pct_change = (
                PARTIAL_TAKE_FRACTION * pos["partial_pct_change"]
                + (1 - PARTIAL_TAKE_FRACTION) * remainder_pct_change
            )
            reason = f"partial_then_{remainder_reason}"
        else:
            blended_pct_change = remainder_pct_change
            reason = remainder_reason
        append_jsonl({
            "type": "scaled_exit_counterfactual",
            "ts": time.time(),
            "mint": mint,
            "reason": reason,
            "partial_taken": pos["partial_taken"],
            "partial_pct_change": pos["partial_pct_change"],
            "remainder_pct_change": remainder_pct_change,
            "blended_pct_change": blended_pct_change,
            "trade_size_sol": pos["trade_size_sol"],
        })
        del self._positions[mint]
