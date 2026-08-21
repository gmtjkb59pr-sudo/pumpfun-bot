"""
Follows up on simulated buys with an actual exit strategy - take-profit,
stop-loss, or a timeout - instead of just watching price drift forever.
This is what makes the bot's simulated P&L mean something: measuring raw
buy-and-hold price movement doesn't tell you whether a strategy is
profitable, because real trading is entries AND exits.

Also still records informational price-drift checkpoints (60s/300s/900s)
for tokens that haven't exited yet, for the dashboard's Learning Stats
panel - these are separate from the exit events that actually close a
position and register P&L.

After a position exits, it isn't dropped - it's watched passively for
another 60s/300s/900s (now measured from the exit) so the log has real
data on whether holding longer would have beaten the exit strategy, not
just an assumption that 50%/25% are the right thresholds.

Runs as a single long-lived background task shared across all tracked mints,
rather than one connection per mint, to keep the number of open websockets
reasonable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .activity_log import append_jsonl
from .alerts import Alerter
from .price_ref import extract_price_ref
from .pumpportal_client import authenticated_ws_url
from .risk import RiskManager
from .state import bot_state

logger = logging.getLogger("pumpfun_bot.outcome_tracker")

CHECKPOINTS_SEC = (60, 300, 900)
MAX_HOLD_SEC = CHECKPOINTS_SEC[-1]
POLL_WINDOW_SEC = 20
IDLE_SLEEP_SEC = 5

EXIT_EMOJI = {"take_profit": "🟢", "stop_loss": "🔴", "timeout": "⏱️"}


def is_funded_key_rejection(message: str) -> bool:
    """PumpPortal replies with a bare {"message": ...} for both subscribe
    confirmations ("Successfully subscribed...") and the funded-API-key
    rejection - only the latter should be surfaced as a warning."""
    return "funded" in message.lower()


class OutcomeTracker:
    def __init__(
        self,
        ws_url: str,
        api_key: str = "",
        risk: RiskManager | None = None,
        alerter: Alerter | None = None,
        take_profit_pct: float = 50.0,
        stop_loss_pct: float = 25.0,
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        # closes the simulated position (feeds P&L back into the risk
        # manager/dashboard) on exit - optional so tests/simple usage don't
        # need one
        self.risk = risk
        self.alerter = alerter
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self._pending: dict[str, dict] = {}
        # mints that already exited - kept under passive observation (no
        # P&L/exposure effect, already realized) purely to answer "would
        # holding longer have done better than the exit strategy did?"
        self._post_exit: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._warned_no_access = False

    async def track(
        self, mint: str, name: str, symbol: str, entry_ref: float | None, trade_size_sol: float = 0.0
    ) -> None:
        if entry_ref is None:
            logger.debug("Geen price-ref beschikbaar voor %s, sla outcome-tracking over.", mint)
            return
        async with self._lock:
            self._pending[mint] = {
                "entry_ts": time.time(),
                "entry_ref": entry_ref,
                "last_ref": entry_ref,
                "name": name,
                "symbol": symbol,
                "trade_size_sol": trade_size_sol,
                "hit": set(),
                # only set once we've actually seen a trade event for this mint -
                # without this, a rejected/empty subscription would silently look
                # like "0% change" instead of "never measured"
                "has_real_update": False,
            }

    async def run(self) -> None:
        while True:
            async with self._lock:
                mints = list(self._pending.keys()) + list(self._post_exit.keys())

            if not mints:
                await asyncio.sleep(IDLE_SLEEP_SEC)
                continue

            await self._poll_once(mints)
            await self._emit_due_checkpoints()
            await self._emit_post_exit_checkpoints()
            await asyncio.sleep(IDLE_SLEEP_SEC)

    async def _poll_once(self, mints: list[str]) -> None:
        try:
            ws_url = authenticated_ws_url(self.ws_url, self.api_key)
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                await ws.send(json.dumps({"method": "subscribeTokenTrade", "keys": mints}))
                deadline = time.time() + POLL_WINDOW_SEC
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    mint = data.get("mint")
                    if mint is None:
                        message = data.get("message", "")
                        if not self._warned_no_access and is_funded_key_rejection(message):
                            self._warned_no_access = True
                            bot_state.set_outcome_tracking_rejected(True)
                            logger.warning(
                                "PumpPortal wees subscribeTokenTrade af: %s "
                                "-> outcome-tracking levert geen echte data zonder "
                                "een funded PUMPPORTAL_API_KEY.",
                                message,
                            )
                        continue

                    ref = extract_price_ref(data)
                    if ref is not None:
                        if self._warned_no_access:
                            # a real trade event means the feed is actually
                            # working now, despite an earlier rejection
                            self._warned_no_access = False
                            bot_state.set_outcome_tracking_rejected(False)
                        await self._handle_price_update(mint, ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Outcome tracker WS fout (probeer volgende ronde opnieuw): %s", exc)

    async def _handle_price_update(self, mint: str, ref: float) -> None:
        """Records the new price and exits the position immediately if it
        crosses take-profit or stop-loss. Separated from _poll_once so this
        decision logic can be exercised directly in tests."""
        exit_args = None
        async with self._lock:
            info = self._pending.get(mint)
            if info is not None:
                info["last_ref"] = ref
                info["has_real_update"] = True
                pct_change = ((ref - info["entry_ref"]) / info["entry_ref"]) * 100
                if pct_change >= self.take_profit_pct:
                    exit_args = (mint, dict(info), "take_profit", pct_change)
                elif pct_change <= -self.stop_loss_pct:
                    exit_args = (mint, dict(info), "stop_loss", pct_change)
                if exit_args:
                    del self._pending[mint]

            post = self._post_exit.get(mint)
            if post is not None:
                post["last_ref"] = ref
                post["has_real_update"] = True
        if exit_args:
            await self._exit(*exit_args)

    async def _exit(self, mint: str, info: dict, reason: str, pct_change: float) -> None:
        pct_change = round(pct_change, 2)
        if self.risk is not None and info["trade_size_sol"]:
            pnl_sol = round(info["trade_size_sol"] * (pct_change / 100), 6)
            self.risk.register_trade_closed(info["trade_size_sol"], pnl_sol)
        append_jsonl({
            "type": "exit",
            "ts": time.time(),
            "mint": mint,
            "name": info["name"],
            "symbol": info["symbol"],
            "reason": reason,
            "pct_change": pct_change,
            "trade_size_sol": info["trade_size_sol"],
        })
        if self.alerter is not None:
            emoji = EXIT_EMOJI.get(reason, "")
            await self.alerter.send(
                f"{emoji} Exit ({reason}): {info['name']} ({info['symbol']}) @ {pct_change:+.1f}%"
            )

        # keep passively watching after the exit - doesn't touch risk/P&L
        # again, purely so we can later tell whether holding past this exit
        # would actually have done better
        async with self._lock:
            self._post_exit[mint] = {
                "exit_ts": time.time(),
                "entry_ref": info["entry_ref"],
                "exit_ref": info["last_ref"],
                "realized_pct_change": pct_change,
                "reason": reason,
                "name": info["name"],
                "symbol": info["symbol"],
                "last_ref": info["last_ref"],
                "hit": set(),
                "has_real_update": False,
            }

    async def _emit_due_checkpoints(self) -> None:
        now = time.time()
        to_timeout_exit = []
        async with self._lock:
            finished_mints = []
            for mint, info in self._pending.items():
                age = now - info["entry_ts"]
                for cp in CHECKPOINTS_SEC:
                    if cp in info["hit"] or age < cp:
                        continue
                    if info["has_real_update"]:
                        pct_change = round(
                            ((info["last_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100, 2
                        )
                    else:
                        # never received a real trade event for this mint (most
                        # likely subscribeTokenTrade was rejected for lack of a
                        # funded API key, or the token just had zero trades) -
                        # record as unmeasured, not "0% change"
                        pct_change = None
                    append_jsonl({
                        "type": "outcome",
                        "ts": now,
                        "mint": mint,
                        "name": info["name"],
                        "symbol": info["symbol"],
                        "checkpoint_sec": cp,
                        "entry_ref": info["entry_ref"],
                        "ref_at_checkpoint": info["last_ref"],
                        "pct_change": pct_change,
                        "measured": info["has_real_update"],
                    })
                    info["hit"].add(cp)
                if age >= MAX_HOLD_SEC:
                    finished_mints.append(mint)
                    if info["has_real_update"]:
                        pct_change = round(
                            ((info["last_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100, 2
                        )
                        to_timeout_exit.append((mint, dict(info), "timeout", pct_change))
            for mint in finished_mints:
                del self._pending[mint]
        for mint, info, reason, pct_change in to_timeout_exit:
            await self._exit(mint, info, reason, pct_change)

    async def _emit_post_exit_checkpoints(self) -> None:
        """For mints that already exited, records - at the same 60/300/900s
        cadence, now measured from the exit instead of the entry - what the
        price actually did afterward, so we can tell whether the exit
        strategy (take-profit/stop-loss/timeout) is well-calibrated or
        whether holding longer would have realized more."""
        now = time.time()
        async with self._lock:
            finished_mints = []
            for mint, info in self._post_exit.items():
                age = now - info["exit_ts"]
                for cp in CHECKPOINTS_SEC:
                    if cp in info["hit"] or age < cp:
                        continue
                    if info["has_real_update"]:
                        pct_change_since_exit = round(
                            ((info["last_ref"] - info["exit_ref"]) / info["exit_ref"]) * 100, 2
                        )
                        pct_change_if_held_from_entry = round(
                            ((info["last_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100, 2
                        )
                        vs_realized_pct = round(
                            pct_change_if_held_from_entry - info["realized_pct_change"], 2
                        )
                    else:
                        pct_change_since_exit = None
                        pct_change_if_held_from_entry = None
                        vs_realized_pct = None
                    append_jsonl({
                        "type": "post_exit_check",
                        "ts": now,
                        "mint": mint,
                        "name": info["name"],
                        "symbol": info["symbol"],
                        "exit_reason": info["reason"],
                        "checkpoint_sec_after_exit": cp,
                        "realized_pct_change": info["realized_pct_change"],
                        "pct_change_since_exit": pct_change_since_exit,
                        "pct_change_if_held_from_entry": pct_change_if_held_from_entry,
                        # positive = holding past the exit would have made
                        # more than the exit strategy actually realized;
                        # negative = exiting when we did was the right call
                        "vs_realized_pct": vs_realized_pct,
                        "measured": info["has_real_update"],
                    })
                    info["hit"].add(cp)
                if age >= CHECKPOINTS_SEC[-1]:
                    finished_mints.append(mint)
            for mint in finished_mints:
                del self._post_exit[mint]
