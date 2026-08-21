"""
Follows up on simulated buys to see what actually happened to the token's
price afterward, so there's real signal to learn from instead of just a log
of what the bot would have bought. Records a percentage change (vs. the
price_ref proxy at entry) at a few checkpoints after each tracked buy, into
the same activity_log.jsonl the rest of the bot writes to.

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
from .price_ref import extract_price_ref
from .pumpportal_client import authenticated_ws_url
from .risk import RiskManager

logger = logging.getLogger("pumpfun_bot.outcome_tracker")

CHECKPOINTS_SEC = (60, 300, 900)
POLL_WINDOW_SEC = 20
IDLE_SLEEP_SEC = 5


def is_funded_key_rejection(message: str) -> bool:
    """PumpPortal replies with a bare {"message": ...} for both subscribe
    confirmations ("Successfully subscribed...") and the funded-API-key
    rejection - only the latter should be surfaced as a warning."""
    return "funded" in message.lower()


class OutcomeTracker:
    def __init__(self, ws_url: str, api_key: str = "", risk: RiskManager | None = None):
        self.ws_url = ws_url
        self.api_key = api_key
        # closes the simulated position (feeds P&L back into the risk
        # manager/dashboard) once a tracked mint reaches its final checkpoint
        # with real measured data - optional so tests/simple usage don't need one
        self.risk = risk
        self._pending: dict[str, dict] = {}
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
                mints = list(self._pending.keys())

            if not mints:
                await asyncio.sleep(IDLE_SLEEP_SEC)
                continue

            await self._poll_once(mints)
            await self._emit_due_checkpoints()
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
                            logger.warning(
                                "PumpPortal wees subscribeTokenTrade af: %s "
                                "-> outcome-tracking levert geen echte data zonder "
                                "een funded PUMPPORTAL_API_KEY.",
                                message,
                            )
                        continue

                    ref = extract_price_ref(data)
                    if ref is not None:
                        async with self._lock:
                            if mint in self._pending:
                                self._pending[mint]["last_ref"] = ref
                                self._pending[mint]["has_real_update"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Outcome tracker WS fout (probeer volgende ronde opnieuw): %s", exc)

    async def _emit_due_checkpoints(self) -> None:
        now = time.time()
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
                        # funded API key) - record as unmeasured, not "0% change"
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

                    # at the final checkpoint, close the simulated position so
                    # the result shows up in the dashboard's P&L/exposure -
                    # only when we actually measured something real; an
                    # unmeasured mint is left open rather than faking a result
                    if (
                        cp == CHECKPOINTS_SEC[-1]
                        and info["has_real_update"]
                        and self.risk is not None
                        and info["trade_size_sol"]
                    ):
                        pnl_sol = round(info["trade_size_sol"] * (pct_change / 100), 6)
                        self.risk.register_trade_closed(info["trade_size_sol"], pnl_sol)
                if len(info["hit"]) == len(CHECKPOINTS_SEC):
                    finished_mints.append(mint)
            for mint in finished_mints:
                del self._pending[mint]
