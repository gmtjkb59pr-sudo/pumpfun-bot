"""
Buffers real, live trade-tick price history for tokens social_watch is
currently evaluating (candidates that haven't been bought yet), so buy-time
momentum can be measured over genuinely short windows (1m/2m) that
DexScreener's public API doesn't expose (see dexscreener.py - only
m5/h1/h6/h24 available there, and those turned out to be identical for
tokens this young anyway, since there's no earlier price to diff against).

Mirrors OutcomeTracker's persistent-connection, incrementally-resubscribed
WS pattern (see outcome_tracker.py) rather than opening one connection per
candidate - a single shared connection keeps the number of open websockets
bounded regardless of how many tokens are being watched at once.

User-requested despite a real, structural cost: PumpPortal meters
subscribeTokenTrade at 0.01 SOL per 10,000 events, there's no documented
unsubscribe method, and every candidate ever watched (most of which never
even pass the socials/holder/concentration filters) stays subscribed for
the life of the connection - the subscription list only grows. Explicitly
accepted for now to get real data ("deal with the costs later"), not
because the cost is negligible - worth revisiting if this runs for a long
session.

A candidate must be watch()ed before a real answer is possible -
price_change_pct() returns None if there isn't yet a buffered tick old
enough to cover the requested window, exactly like every other
"unknown, don't guess" filter in this codebase (never returns "0%" for
missing data).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

from .price_ref import extract_price_ref
from .pumpportal_client import authenticated_ws_url

logger = logging.getLogger("pumpfun_bot.candidate_price_tracker")

HOUSEKEEPING_INTERVAL_SEC = 3
RECONNECT_BACKOFF_SEC = 3
# how long to keep buffered ticks around after a mint is unwatch()ed -
# covers the gap between a candidate clearing its filters and _buy()
# actually reading the buffer, without holding history forever
BUFFER_RETENTION_SEC = 150


class CandidatePriceTracker:
    def __init__(self, ws_url: str, api_key: str = ""):
        self.ws_url = ws_url
        self.api_key = api_key
        self._history: dict[str, list[tuple[float, float]]] = {}
        self._watched: set[str] = set()
        self._subscribed_mints: set[str] = set()
        self._lock = asyncio.Lock()

    async def watch(self, mint: str) -> None:
        async with self._lock:
            self._watched.add(mint)
            self._history.setdefault(mint, [])

    async def unwatch(self, mint: str) -> None:
        # deliberately does NOT clear history here - a candidate can be
        # bought right after its last poll cycle confirms socials, and
        # _buy() still needs the buffered ticks at that moment. Pruned
        # later by _prune_old_history() once BUFFER_RETENTION_SEC passes.
        async with self._lock:
            self._watched.discard(mint)

    def price_change_pct(self, mint: str, window_sec: float) -> float | None:
        """Returns the % price change from the oldest tick at least
        window_sec old to the latest tick, or None if there isn't a real
        tick that old yet (never guesses at a shorter window)."""
        history = self._history.get(mint)
        if not history:
            return None
        latest_ts, latest_price = history[-1]
        oldest_needed_ts = latest_ts - window_sec
        base_price = None
        for ts, price in history:
            if ts <= oldest_needed_ts:
                base_price = price
            else:
                break
        if base_price is None or base_price <= 0:
            return None
        return ((latest_price - base_price) / base_price) * 100

    async def run(self) -> None:
        while True:
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # mirrors outcome_tracker.run() - a bad connection cycle here
                # must never crash the whole bot's asyncio.gather
                logger.exception(
                    "Onverwachte fout in candidate-price-tracker verbinding - opnieuw verbinden."
                )
            await asyncio.sleep(RECONNECT_BACKOFF_SEC)

    async def _run_connection(self) -> None:
        self._subscribed_mints = set()
        ws_url = authenticated_ws_url(self.ws_url, self.api_key)
        async with websockets.connect(ws_url, ping_interval=20) as ws:
            await self._sync_subscription(ws)
            housekeeping = asyncio.create_task(self._housekeeping_loop(ws))
            try:
                async for raw in ws:
                    await self._handle_ws_message(raw)
            finally:
                housekeeping.cancel()
                try:
                    await housekeeping
                except asyncio.CancelledError:
                    pass

    async def _housekeeping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(HOUSEKEEPING_INTERVAL_SEC)
            await self._sync_subscription(ws)
            self._prune_old_history()

    async def _sync_subscription(self, ws) -> None:
        async with self._lock:
            current = set(self._watched)
        new_mints = current - self._subscribed_mints
        if new_mints:
            await ws.send(json.dumps({"method": "subscribeTokenTrade", "keys": sorted(new_mints)}))
            self._subscribed_mints |= new_mints

    async def _handle_ws_message(self, raw) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        mint = data.get("mint")
        if mint is None:
            return
        ref = extract_price_ref(data)
        if ref is None:
            return
        async with self._lock:
            if mint not in self._history:
                return
            self._history[mint].append((time.time(), ref))

    def _prune_old_history(self) -> None:
        cutoff = time.time() - BUFFER_RETENTION_SEC
        for mint, ticks in list(self._history.items()):
            pruned = [t for t in ticks if t[0] >= cutoff]
            if mint not in self._watched and not pruned:
                del self._history[mint]
            else:
                self._history[mint] = pruned
