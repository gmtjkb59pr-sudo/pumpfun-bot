"""
Follows up on buys with an actual exit strategy - take-profit, stop-loss,
a trailing stop, or a timeout - instead of just watching price drift
forever. This is what makes the bot's P&L mean something: measuring raw
buy-and-hold price movement doesn't tell you whether a strategy is
profitable, because real trading is entries AND exits.

The trailing stop exists because live data showed most positions (~59%)
never hit take-profit or stop-loss at all - they just time out after 15
minutes, and that bucket loses money on median. Many of those had pumped
partway before fading back down; a fixed take-profit target that's never
reached gives back the whole move. The trailing stop arms once a position
is up trailing_activation_pct from entry, then exits if price falls
trailing_stop_pct from its peak since entry - locking in part of a move
that didn't reach the take-profit target, instead of riding it to a
timeout loss.

In dry_run mode (the default), exits are simulated exactly as before - no
network calls, no real trade. When dry_run is False, an exit sends a real
build_and_send_full_sell() through the trading client, and a position is
ONLY ever marked closed (P&L registered, tracking stopped) if that sell
actually succeeds. A failed real sell leaves the position exactly as it
was - still open, still tracked, retried on the next signal - specifically
so the bot's own bookkeeping can never say "sold" while the wallet still
holds the tokens.

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

from . import position_store
from .activity_log import append_jsonl
from .alerts import Alerter
from .fees import ROUND_TRIP_PRIORITY_FEE_SOL, net_pct_change_after_fees
from .price_ref import extract_price_ref
from .pumpportal_client import authenticated_ws_url
from .risk import RiskManager
from .state import bot_state

logger = logging.getLogger("pumpfun_bot.outcome_tracker")

CHECKPOINTS_SEC = (60, 300, 900)
MAX_HOLD_SEC = CHECKPOINTS_SEC[-1]
POLL_WINDOW_SEC = 20
IDLE_SLEEP_SEC = 5
# don't hammer a failing real sell on every single price tick - back off
# between attempts, but keep retrying rather than giving up
EXIT_RETRY_COOLDOWN_SEC = 15
# no real trade event for this mint in this long -> likely dead/rugged, exit
# well before the full MAX_HOLD_SEC timeout instead of holding a dead token
STALE_PRICE_TIMEOUT_SEC = 120

EXIT_EMOJI = {
    "take_profit": "🟢", "stop_loss": "🔴", "trailing_stop": "🟡", "timeout": "⏱️",
    "timeout_unmeasured": "❔", "stale_price": "💤", "stale_price_unmeasured": "💤",
}


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
        trailing_activation_pct: float = 20.0,
        trailing_stop_pct: float = 15.0,
        client=None,
        dry_run: bool = True,
        sell_slippage_pct: float = 10.0,
        position_store_path=None,
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
        # trailing stop arms once a position is up trailing_activation_pct
        # from entry, then exits if price falls trailing_stop_pct from its
        # peak since entry - see module docstring for why this exists
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_stop_pct = trailing_stop_pct
        # PumpPortalClient used to send a real sell when dry_run is False -
        # required for live exits, unused (and unneeded) in dry-run
        self.client = client
        self.dry_run = dry_run
        self.sell_slippage_pct = sell_slippage_pct
        # resolved at construction time, not import time, so tests (and
        # anything else) can point position_store.DEFAULT_STORE_PATH
        # elsewhere before any OutcomeTracker is constructed
        self.position_store_path = position_store_path or position_store.DEFAULT_STORE_PATH
        self._pending: dict[str, dict] = {}
        # mints that already exited - kept under passive observation (no
        # P&L/exposure effect, already realized) purely to answer "would
        # holding longer have done better than the exit strategy did?"
        self._post_exit: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._warned_no_access = False

    def load_pending(self) -> None:
        """Reconstructs _pending from disk - call once at startup, before
        run(), so a restart resumes tracking real open positions instead of
        silently abandoning them (see position_store.py module docstring)."""
        loaded = position_store.load(self.position_store_path)
        if loaded:
            logger.warning(
                "%d open positie(s) hersteld van vorige run: %s",
                len(loaded), ", ".join(loaded.keys()),
            )
        self._pending.update(loaded)

    def _persist_pending(self) -> None:
        position_store.save(self._pending, self.position_store_path)

    def open_position_count(self) -> int:
        """The real, current number of positions actually being tracked -
        the single source of truth for "is there room to buy another",
        rather than a separately-incremented counter that can drift from
        reality (e.g. if track() ever returns early - no price ref, an
        already-tracked mint - a naive counter would still say a slot is
        taken even though nothing is actually being held/managed there)."""
        return len(self._pending)

    def is_tracking(self, mint: str) -> bool:
        """Whether a position for this mint is already open - strategies
        should check this BEFORE buying, not just before calling track(),
        since two strategies sharing this tracker could otherwise both spend
        real SOL buying the same mint (only the second call's position would
        ever be tracked/managed; the first would be bought for nothing)."""
        return mint in self._pending

    async def track(
        self, mint: str, name: str, symbol: str, entry_ref: float | None, trade_size_sol: float = 0.0,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_activation_pct: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> None:
        """take_profit_pct/stop_loss_pct/trailing_*_pct default to this
        instance's own thresholds - pass explicit values when a DIFFERENT
        strategy shares this same tracker (e.g. social_watch) and needs its
        own exit thresholds instead of sniper's. Without this, a shared
        OutcomeTracker silently applies only whichever strategy's config it
        was constructed with to every position, regardless of which
        strategy actually opened it."""
        if entry_ref is None:
            logger.debug("Geen price-ref beschikbaar voor %s, sla outcome-tracking over.", mint)
            return
        async with self._lock:
            if mint in self._pending:
                # a second buy of a mint we already hold slipped through the
                # strategy-level is_tracking() check (a real but rare race) -
                # refuse to clobber the existing tracked position rather than
                # silently losing its entry_ref/P&L and leaking exposure that
                # would never get released
                logger.warning(
                    "track() opnieuw aangeroepen voor %s terwijl al gevolgd - "
                    "genegeerd, bestaande positie blijft leidend.", mint,
                )
                return
            now = time.time()
            self._pending[mint] = {
                "entry_ts": now,
                "entry_ref": entry_ref,
                "last_ref": entry_ref,
                "peak_ref": entry_ref,
                "name": name,
                "symbol": symbol,
                "trade_size_sol": trade_size_sol,
                "hit": set(),
                # only set once we've actually seen a trade event for this mint -
                # without this, a rejected/empty subscription would silently look
                # like "0% change" instead of "never measured"
                "has_real_update": False,
                # last time we got ANY real trade event for this mint - used to
                # detect a token that's gone quiet (likely dead/rugged) well
                # before the full MAX_HOLD_SEC timeout
                "last_update_ts": now,
                "take_profit_pct": take_profit_pct if take_profit_pct is not None else self.take_profit_pct,
                "stop_loss_pct": stop_loss_pct if stop_loss_pct is not None else self.stop_loss_pct,
                "trailing_activation_pct": (
                    trailing_activation_pct if trailing_activation_pct is not None
                    else self.trailing_activation_pct
                ),
                "trailing_stop_pct": (
                    trailing_stop_pct if trailing_stop_pct is not None else self.trailing_stop_pct
                ),
            }
            self._persist_pending()

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
        crosses take-profit, stop-loss, or the trailing stop. Separated from
        _poll_once so this decision logic can be exercised directly in
        tests."""
        exit_args = None
        async with self._lock:
            info = self._pending.get(mint)
            if info is not None:
                info["last_ref"] = ref
                info["last_update_ts"] = time.time()
                info["has_real_update"] = True
                if ref > info["peak_ref"]:
                    info["peak_ref"] = ref
                pct_change = ((ref - info["entry_ref"]) / info["entry_ref"]) * 100
                peak_pct_change = ((info["peak_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100
                drawdown_from_peak_pct = ((ref - info["peak_ref"]) / info["peak_ref"]) * 100

                # per-position thresholds - falls back to this instance's
                # own defaults for positions tracked before this field
                # existed (old persisted state, hand-built test fixtures)
                take_profit_pct = info.get("take_profit_pct", self.take_profit_pct)
                stop_loss_pct = info.get("stop_loss_pct", self.stop_loss_pct)
                trailing_activation_pct = info.get("trailing_activation_pct", self.trailing_activation_pct)
                trailing_stop_pct = info.get("trailing_stop_pct", self.trailing_stop_pct)

                triggered_reason = None
                if pct_change >= take_profit_pct:
                    triggered_reason = "take_profit"
                elif pct_change <= -stop_loss_pct:
                    triggered_reason = "stop_loss"
                elif (
                    peak_pct_change >= trailing_activation_pct
                    and drawdown_from_peak_pct <= -trailing_stop_pct
                ):
                    triggered_reason = "trailing_stop"
                if triggered_reason and self._exit_attempt_allowed(info):
                    exit_args = (mint, dict(info), triggered_reason, pct_change)
                self._persist_pending()

            post = self._post_exit.get(mint)
            if post is not None:
                post["last_ref"] = ref
                post["has_real_update"] = True
        if exit_args:
            await self._attempt_exit(*exit_args)

    def _exit_attempt_allowed(self, info: dict) -> bool:
        """Marks an attempt as starting now and returns whether enough time
        has passed since the last one - keeps a failing real sell from being
        retried on every single price tick."""
        last_attempt = info.get("last_exit_attempt_ts", 0)
        if time.time() - last_attempt < EXIT_RETRY_COOLDOWN_SEC:
            return False
        info["last_exit_attempt_ts"] = time.time()
        return True

    async def _attempt_exit(self, mint: str, info: dict, reason: str, pct_change: float) -> None:
        """Calls _exit() and only removes the mint from _pending if it
        actually closed - a failed real sell leaves it exactly as it was."""
        closed = await self._exit(mint, info, reason, pct_change)
        if closed:
            async with self._lock:
                self._pending.pop(mint, None)
                self._persist_pending()

    async def _exit(self, mint: str, info: dict, reason: str, pct_change: float) -> bool:
        """Returns True if the position is now closed (simulated close, or a
        real sell that actually succeeded). Returns False if a real sell was
        attempted and failed - the caller must leave the position tracked."""
        pct_change = round(pct_change, 2) if pct_change is not None else None
        tx_signature = ""

        if not self.dry_run:
            if self.client is None:
                logger.error(
                    "LIVE modus maar geen trading client ingesteld op de outcome-tracker - "
                    "kan %s niet verkopen. Positie blijft open.", info["symbol"],
                )
                return False
            try:
                result = await self.client.build_and_send_full_sell(
                    mint=mint, slippage_pct=self.sell_slippage_pct,
                )
                tx_signature = result["signature"]
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "ECHTE sell mislukt voor %s (%s, reden=%s): %s - positie blijft open, "
                    "wordt over %ds opnieuw geprobeerd.",
                    info["symbol"], mint, reason, exc, EXIT_RETRY_COOLDOWN_SEC,
                )
                if self.alerter is not None:
                    await self.alerter.send(
                        f"❌ Sell mislukt voor {info['symbol']} ({reason}): {exc} - "
                        f"positie blijft open, wordt opnieuw geprobeerd."
                    )
                return False

        if self.risk is not None and info["trade_size_sol"]:
            # pct_change is None for a blind forced sell (timeout_unmeasured -
            # never got price data) - pnl is genuinely unknown, so record 0
            # rather than guessing, but still release the exposure slot since
            # the position really is closing
            if pct_change is not None:
                net_pct = net_pct_change_after_fees(pct_change)
                pnl_sol = round(info["trade_size_sol"] * (net_pct / 100), 6)
            else:
                pnl_sol = 0.0
            if not self.dry_run:
                # a real buy and a real sell transaction were each submitted
                # with a real priority fee attached - subtract that actual
                # on-chain cost so the dashboard's P&L matches the wallet
                pnl_sol = round(pnl_sol - ROUND_TRIP_PRIORITY_FEE_SOL, 6)
            self.risk.register_trade_closed(info["trade_size_sol"], pnl_sol)
        append_jsonl({
            "type": "exit",
            "ts": time.time(),
            "mint": mint,
            "name": info["name"],
            "symbol": info["symbol"],
            "reason": reason,
            "pct_change": pct_change,
            "measured": pct_change is not None,
            "trade_size_sol": info["trade_size_sol"],
            "dry_run": self.dry_run,
            "tx_signature": tx_signature,
        })
        if self.alerter is not None:
            emoji = EXIT_EMOJI.get(reason, "")
            prefix = "[DRY RUN] " if self.dry_run else ""
            pct_str = f"{pct_change:+.1f}%" if pct_change is not None else "onbekend (geen koersdata)"
            await self.alerter.send(
                f"{prefix}{emoji} Exit ({reason}): {info['name']} ({info['symbol']}) @ {pct_str}"
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
        return True

    async def _emit_due_checkpoints(self) -> None:
        now = time.time()
        to_timeout_exit = []
        no_data_warnings = []
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

                last_update_age = now - info.get("last_update_ts", info["entry_ts"])
                if last_update_age >= STALE_PRICE_TIMEOUT_SEC:
                    # no real trade event for STALE_PRICE_TIMEOUT_SEC - likely
                    # dead/rugged, don't wait for the full MAX_HOLD_SEC timeout
                    if info["has_real_update"]:
                        if self._exit_attempt_allowed(info):
                            pct_change = round(
                                ((info["last_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100, 2
                            )
                            to_timeout_exit.append((mint, dict(info), "stale_price", pct_change))
                    elif self.dry_run:
                        finished_mints.append(mint)
                    elif self._exit_attempt_allowed(info):
                        if not info.get("no_data_warned"):
                            info["no_data_warned"] = True
                            logger.error(
                                "LIVE positie %s (%s) heeft na %ds nog geen koersdata - "
                                "forceer verkoop blind (stale), resultaat wordt als "
                                "'unmeasured' gelogd.",
                                info["symbol"], mint, STALE_PRICE_TIMEOUT_SEC,
                            )
                            no_data_warnings.append((info["symbol"], STALE_PRICE_TIMEOUT_SEC))
                        to_timeout_exit.append((mint, dict(info), "stale_price_unmeasured", None))
                    continue

                if age >= MAX_HOLD_SEC:
                    if info["has_real_update"]:
                        if self._exit_attempt_allowed(info):
                            pct_change = round(
                                ((info["last_ref"] - info["entry_ref"]) / info["entry_ref"]) * 100, 2
                            )
                            to_timeout_exit.append((mint, dict(info), "timeout", pct_change))
                    elif self.dry_run:
                        # never measured, nothing simulated is actually at
                        # stake - fine to just stop tracking in dry-run
                        finished_mints.append(mint)
                    elif self._exit_attempt_allowed(info):
                        # live mode + never measured = we genuinely don't know
                        # the price, but selling doesn't require knowing it in
                        # advance - a real held position with no exit path at
                        # all is worse than one closed blind, so force the
                        # sell anyway and record the outcome as unmeasured
                        # rather than leaving it stuck open forever
                        if not info.get("no_data_warned"):
                            info["no_data_warned"] = True
                            logger.error(
                                "LIVE positie %s (%s) heeft na %ds nog geen koersdata - "
                                "forceer verkoop blind, resultaat wordt als 'unmeasured' gelogd.",
                                info["symbol"], mint, MAX_HOLD_SEC,
                            )
                            no_data_warnings.append((info["symbol"], MAX_HOLD_SEC))
                        to_timeout_exit.append((mint, dict(info), "timeout_unmeasured", None))
            for mint in finished_mints:
                del self._pending[mint]
            if finished_mints:
                self._persist_pending()
        for mint, info, reason, pct_change in to_timeout_exit:
            await self._attempt_exit(mint, info, reason, pct_change)
        if self.alerter is not None:
            for symbol, after_sec in no_data_warnings:
                await self.alerter.send(
                    f"⚠️ Geen koersdata voor {symbol} na {after_sec}s - "
                    f"forceer automatische verkoop blind (resultaat onbekend)."
                )

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
