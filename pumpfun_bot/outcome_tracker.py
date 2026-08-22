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
from .wallet_reconciliation import fetch_wallet_token_mints

logger = logging.getLogger("pumpfun_bot.outcome_tracker")

CHECKPOINTS_SEC = (60, 300, 900)
MAX_HOLD_SEC = CHECKPOINTS_SEC[-1]
# how often the housekeeping loop (checkpoints, stale-price check, wallet
# reconcile, subscription sync) runs while the WS connection is up - this no
# longer gates how fast price ticks are seen (see _run_connection: messages
# are handled the instant they arrive over a single long-lived connection),
# it only paces the periodic checks that don't need per-tick precision
HOUSEKEEPING_INTERVAL_SEC = 5
# brief backoff before reconnecting after the WS connection actually drops
# (network hiccup, server restart) - NOT a scheduled disconnect. Before this
# fix, the tracker tore down and reopened its WS every ~11s on purpose (see
# git history), which meant a real blind window on every single cycle: any
# price tick during the teardown/reconnect gap was missed entirely. For a
# token that pumps and dumps in seconds, that gap could - and did - miss the
# actual peak, so the trailing stop never armed and take-profit was never
# seen, only found out about after price had already fallen back. A single
# persistent connection removes that recurring gap; this constant now only
# covers the rare case of an actual disconnect.
RECONNECT_BACKOFF_SEC = 3
# don't hammer a failing real sell on every single price tick - back off
# between attempts, but keep retrying rather than giving up
EXIT_RETRY_COOLDOWN_SEC = 15
# no real trade event for this mint in this long -> likely dead/rugged, exit
# well before the full MAX_HOLD_SEC timeout instead of holding a dead token
STALE_PRICE_TIMEOUT_SEC = 10
# PumpPortal's own balance index needs this long to catch up with a real buy
# before it can build an accurate "sell 100%" quote - confirmed directly:
# selling sooner reliably comes back SellZeroAmount (their indexer still
# thought our balance was 0), which still costs a real fee to fail. Roughly
# matches STALE_PRICE_TIMEOUT_SEC so a position isn't force-detected as
# stale well before a sell attempt could ever actually succeed.
MIN_SELL_DELAY_SEC = 15
# after this many consecutive REAL sell failures (not the MIN_SELL_DELAY_SEC
# defer, which isn't a failure) for the same position, stop auto-retrying and
# alert instead - confirmed live: a position hit PumpPortal's own program
# throwing "AnchorError ... Error Code: Overflow" (Custom 6024) on every sell
# attempt, a deterministic on-chain math error that can never succeed no
# matter how many times it's retried. Nothing capped that retry loop before -
# it burned a real priority fee every EXIT_RETRY_COOLDOWN_SEC forever. The
# position stays tracked and visible (never silently dropped - only wallet
# reconciliation does that, and only once the wallet genuinely no longer
# holds it), just no longer auto-retried, so a human can decide how to
# actually get it out (e.g. a manual swap).
MAX_CONSECUTIVE_SELL_FAILURES = 5
# how often to reconcile _pending against real wallet holdings - not just
# once at startup, since drift can happen any time a position gets closed
# outside the bot's own exit logic (e.g. a manual sale). Found live: a
# manually-sold position kept getting retried forever, since nothing was
# selling anything, there was nothing left to sell - each retry still cost
# a real fee failing with SellZeroAmount, for a position that was already
# gone. Not too frequent - this is a real RPC call, not free.
WALLET_RECONCILE_INTERVAL_SEC = 60
# a position must be missing from fetch_wallet_token_mints() on this many
# CONSECUTIVE reconciliation cycles before being dropped as stale - a single
# miss could be an incomplete RPC read (confirmed live: a genuinely
# still-held position, verified directly on-chain, was missing from one
# wallet fetch and present again the very next cycle with nothing bought or
# sold in between), not a real sale. Trades a bit of extra lag on a
# genuinely-stale drop (up to ~2x WALLET_RECONCILE_INTERVAL_SEC) for real
# protection against silently abandoning a position that's still held.
WALLET_MISS_CONFIRMATION_COUNT = 2

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
        price_observers: list | None = None,
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
        # elsewhere before any OutcomeTracker is constructed. Dry-run and
        # live get separate files (path_for_mode) so a restart can never
        # load one mode's positions into the other - see position_store.py
        self.position_store_path = position_store_path or position_store.path_for_mode(dry_run)
        # async callables notified with (mint, ref) on every real price tick
        # for a currently-open position - purely additive, e.g. for a
        # parallel counterfactual-strategy simulator (see
        # scaled_exit_simulator.py) that must never affect real exit
        # decisions. Empty by default so existing callers are unaffected.
        self._price_observers = price_observers or []
        self._pending: dict[str, dict] = {}
        # mints that already exited - kept under passive observation (no
        # P&L/exposure effect, already realized) purely to answer "would
        # holding longer have done better than the exit strategy did?"
        self._post_exit: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._warned_no_access = False
        self._last_wallet_reconcile_ts = 0.0
        # mints already sent in a subscribeTokenTrade message on the current
        # WS connection - a fresh connection starts empty, and gets a new
        # subscribe message merely for whatever's newly tracked, not the
        # whole set again (see _sync_subscription)
        self._subscribed_mints: set[str] = set()
        # per-mint liquidation-attempt state for untracked wallet holdings
        # (see _liquidate_untracked_holdings) - separate from _pending since
        # these were never real tracked positions (no entry price, no risk/
        # exposure attribution), just something being cleaned up
        self._untracked_liquidation: dict[str, dict] = {}
        # the currently in-flight liquidation sweep (if any) - guards
        # against starting an overlapping sweep every reconciliation cycle,
        # which would pile up concurrent RPC/confirm-polling load on top of
        # a sweep that's already slow (see _reconcile_with_wallet)
        self._liquidation_task: asyncio.Task | None = None

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

    def open_position_count(self, strategy: str | None = None) -> int:
        """The real, current number of positions actually being tracked -
        the single source of truth for "is there room to buy another",
        rather than a separately-incremented counter that can drift from
        reality (e.g. if track() ever returns early - no price ref, an
        already-tracked mint - a naive counter would still say a slot is
        taken even though nothing is actually being held/managed there).

        Excludes positions with sell_paused=True - user-requested: a
        permanently-stuck position (see MAX_CONSECUTIVE_SELL_FAILURES)
        still holds real capital (still counted in RiskManager's exposure,
        unaffected by this), but nothing further can be done for it
        automatically, so it shouldn't also block ALL new buying by sitting
        in a max_open_positions slot forever.

        strategy: user-requested per-strategy position budgets - pass a
        strategy name (as given to track()) to count only that strategy's
        own open positions, e.g. so a birdeye_movers burst can't crowd
        social_watch out of a shared pool. None (default) counts across
        all strategies, the original shared-pool behavior. Positions
        tracked before the "strategy" field existed have strategy="" and
        are only ever counted by the unscoped (None) call."""
        return sum(
            1 for info in self._pending.values()
            if not info.get("sell_paused") and (strategy is None or info.get("strategy") == strategy)
        )

    def tracked_mints(self) -> set[str]:
        """The mints currently being tracked - used at startup to reconcile
        against what the wallet actually holds on-chain (see
        wallet_reconciliation.py), so leftover tokens from an earlier
        session never get silently confused with what this session owns."""
        return set(self._pending.keys())

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
        strategy: str = "",
    ) -> None:
        """take_profit_pct/stop_loss_pct/trailing_*_pct default to this
        instance's own thresholds - pass explicit values when a DIFFERENT
        strategy shares this same tracker (e.g. social_watch) and needs its
        own exit thresholds instead of sniper's. Without this, a shared
        OutcomeTracker silently applies only whichever strategy's config it
        was constructed with to every position, regardless of which
        strategy actually opened it.

        strategy: user-requested per-strategy position budgets - tags this
        position so open_position_count(strategy=...) can scope to just
        this strategy's own open positions later. Optional/blank for
        callers (sniper, copytrade, market_maker) that don't need their
        own budget yet - they stay on the original shared-pool count."""
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
                "strategy": strategy,
            }
            self._persist_pending()

    async def run(self) -> None:
        while True:
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # this loop manages real, live positions - one bad connection
                # cycle (a dropped WS, a bug anywhere below) must never crash
                # the whole bot. Confirmed live: an unhandled exception here
                # took down the entire asyncio.gather in main.py, abandoning
                # every open position mid-session with no graceful shutdown
                # at all. Log it, keep the loop alive, positions stay tracked.
                logger.exception(
                    "Onverwachte fout in outcome-tracker verbinding - "
                    "opnieuw verbinden, posities blijven gevolgd."
                )
            await asyncio.sleep(RECONNECT_BACKOFF_SEC)

    async def _run_connection(self) -> None:
        """Opens ONE WebSocket connection and keeps it open for the life of
        this call, instead of tearing it down and reopening it every cycle -
        see RECONNECT_BACKOFF_SEC's docstring for why that mattered. Price
        ticks are handled the instant they arrive via `async for`; a
        separate housekeeping task (checkpoints, stale-price check, wallet
        reconcile, subscription sync) runs concurrently on its own cadence
        so it never blocks message receipt."""
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
            await self._reconcile_with_wallet_if_due()
            await self._sync_subscription(ws)
            await self._emit_due_checkpoints()
            await self._emit_post_exit_checkpoints()

    async def _sync_subscription(self, ws) -> None:
        """Adds newly-tracked mints to the live subscription without
        touching the connection itself - a new position opened mid-connection
        must start getting price ticks immediately, not wait for the next
        reconnect (there may not be one for a long time now)."""
        async with self._lock:
            current = set(self._pending.keys()) | set(self._post_exit.keys())
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
            return

        ref = extract_price_ref(data)
        if ref is not None:
            if self._warned_no_access:
                # a real trade event means the feed is actually
                # working now, despite an earlier rejection
                self._warned_no_access = False
                bot_state.set_outcome_tracking_rejected(False)
            await self._handle_price_update(mint, ref)

    async def _reconcile_with_wallet_if_due(self) -> None:
        if self.dry_run or self.client is None:
            return  # dry-run never touches the real wallet, nothing to check
        now = time.time()
        if now - self._last_wallet_reconcile_ts < WALLET_RECONCILE_INTERVAL_SEC:
            return
        self._last_wallet_reconcile_ts = now
        await self._reconcile_with_wallet()

    async def _reconcile_with_wallet(self) -> None:
        """Keeps _pending in sync with what the wallet actually holds, in
        both directions - not just once at startup, since drift can happen
        any time a position closes outside the bot's own exit logic (a
        manual sale, or anything else). A position the wallet no longer
        holds gets dropped from tracking (nothing left to sell there,
        retrying forever just burns fees failing with SellZeroAmount); a
        wallet holding the bot isn't tracking gets surfaced so it can be
        checked manually - it isn't auto-adopted, since we have no idea
        what price it was bought at or what strategy (if any) intended it.

        A single miss isn't enough to drop a position - confirmed live: a
        genuinely still-held position (verified directly on-chain) was
        missing from ONE fetch_wallet_token_mints() call and present again
        the very next cycle, with nothing bought or sold in between. Now
        fixed at the source (see wallet_reconciliation.py - it was only
        querying one of the two token programs a pump.fun mint can use), but
        an RPC read can still fail incompletely in other ways, so a position
        only gets dropped after WALLET_MISS_CONFIRMATION_COUNT CONSECUTIVE
        misses, not one."""
        async with self._lock:
            tracked = set(self._pending.keys())

        held = await fetch_wallet_token_mints(str(self.client.keypair.pubkey()), self.client.rpc_http_url)
        if held is None:
            logger.debug("Kon wallet niet verifiëren voor reconciliatie, sla deze ronde over.")
            return

        untracked = held - tracked
        # surfaces the gap in the dashboard's "open exposure" figure - that
        # number only ever sums positions this bot tracked opening, so it
        # silently misses whatever's untracked here (confirmed live: the
        # wallet held real balances of several leftover mints never counted
        # toward it at all)
        bot_state.set_untracked_holdings_count(len(untracked))

        confirmed_stale = []
        async with self._lock:
            for mint in tracked:
                info = self._pending.get(mint)
                if info is None:
                    continue
                if mint in held:
                    info["wallet_miss_streak"] = 0
                    continue
                info["wallet_miss_streak"] = info.get("wallet_miss_streak", 0) + 1
                if info["wallet_miss_streak"] >= WALLET_MISS_CONFIRMATION_COUNT:
                    confirmed_stale.append(mint)
            for mint in confirmed_stale:
                info = self._pending.pop(mint, None)
                if info is not None and self.risk is not None and info.get("trade_size_sol"):
                    # genuinely unknown what it sold for (if anything) -
                    # release the exposure slot without guessing a P&L
                    self.risk.register_trade_closed(info["trade_size_sol"], 0.0)
            self._persist_pending()

        if confirmed_stale:
            message = (
                f"🔄 {len(confirmed_stale)} positie(s) niet meer echt in de wallet (waarschijnlijk "
                f"handmatig verkocht, {WALLET_MISS_CONFIRMATION_COUNT}x op rij bevestigd) - gestopt "
                f"met volgen, niet langer geteld tegen max_open_positions/exposure: "
                f"{', '.join(sorted(confirmed_stale))}"
            )
            logger.warning(message)
            if self.alerter is not None:
                await self.alerter.send(message)

        # a sell_paused position that's STILL confirmed held here (not
        # dropped above as actually-already-sold) is a real, structurally
        # broken sell, not a confirmation-timeout false alarm - confirmed
        # live: several paused positions turned out to have already sold
        # fine (a slow confirmation, not a real failure) and self-resolved
        # via the stale-drop above; only what survives BOTH the failure cap
        # AND this on-chain re-check is trustworthy enough to hold against
        # the launcher wallet's reputation (see wallet_reputation.py).
        # reputation_logged avoids re-logging the same mint every cycle.
        newly_confirmed_unsellable = []
        async with self._lock:
            for mint, info in self._pending.items():
                if info.get("sell_paused") and mint in held and not info.get("reputation_logged"):
                    info["reputation_logged"] = True
                    newly_confirmed_unsellable.append(mint)
            if newly_confirmed_unsellable:
                self._persist_pending()
        for mint in newly_confirmed_unsellable:
            append_jsonl({"type": "sell_paused", "ts": time.time(), "mint": mint})

        if untracked:
            message = (
                f"⚠️ {len(untracked)} token(s) in de wallet worden niet gevolgd - "
                f"probeer automatisch te liquideren: {', '.join(sorted(untracked))}"
            )
            logger.warning(message)
            if self.alerter is not None:
                await self.alerter.send(message)
            # fired off as a background task, NEVER awaited inline here -
            # confirmed live: awaiting this directly blocked the entire
            # housekeeping loop (checkpoints, stale-price detection,
            # subscription sync for REAL tracked positions) for up to
            # 5 mints * 30s confirmation-timeout each = 150s, because each
            # failed liquidation attempt can take the full 30s to time out.
            # A real position's stale-price detection was delayed from the
            # expected ~10-15s to 44s because of exactly this. Guarded
            # against overlap - only one liquidation sweep runs at a time,
            # so a slow sweep doesn't spawn more concurrent RPC load on top
            # of itself every reconciliation cycle.
            if self._liquidation_task is None or self._liquidation_task.done():
                self._liquidation_task = asyncio.create_task(
                    self._liquidate_untracked_holdings(untracked)
                )

    async def _liquidate_untracked_holdings(self, untracked: set[str]) -> None:
        """Best-effort sell of wallet holdings the bot isn't tracking (see
        _reconcile_with_wallet's docstring for why they're never auto-
        adopted as real positions - no known entry price or strategy).
        Purely a cleanup pass: no take-profit/stop-loss/trailing logic
        applies since there's no entry price to measure against, and it
        never touches risk/exposure - these tokens were never counted as
        open exposure to begin with, so nothing to release. Same failure-
        cap protection as a normal tracked position's exit (see
        MAX_CONSECUTIVE_SELL_FAILURES) - a mint that keeps failing the same
        way (e.g. the same on-chain Overflow bug that hit a tracked
        position) stops being retried instead of burning fees forever.

        Runs as a detached background task (see _reconcile_with_wallet) -
        wrapped in its own try/except so a bug here can't vanish silently
        (asyncio only logs an unretrieved task exception generically) or,
        worse, take down anything else."""
        try:
            await self._liquidate_untracked_holdings_inner(untracked)
        except Exception:  # noqa: BLE001
            logger.exception("Onverwachte fout tijdens liquidatie van niet-getrackte holdings.")

    async def _liquidate_untracked_holdings_inner(self, untracked: set[str]) -> None:
        if self.dry_run or self.client is None:
            return
        for mint in untracked:
            state = self._untracked_liquidation.setdefault(mint, {
                "consecutive_failures": 0, "last_attempt_ts": 0.0, "paused": False,
            })
            if state["paused"]:
                continue
            if time.time() - state["last_attempt_ts"] < EXIT_RETRY_COOLDOWN_SEC:
                continue
            state["last_attempt_ts"] = time.time()

            try:
                result = await self.client.build_and_send_full_sell(
                    mint=mint, slippage_pct=self.sell_slippage_pct,
                )
            except Exception as exc:  # noqa: BLE001
                state["consecutive_failures"] += 1
                if state["consecutive_failures"] >= MAX_CONSECUTIVE_SELL_FAILURES:
                    state["paused"] = True
                    # already confirmed held (it's only in `untracked` because
                    # _reconcile_with_wallet found it in the wallet) and now
                    # confirmed unsellable - same reputation signal as a
                    # tracked position hitting this, see that code path
                    append_jsonl({"type": "sell_paused", "ts": time.time(), "mint": mint})
                    message = (
                        f"⏸️ Kon niet-getrackte holding {mint} niet liquideren na "
                        f"{state['consecutive_failures']} pogingen (laatste fout: {exc}) - "
                        f"gestopt met proberen, controleer handmatig."
                    )
                else:
                    message = f"❌ Liquidatie mislukt voor niet-getrackte holding {mint}: {exc}"
                logger.warning(message)
                if self.alerter is not None:
                    await self.alerter.send(message)
                continue

            del self._untracked_liquidation[mint]
            append_jsonl({
                "type": "untracked_liquidation",
                "ts": time.time(),
                "mint": mint,
                "tx_signature": result["signature"],
            })
            message = f"🧹 Niet-getrackte holding {mint} geliquideerd - tx: {result['signature']}"
            logger.info(message)
            if self.alerter is not None:
                await self.alerter.send(message)

    async def _handle_price_update(self, mint: str, ref: float) -> None:
        """Records the new price and exits the position immediately if it
        crosses take-profit, stop-loss, or the trailing stop. Separated from
        _handle_ws_message so this decision logic can be exercised directly
        in tests."""
        exit_args = None
        is_pending = False
        async with self._lock:
            info = self._pending.get(mint)
            if info is not None:
                is_pending = True
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
        if is_pending:
            for observer in self._price_observers:
                await observer(mint, ref)

    def _exit_attempt_allowed(self, info: dict) -> bool:
        """Marks an attempt as starting now and returns whether enough time
        has passed since the last one - keeps a failing real sell from being
        retried on every single price tick. Also refuses once a position has
        been paused after MAX_CONSECUTIVE_SELL_FAILURES real failures - see
        that constant's docstring.

        Deliberately checks MIN_SELL_DELAY_SEC BEFORE consuming the retry
        cooldown below, not after. Confirmed live: about half of all real
        exits took 18-23s to actually confirm, vs. 1-3s for the rest -
        because the FIRST attempt (right when a token was detected as dead,
        as early as ~10s post-buy) was still too soon for MIN_SELL_DELAY_SEC,
        so _exit() deferred it - but the cooldown timestamp below was
        already consumed at that moment regardless. The bot then had to
        wait out the FULL EXIT_RETRY_COOLDOWN_SEC from that deferred, never-
        actually-attempted check, instead of retrying the moment
        MIN_SELL_DELAY_SEC itself elapsed - adding up to ~10s of pure dead
        time to a real, already-decided sell for no benefit."""
        if info.get("sell_paused"):
            return False
        if not self.dry_run and time.time() - info["entry_ts"] < MIN_SELL_DELAY_SEC:
            return False
        last_attempt = info.get("last_exit_attempt_ts", 0)
        if time.time() - last_attempt < EXIT_RETRY_COOLDOWN_SEC:
            return False
        info["last_exit_attempt_ts"] = time.time()
        return True

    async def _record_sell_failure(self, mint: str) -> int:
        """Increments and persists the consecutive-real-sell-failure count
        for a position, pausing further automatic attempts once
        MAX_CONSECUTIVE_SELL_FAILURES is reached. Operates on the real
        self._pending entry (not the dict copy _exit() was called with),
        since that copy is what gets handed around as exit_args and isn't
        the persisted state. Returns the new count (0 if the position was
        removed from tracking - e.g. by a concurrent wallet reconcile -
        before this ran)."""
        async with self._lock:
            info = self._pending.get(mint)
            if info is None:
                return 0
            count = info.get("consecutive_sell_failures", 0) + 1
            info["consecutive_sell_failures"] = count
            if count >= MAX_CONSECUTIVE_SELL_FAILURES:
                info["sell_paused"] = True
            self._persist_pending()
            return count

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
            time_since_entry = time.time() - info["entry_ts"]
            if time_since_entry < MIN_SELL_DELAY_SEC:
                # PumpPortal's own balance index needs a moment to catch up
                # with our buy - a "sell 100%" quote built before that
                # reliably comes back as SellZeroAmount (confirmed directly
                # from a failed transaction's logs), which still costs a
                # real fee to fail. Skip the doomed attempt entirely rather
                # than submit-and-fail; the caller's retry cooldown already
                # means we'll try again shortly, once this floor has passed.
                logger.debug(
                    "Sell voor %s uitgesteld - pas %.0fs sinds aankoop, "
                    "PumpPortal's eigen index heeft nog niet bijgewerkt (min %ds).",
                    info["symbol"], time_since_entry, MIN_SELL_DELAY_SEC,
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
                failure_count = await self._record_sell_failure(mint)
                if failure_count >= MAX_CONSECUTIVE_SELL_FAILURES:
                    message = (
                        f"⏸️ Automatische verkoop gepauzeerd voor {info['symbol']} ({mint}) - "
                        f"{failure_count} verkopen op rij mislukt (laatste fout: {exc}). "
                        f"Waarschijnlijk een blijvende fout die niet opgelost wordt door "
                        f"opnieuw te proberen. Positie blijft open en getrackt, maar wordt "
                        f"niet langer automatisch verkocht - controleer handmatig."
                    )
                    logger.error(message)
                else:
                    message = (
                        f"❌ Sell mislukt voor {info['symbol']} ({reason}): {exc} - "
                        f"positie blijft open, wordt opnieuw geprobeerd."
                    )
                if self.alerter is not None:
                    await self.alerter.send(message)
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
                        # realized_pct_change is None for a blind forced exit
                        # (timeout_unmeasured/stale_price_unmeasured - we
                        # never knew the price at exit time) - can't compare
                        # "vs what we realized" against an unknown baseline.
                        # This crashed the entire bot process live (an
                        # unhandled exception here brings down the whole
                        # asyncio.gather in main.py) - never again.
                        if info["realized_pct_change"] is None:
                            vs_realized_pct = None
                        else:
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
