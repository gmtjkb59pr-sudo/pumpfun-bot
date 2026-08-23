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

from . import activity_log, position_store
from .activity_log import append_jsonl
from .alerts import Alerter
from .dexscreener import fetch_price_usd
from .fees import ROUND_TRIP_PRIORITY_FEE_SOL, net_pct_change_after_fees
from .price_ref import extract_price_ref, extract_price_ref_for_field
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
# confirmed live: PumpPortal's subscribeTokenTrade WS feed only covers
# trades routed through PumpPortal's own indexed venues (bonding curve /
# PumpSwap), not every venue a mint might actually trade on - a real,
# heavily-traded token (~$51M/24h volume) received ZERO WS trade events
# over 15s, and would otherwise be silently abandoned at STALE_PRICE_
# TIMEOUT_SEC with no exit ever attempted, despite a real market price
# being available via DexScreener the whole time. A position only ever
# falls back to this REST poll when it has NEVER received a real WS tick
# (see _apply_rest_fallback_prices) - a position with working WS data that
# genuinely goes quiet later is left to the existing stale-price path,
# since that silence is real signal (likely dead/rugged), not a feed gap.
REST_FALLBACK_POLL_INTERVAL_SEC = 10
# PumpPortal's own balance index needs this long to catch up with a real buy
# before it can build an accurate "sell 100%" quote - confirmed directly:
# selling sooner reliably comes back SellZeroAmount (their indexer still
# thought our balance was 0), which still costs a real fee to fail. Roughly
# matches STALE_PRICE_TIMEOUT_SEC so a position isn't force-detected as
# stale well before a sell attempt could ever actually succeed.
MIN_SELL_DELAY_SEC = 15
# after this many consecutive REAL sell failures (not the MIN_SELL_DELAY_SEC
# defer, which isn't a failure) for the same position, stop auto-retrying and
# alert instead - confirmed live TWICE now: a position hitting PumpPortal's
# own program throwing a deterministic on-chain error on every attempt
# (AnchorError "Overflow", Custom 6024; and separately Custom 6022) that can
# never succeed no matter how many times it's retried. user-requested:
# lowered from 5 -> 2 after watching a stuck position burn a real priority
# fee on every failed attempt - 2 is enough to rule out a one-off transient
# failure without paying for 3-4 more attempts against a deterministic
# error. The position stays tracked and visible (never silently dropped -
# only wallet reconciliation does that, and only once the wallet genuinely
# no longer holds it), just no longer auto-retried, so a human can decide
# how to actually get it out (e.g. a manual swap).
MAX_CONSECUTIVE_SELL_FAILURES = 2
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


def load_confirmed_unsellable_mints(log_path=None) -> set[str]:
    """Returns every mint that has ever logged a "sell_paused" record (see
    _record_sell_failure and _liquidate_untracked_holdings_inner, both of
    which append one once a mint hits MAX_CONSECUTIVE_SELL_FAILURES) -
    user-requested: without this, the untracked-holdings liquidation sweep
    forgot this history on every restart (its pause state is in-memory
    only) and re-burned a real priority fee re-proving the same
    already-known-unsellable mint every single time, since the deterministic
    on-chain errors behind this never resolve. Fails open (empty set) on a
    missing/unreadable log - never treat "can't read the history" as
    license to skip a real liquidation attempt that might actually work.

    log_path defaults to None (not activity_log.DATA_LOG_PATH bound at
    import time) and is resolved fresh here on every call - same reasoning
    as OutcomeTracker.position_store_path: a module-level default captured
    at import time would never see a test's later reassignment of
    activity_log.DATA_LOG_PATH."""
    if log_path is None:
        log_path = activity_log.DATA_LOG_PATH
    mints: set[str] = set()
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("type") == "sell_paused":
                    mint = d.get("mint")
                    if mint:
                        mints.add(mint)
    except FileNotFoundError:
        pass
    return mints


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
        silently abandoning them (see position_store.py module docstring).

        Also pre-seeds _untracked_liquidation with mints already confirmed
        unsellable in a PAST session (see load_confirmed_unsellable_mints) -
        without this, every restart forgot that history and re-attempted
        (and re-failed, and re-paid the fee for) a liquidation already
        proven doomed. A mint that's back in _pending this run isn't
        untracked anymore, so it's excluded here - that's the real, still-
        tracked position's own sell_paused flag (persisted separately in
        position_store) to manage, not this one."""
        loaded = position_store.load(self.position_store_path)
        if loaded:
            logger.warning(
                "%d open positie(s) hersteld van vorige run: %s",
                len(loaded), ", ".join(loaded.keys()),
            )
        self._pending.update(loaded)

        if not self.dry_run:
            previously_unsellable = load_confirmed_unsellable_mints() - set(self._pending.keys())
            if previously_unsellable:
                logger.info(
                    "%d niet-getrackte holding(s) al eerder bevestigd onverkoopbaar - "
                    "wordt deze run niet opnieuw geprobeerd: %s",
                    len(previously_unsellable), ", ".join(sorted(previously_unsellable)),
                )
            for mint in previously_unsellable:
                self._untracked_liquidation[mint] = {
                    "consecutive_failures": MAX_CONSECUTIVE_SELL_FAILURES,
                    "last_attempt_ts": 0.0,
                    "paused": True,
                }

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
        price_source: str = "bonding_curve_sol",
        take_profit_ladder: list | None = None,
        price_ref_field: str | None = None,
        max_hold_sec: float | None = None,
        stale_price_timeout_sec: float | None = None,
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
        own budget yet - they stay on the original shared-pool count.

        price_source: confirmed live - sniper/social_watch's entry_ref
        comes from extract_price_ref() (a bonding-curve-relative SOL-scale
        proxy: marketCapSol/vSolInBondingCurve/etc), the SAME basis every
        real WS trade tick is extracted in. birdeye_movers/coingecko_movers'
        entry_ref instead comes from an external absolute USD price
        (Birdeye/CoinGecko). Comparing a real WS tick's bonding-curve-scale
        ref against a USD-scale entry_ref produced a nonsensical exit
        (+850,255,839% observed live) - not a real price move, an apples-
        to-oranges unit mismatch. price_source="usd" (set by those two
        strategies) makes _handle_price_update ignore real WS ticks for
        this position entirely and rely solely on the REST fallback (see
        REST_FALLBACK_POLL_INTERVAL_SEC), which is USD-denominated on both
        ends (entry from Birdeye/CoinGecko, updates from DexScreener).

        price_ref_field: confirmed live - the SAME unit-mismatch failure
        mode can happen WITHIN price_source="bonding_curve_sol" alone, not
        just across bonding-curve-vs-USD. A real position exited at exactly
        -100.0% (twice) 1.7s and 4.9s after buying - not a real crash, its
        entry_ref came from one PumpPortal field (e.g. marketCapSol) but a
        later WS trade event for the same mint lacked that field and
        extract_price_ref() fell through to a different, scale-incompatible
        one (e.g. price - ~1e9x smaller for a fixed-supply token). Pass
        whichever field name extract_price_ref_with_field() reported for
        this entry_ref (see price_ref.py's module docstring) so later WS
        ticks only ever re-extract that SAME field - see
        _handle_price_update_from_event().

        max_hold_sec/stale_price_timeout_sec: user-requested for a
        moonshot-style strategy meant to hold a position for potentially
        days/weeks aiming at a huge multiple - the module-level MAX_HOLD_SEC
        (15 min) and STALE_PRICE_TIMEOUT_SEC (10s!) defaults exist for
        fast-moving sniper/social_watch positions and would force-exit a
        genuinely healthy long-hold position the moment it saw one natural
        trading lull. None (default) keeps the existing shared-constant
        behavior for every other strategy unchanged."""
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
                # see track()'s max_hold_sec/stale_price_timeout_sec docstring -
                # None (the default for every existing strategy) falls back to
                # the shared MAX_HOLD_SEC/STALE_PRICE_TIMEOUT_SEC constants
                "max_hold_sec": max_hold_sec,
                "stale_price_timeout_sec": stale_price_timeout_sec,
                "strategy": strategy,
                # see track()'s own docstring for the full reasoning
                "price_source": price_source,
                # see track()'s price_ref_field docstring - None (e.g. for
                # price_source="usd" positions, which never accept a real WS
                # tick anyway) falls back to the old any-field extraction
                "price_ref_field": price_ref_field,
                # see REST_FALLBACK_POLL_INTERVAL_SEC's docstring - set True
                # once a REST-fallback price has been used for this position
                # at least once, so it keeps getting REST-polled even after
                # has_real_update flips True from that same fallback price
                "rest_fallback_active": False,
                "last_rest_fallback_ts": 0.0,
                # user-requested: scaled take-profit ladder (see
                # TakeProfitLevel's docstring in config.py). Empty list keeps
                # the existing single-shot take_profit_pct behavior in
                # _handle_price_update unchanged. Stored as plain dicts (not
                # TakeProfitLevel instances) so position_store.py's JSON
                # persistence round-trips this without a custom encoder.
                "take_profit_ladder": (
                    [{"multiplier": lvl.multiplier, "sell_pct": lvl.sell_pct} for lvl in take_profit_ladder]
                    if take_profit_ladder else []
                ),
                "triggered_ladder_levels": [],
                "remaining_fraction": 1.0,
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
            # before the stale-price check below can give up on a position -
            # see REST_FALLBACK_POLL_INTERVAL_SEC's docstring for why
            await self._apply_rest_fallback_prices()
            await self._emit_due_checkpoints()
            await self._emit_post_exit_checkpoints()

    async def _apply_rest_fallback_prices(self) -> None:
        """Fetches a DexScreener price for any position that has never
        received a real WS tick (or is already relying on this same
        fallback), rate-limited per-position to REST_FALLBACK_POLL_INTERVAL_
        SEC. A position with working WS data is never touched here - zero
        extra REST calls for the common case.

        Also always polls price_source="usd" positions (birdeye_movers/
        coingecko_movers), regardless of has_real_update - a real WS tick
        is never applied to those (see _handle_price_update's unit-mismatch
        guard), so has_real_update for one of these can ONLY ever have come
        from this same REST fallback, never a reason to stop polling it."""
        now = time.time()
        candidates = []
        async with self._lock:
            for mint, info in self._pending.items():
                needs_fallback = (
                    not info["has_real_update"]
                    or info.get("rest_fallback_active")
                    or info.get("price_source") == "usd"
                )
                if not needs_fallback:
                    continue
                if now - info.get("last_rest_fallback_ts", 0.0) < REST_FALLBACK_POLL_INTERVAL_SEC:
                    continue
                info["last_rest_fallback_ts"] = now
                candidates.append(mint)
        for mint in candidates:
            price = await fetch_price_usd(mint)
            if price is not None:
                await self._handle_price_update(mint, price, via_rest_fallback=True)

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

        async with self._lock:
            info = self._pending.get(mint)
            price_ref_field = info.get("price_ref_field") if info else None

        if price_ref_field:
            # this position has an established field from its entry
            # snapshot - never substitute a different, scale-incompatible
            # one just because this particular event happens to be missing
            # it (see price_ref.py's module docstring for the bogus -100%
            # exit this prevents). A miss here means "no update this tick",
            # not "try a fallback".
            ref = extract_price_ref_for_field(data, price_ref_field)
        else:
            # no tracked position with a recorded field (untracked mint,
            # post_exit-only, or a position persisted before this fix) -
            # fall back to the original any-field extraction
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
            # user-requested: don't keep re-alerting about the SAME mints
            # every reconciliation cycle once they're already confirmed
            # unsellable (paused) - nothing new to act on, just noise. Only
            # surface this when at least one untracked mint is actually
            # still worth a real attempt.
            worth_alerting = {
                m for m in untracked
                if not self._untracked_liquidation.get(m, {}).get("paused")
            }
            if worth_alerting:
                message = (
                    f"⚠️ {len(worth_alerting)} token(s) in de wallet worden niet gevolgd - "
                    f"probeer automatisch te liquideren: {', '.join(sorted(worth_alerting))}"
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
                    # same 99% fallback as _exit() (see its docstring for the
                    # real on-chain Overflow/Custom 6024 this targets) -
                    # untracked holdings hit the identical failure mode, so
                    # give them the same one extra shot before pausing
                    try:
                        fallback_result = await self.client.build_and_send_full_sell(
                            mint=mint, slippage_pct=self.sell_slippage_pct, amount_pct=99,
                        )
                        del self._untracked_liquidation[mint]
                        append_jsonl({
                            "type": "untracked_liquidation",
                            "ts": time.time(),
                            "mint": mint,
                            "tx_signature": fallback_result["signature"],
                            "amount_pct": 99,
                        })
                        message = (
                            f"🧹 Niet-getrackte holding {mint} geliquideerd via 99% fallback "
                            f"(100% mislukte {state['consecutive_failures']}x, laatste fout: {exc}) - "
                            f"tx: {fallback_result['signature']}"
                        )
                        logger.warning(message)
                        if self.alerter is not None:
                            await self.alerter.send(message)
                        continue
                    except Exception as fallback_exc:  # noqa: BLE001
                        state["paused"] = True
                        # already confirmed held (it's only in `untracked`
                        # because _reconcile_with_wallet found it in the
                        # wallet) and now confirmed unsellable at both 100%
                        # and 99% - same reputation signal as a tracked
                        # position hitting this, see that code path
                        append_jsonl({"type": "sell_paused", "ts": time.time(), "mint": mint})
                        message = (
                            f"⏸️ Kon niet-getrackte holding {mint} niet liquideren na "
                            f"{state['consecutive_failures']} pogingen (laatste fout: {exc}), "
                            f"99% fallback ook geprobeerd en mislukt ({fallback_exc}) - "
                            f"gestopt met proberen, controleer handmatig."
                        )
                        logger.warning(message)
                        if self.alerter is not None:
                            await self.alerter.send(message)
                        continue
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

    async def _handle_price_update(
        self, mint: str, ref: float, via_rest_fallback: bool = False,
    ) -> None:
        """Records the new price and exits the position immediately if it
        crosses take-profit, stop-loss, or the trailing stop. Separated from
        _handle_ws_message so this decision logic can be exercised directly
        in tests.

        via_rest_fallback: True when `ref` came from _apply_rest_fallback_
        prices (DexScreener), not a real WS tick - marks the position as
        needing continued REST polling (rest_fallback_active) even after
        has_real_update flips True from this same price. A position that
        DOES get a genuine WS tick clears that flag - real-time WS data is
        cheaper and preferred whenever it's actually available."""
        exit_args = None
        is_pending = False
        async with self._lock:
            info = self._pending.get(mint)
            if info is not None and not via_rest_fallback and info.get("price_source") == "usd":
                # a real WS tick's ref is bonding-curve-scale (marketCapSol/
                # vSolInBondingCurve/etc) - applying it to a USD-denominated
                # entry_ref is an apples-to-oranges unit mismatch, confirmed
                # live to produce a nonsensical exit (+850,255,839% observed
                # on a real position). Only the REST fallback (USD-scale on
                # both ends) is a valid update source for this position -
                # see track()'s price_source docstring.
                info = None
            if info is not None:
                is_pending = True
                info["last_ref"] = ref
                info["last_update_ts"] = time.time()
                info["has_real_update"] = True
                if via_rest_fallback:
                    info["rest_fallback_active"] = True
                elif info.get("rest_fallback_active"):
                    info["rest_fallback_active"] = False
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

                ladder = info.get("take_profit_ladder") or []
                triggered_reason = None
                sell_fraction_of_remaining = 1.0
                ladder_level_multiplier = None
                if ladder:
                    # scaled take-profit ladder replaces the single-shot
                    # take_profit_pct trigger entirely for this position -
                    # see TakeProfitLevel's docstring in config.py. Stop-loss
                    # always applies (closes whatever remains); the trailing
                    # stop only takes over once every ladder rung has fired,
                    # to protect the leftover runner.
                    if pct_change <= -stop_loss_pct:
                        triggered_reason = "stop_loss"
                        sell_fraction_of_remaining = 1.0
                    else:
                        current_multiplier = 1 + (pct_change / 100)
                        triggered_levels = info.get("triggered_ladder_levels") or []
                        next_level = next(
                            (
                                lvl for lvl in ladder
                                if lvl["multiplier"] not in triggered_levels
                                and current_multiplier >= lvl["multiplier"]
                            ),
                            None,
                        )
                        if next_level is not None:
                            # at most one rung per price update - real ticks
                            # arrive often enough to catch up quickly, and
                            # this keeps a single tick from ever triggering
                            # two partial sells at once
                            triggered_reason = "take_profit_ladder"
                            sell_fraction_of_remaining = next_level["sell_pct"] / 100
                            ladder_level_multiplier = next_level["multiplier"]
                        elif (
                            len(triggered_levels) >= len(ladder)
                            and info.get("remaining_fraction", 1.0) > 0
                            and peak_pct_change >= trailing_activation_pct
                            and drawdown_from_peak_pct <= -trailing_stop_pct
                        ):
                            triggered_reason = "trailing_stop"
                            sell_fraction_of_remaining = 1.0
                elif pct_change >= take_profit_pct:
                    triggered_reason = "take_profit"
                elif pct_change <= -stop_loss_pct:
                    triggered_reason = "stop_loss"
                elif (
                    peak_pct_change >= trailing_activation_pct
                    and drawdown_from_peak_pct <= -trailing_stop_pct
                ):
                    triggered_reason = "trailing_stop"
                if triggered_reason and self._exit_attempt_allowed(info):
                    exit_args = (
                        mint, dict(info), triggered_reason, pct_change,
                        sell_fraction_of_remaining, ladder_level_multiplier,
                    )
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

    async def _attempt_exit(
        self, mint: str, info: dict, reason: str, pct_change: float,
        sell_fraction_of_remaining: float = 1.0, ladder_level_multiplier: float | None = None,
    ) -> None:
        """Calls _exit() and only removes the mint from _pending if it
        actually closed - a failed real sell leaves it exactly as it was.
        A partial ladder sell (_exit() returning False by design, not
        failure) also leaves it tracked - see _exit()'s docstring."""
        closed = await self._exit(
            mint, info, reason, pct_change, sell_fraction_of_remaining, ladder_level_multiplier,
        )
        if closed:
            async with self._lock:
                self._pending.pop(mint, None)
                self._persist_pending()

    async def _exit(
        self, mint: str, info: dict, reason: str, pct_change: float,
        sell_fraction_of_remaining: float = 1.0, ladder_level_multiplier: float | None = None,
    ) -> bool:
        """Returns True if the position is now fully closed (simulated
        close, or a real sell that actually succeeded). Returns False both
        when a real sell was attempted and failed (caller must leave the
        position tracked, unchanged) AND when a ladder rung fired a partial
        sell that succeeded (caller must also leave the position tracked,
        but now with a reduced remaining_fraction) - the caller can't tell
        these apart from the return value alone, which is fine: both cases
        want the same "don't pop it" behavior.

        sell_fraction_of_remaining: fraction of whatever's currently left of
        the position to sell now (1.0 = everything, the default/non-ladder
        behavior). A ladder rung passes next_level["sell_pct"] / 100 here -
        see _handle_price_update's ladder branch."""
        remaining_fraction = info.get("remaining_fraction", 1.0)
        sold_fraction_of_original = remaining_fraction * sell_fraction_of_remaining
        new_remaining_fraction = remaining_fraction - sold_fraction_of_original
        # a rung that would leave only dust behind (<=1% of the original
        # position) closes the position outright instead - avoids a
        # meaningless tiny remainder lingering until timeout
        is_partial = sell_fraction_of_remaining < 0.999999 and new_remaining_fraction > 0.01

        pct_change = round(pct_change, 2) if pct_change is not None else None
        tx_signature = ""

        if is_partial:
            return await self._partial_exit(
                mint, info, reason, pct_change, sold_fraction_of_original,
                new_remaining_fraction, ladder_level_multiplier,
            )

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
                    # user-requested fallback, confirmed live on 3 real
                    # stuck positions: PumpPortal's on-chain program threw a
                    # deterministic AnchorError ("Overflow", Custom 6024) on
                    # every single 100% sell attempt - the failure happened
                    # AFTER a successful GetFees sub-call, right at the
                    # point of computing the sell amount, consistent with an
                    # edge case in fully-liquidating a position (a divide/
                    # overflow at the curve's zero-remaining boundary), not
                    # a slippage or balance-index problem (those fail
                    # differently, e.g. SellZeroAmount). Before giving up
                    # entirely, try once more selling 99% instead of 100% -
                    # if that clears the same edge case, treat the position
                    # as closed (a ~1% dust remainder left in the wallet is
                    # the same threshold the take-profit ladder already
                    # treats as "close enough", see _exit()'s is_partial
                    # dust-closes-outright comment). Costs one more real fee
                    # attempt either way, but could recover an otherwise
                    # permanently stuck position.
                    try:
                        fallback_result = await self.client.build_and_send_full_sell(
                            mint=mint, slippage_pct=self.sell_slippage_pct, amount_pct=99,
                        )
                        tx_signature = fallback_result["signature"]
                        logger.warning(
                            "99%% fallback-verkoop GESLAAGD voor %s na %d mislukte 100%% "
                            "pogingen (laatste fout: %s) - positie wordt als gesloten "
                            "beschouwd, kleine dust (~1%%) blijft mogelijk achter.",
                            info["symbol"], failure_count, exc,
                        )
                    except Exception as fallback_exc:  # noqa: BLE001
                        logger.exception(
                            "99%% fallback-verkoop OOK mislukt voor %s: %s",
                            info["symbol"], fallback_exc,
                        )
                        message = (
                            f"⏸️ Automatische verkoop gepauzeerd voor {info['symbol']} ({mint}) - "
                            f"{failure_count} verkopen op rij mislukt (laatste fout: {exc}), "
                            f"99% fallback ook geprobeerd en mislukt ({fallback_exc}). "
                            f"Waarschijnlijk een blijvende fout die niet opgelost wordt door "
                            f"opnieuw te proberen. Positie blijft open en getrackt, maar wordt "
                            f"niet langer automatisch verkocht - controleer handmatig."
                        )
                        logger.error(message)
                        if self.alerter is not None:
                            await self.alerter.send(message)
                        return False
                else:
                    message = (
                        f"❌ Sell mislukt voor {info['symbol']} ({reason}): {exc} - "
                        f"positie blijft open, wordt opnieuw geprobeerd."
                    )
                    if self.alerter is not None:
                        await self.alerter.send(message)
                    return False

        # scales to whatever's actually left when this closes a position
        # that already had ladder rungs partially sold off - remaining_
        # fraction defaults to 1.0 (the full original size) for every
        # position that never had a ladder, so this is a no-op for them
        effective_trade_size_sol = round(info["trade_size_sol"] * remaining_fraction, 9)
        if self.risk is not None and effective_trade_size_sol:
            # pct_change is None for a blind forced sell (timeout_unmeasured -
            # never got price data) - pnl is genuinely unknown, so record 0
            # rather than guessing, but still release the exposure slot since
            # the position really is closing
            if pct_change is not None:
                net_pct = net_pct_change_after_fees(pct_change)
                pnl_sol = round(effective_trade_size_sol * (net_pct / 100), 6)
            else:
                pnl_sol = 0.0
            if not self.dry_run:
                # a real buy and a real sell transaction were each submitted
                # with a real priority fee attached - subtract that actual
                # on-chain cost so the dashboard's P&L matches the wallet
                pnl_sol = round(pnl_sol - ROUND_TRIP_PRIORITY_FEE_SOL, 6)
            self.risk.register_trade_closed(effective_trade_size_sol, pnl_sol)
        append_jsonl({
            "type": "exit",
            "ts": time.time(),
            "mint": mint,
            "name": info["name"],
            "symbol": info["symbol"],
            "reason": reason,
            "pct_change": pct_change,
            "measured": pct_change is not None,
            "trade_size_sol": effective_trade_size_sol,
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

    async def _partial_exit(
        self, mint: str, info: dict, reason: str, pct_change: float | None,
        sold_fraction_of_original: float, new_remaining_fraction: float,
        ladder_level_multiplier: float | None,
    ) -> bool:
        """Sells one ladder rung's worth of the position and leaves the rest
        open and tracked - always returns False (never pops the position;
        _attempt_exit only pops on a fully-closed True). Mutates the REAL
        self._pending[mint] entry (info here is a dict(info) copy handed
        around as exit_args, not the live tracked state) so the reduced
        remaining_fraction and newly-triggered rung actually stick.

        Sizes the real sell in CURRENT SOL terms (slice_cost_sol_at_entry *
        multiplier), since PumpPortal's amount_sol for a sell means "tokens
        worth this much SOL right now" - selling the ORIGINAL entry-time SOL
        cost of this slice would undersell the actual token quantity after
        the price moved up (which is why a rung fired at all)."""
        slice_cost_sol_at_entry = info["trade_size_sol"] * sold_fraction_of_original
        multiplier = 1 + ((pct_change or 0.0) / 100)
        current_slice_value_sol = round(slice_cost_sol_at_entry * multiplier, 9)
        tx_signature = ""

        if not self.dry_run:
            if self.client is None:
                logger.error(
                    "LIVE modus maar geen trading client ingesteld op de outcome-tracker - "
                    "kan take-profit ladder rung niet verkopen voor %s. Positie blijft open.",
                    info["symbol"],
                )
                return False
            time_since_entry = time.time() - info["entry_ts"]
            if time_since_entry < MIN_SELL_DELAY_SEC:
                logger.debug(
                    "Ladder-sell voor %s uitgesteld - pas %.0fs sinds aankoop (min %ds).",
                    info["symbol"], time_since_entry, MIN_SELL_DELAY_SEC,
                )
                return False
            try:
                result = await self.client.build_and_send_trade(
                    action="sell", mint=mint, amount_sol=current_slice_value_sol,
                    slippage_pct=self.sell_slippage_pct,
                )
                tx_signature = result["signature"]
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "ECHTE ladder-sell mislukt voor %s (%s, multiplier=%s): %s - "
                    "positie blijft open, wordt over %ds opnieuw geprobeerd.",
                    info["symbol"], mint, ladder_level_multiplier, exc, EXIT_RETRY_COOLDOWN_SEC,
                )
                await self._record_sell_failure(mint)
                if self.alerter is not None:
                    await self.alerter.send(
                        f"❌ Ladder-sell mislukt voor {info['symbol']} "
                        f"({ladder_level_multiplier}x): {exc} - positie blijft open."
                    )
                return False

        if self.risk is not None and slice_cost_sol_at_entry:
            net_pct = net_pct_change_after_fees(pct_change) if pct_change is not None else 0.0
            pnl_sol = round(slice_cost_sol_at_entry * (net_pct / 100), 6)
            self.risk.register_trade_closed(slice_cost_sol_at_entry, pnl_sol)

        append_jsonl({
            "type": "exit",
            "ts": time.time(),
            "mint": mint,
            "name": info["name"],
            "symbol": info["symbol"],
            "reason": reason,
            "pct_change": round(pct_change, 2) if pct_change is not None else None,
            "measured": pct_change is not None,
            "trade_size_sol": round(slice_cost_sol_at_entry, 9),
            "dry_run": self.dry_run,
            "tx_signature": tx_signature,
            "partial": True,
            "ladder_level_multiplier": ladder_level_multiplier,
            "remaining_fraction_after": round(new_remaining_fraction, 6),
        })
        if self.alerter is not None:
            prefix = "[DRY RUN] " if self.dry_run else ""
            pct_str = f"{pct_change:+.1f}%" if pct_change is not None else "onbekend"
            await self.alerter.send(
                f"{prefix}🪜 Ladder take-profit ({ladder_level_multiplier}x): "
                f"{info['name']} ({info['symbol']}) @ {pct_str} - "
                f"{round(new_remaining_fraction * 100, 1)}% resteert"
            )

        async with self._lock:
            real_info = self._pending.get(mint)
            if real_info is not None:
                real_info["remaining_fraction"] = new_remaining_fraction
                if ladder_level_multiplier is not None:
                    real_info["triggered_ladder_levels"] = (
                        real_info.get("triggered_ladder_levels") or []
                    ) + [ladder_level_multiplier]
                self._persist_pending()
        return False

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

                stale_price_timeout_sec = info.get("stale_price_timeout_sec") or STALE_PRICE_TIMEOUT_SEC
                last_update_age = now - info.get("last_update_ts", info["entry_ts"])
                if last_update_age >= stale_price_timeout_sec:
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
                                info["symbol"], mint, stale_price_timeout_sec,
                            )
                            no_data_warnings.append((info["symbol"], stale_price_timeout_sec))
                        to_timeout_exit.append((mint, dict(info), "stale_price_unmeasured", None))
                    continue

                max_hold_sec = info.get("max_hold_sec") or MAX_HOLD_SEC
                if age >= max_hold_sec:
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
                                info["symbol"], mint, max_hold_sec,
                            )
                            no_data_warnings.append((info["symbol"], max_hold_sec))
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
