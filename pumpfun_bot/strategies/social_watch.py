"""
Watches newly-launched tokens for socials appearing within a window, instead
of deciding instantly at launch like SniperStrategy does. Trades speed for a
real, verifiable quality signal: the launch event itself never carries
twitter/telegram/website directly (confirmed by sampling live launches), so
require_socials on the sniper's instant-decision path can't ever match
anything - this strategy instead polls each candidate's off-chain metadata
(see social_metadata.py) for up to watch_window_sec before giving up.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..alerts import Alerter
from ..candidate_price_tracker import CandidatePriceTracker
from ..config import SocialWatchConfig
from ..dexscreener import fetch_price_changes_pct
from ..holder_concentration import SETTLING_DELAY_SEC as CONCENTRATION_SETTLING_DELAY_SEC
from ..holder_concentration import fetch_top10_concentration_pct
from ..holder_count import INDEXING_DELAY_SEC as HOLDER_COUNT_INDEXING_DELAY_SEC
from ..holder_count import fetch_holder_count
from ..market_cap import fetch_market_cap_usd
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..scaled_exit_simulator import ScaledExitSimulator
from ..social_metadata import fetch_has_socials
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.social_watch")


class SocialWatchStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: SocialWatchConfig,
        risk: RiskManager,
        alerter: Alerter,
        trade_size_sol: float,
        slippage_pct: float,
        dry_run: bool,
        outcome_tracker: OutcomeTracker | None = None,
        fresh_ref_timeout_sec: float = 5.0,
        price_tracker: CandidatePriceTracker | None = None,
        scaled_exit_simulator: ScaledExitSimulator | None = None,
    ):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.alerter = alerter
        self.trade_size_sol = trade_size_sol
        self.slippage_pct = slippage_pct
        self.dry_run = dry_run
        self.outcome_tracker = outcome_tracker
        self.fresh_ref_timeout_sec = fresh_ref_timeout_sec
        # user-requested: real 1m/2m momentum, shorter than anything
        # DexScreener's API exposes (see candidate_price_tracker.py) -
        # optional so tests/callers that don't care about this can omit it
        self.price_tracker = price_tracker
        # user-requested "scaled exit" strategy comparison, run purely as a
        # parallel observer - never affects the real buy/sell decision (see
        # scaled_exit_simulator.py)
        self.scaled_exit_simulator = scaled_exit_simulator
        self._watching: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        logger.info(
            "Social-watch strategie gestart (dry_run=%s, watch_window=%ds, poll_interval=%ds).",
            self.dry_run, self.cfg.watch_window_sec, self.cfg.poll_interval_sec,
        )
        asyncio.create_task(self._poll_watchlist_loop())

        async for event in self.client.stream_new_tokens():
            mint = event.get("mint")
            uri = event.get("uri")
            if not mint or not uri:
                continue
            liquidity_sol = event.get("vSolInBondingCurve") or event.get("initialBuy") or 0
            if liquidity_sol and liquidity_sol < self.risk.cfg.min_liquidity_sol:
                continue
            async with self._lock:
                if mint not in self._watching:
                    self._watching[mint] = {"event": event, "added_ts": time.time()}
                    if self.price_tracker is not None:
                        await self.price_tracker.watch(mint)

    async def _poll_watchlist_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.poll_interval_sec)
            await self._poll_once()

    async def _poll_once(self) -> None:
        now = time.time()
        async with self._lock:
            candidates = dict(self._watching)

        expired = [
            mint for mint, info in candidates.items()
            if now - info["added_ts"] >= self.cfg.watch_window_sec
        ]
        if expired:
            async with self._lock:
                for mint in expired:
                    self._watching.pop(mint, None)
            for mint in expired:
                logger.debug(
                    "Social-watch: %s kwalificeerde niet binnen %ds (geen socials, of holder "
                    "count nooit boven de min_holder_count), laten gaan.",
                    mint, self.cfg.watch_window_sec,
                )
                if self.price_tracker is not None:
                    await self.price_tracker.unwatch(mint)

        still_watching = {m: i for m, i in candidates.items() if m not in expired}
        if not still_watching:
            return

        results = await asyncio.gather(*(
            fetch_has_socials(info["event"].get("uri")) for info in still_watching.values()
        ))

        for (mint, info), has_socials in zip(still_watching.items(), results):
            if not has_socials:
                continue
            async with self._lock:
                # may already be gone (expired between the snapshot above
                # and now) - _buy() itself never removes it concurrently,
                # _poll_once runs one call at a time
                if mint not in self._watching:
                    continue
            done = await self._buy(mint, info["event"], info["added_ts"])
            if not done:
                # holder_count wasn't sufficient (or unverifiable) this
                # poll - user-requested: stays on the watchlist and gets
                # retried next poll, instead of being dropped after one
                # shot. The expiry check above still drops it eventually if
                # it never clears the bar within watch_window_sec.
                continue
            async with self._lock:
                self._watching.pop(mint, None)
            if self.price_tracker is not None:
                await self.price_tracker.unwatch(mint)

    async def _fetch_fresh_ref(self, mint: str) -> float | None:
        """A candidate can sit on the watchlist for up to watch_window_sec -
        buying off the stale launch-time price snapshot would silently
        mis-price entry_ref for every exit threshold that follows, so grab
        one live trade update right before buying instead. Falls back to
        None (caller uses the launch snapshot) if nothing arrives in time -
        an illiquid token might not trade again before we act."""
        async def _one() -> float | None:
            async for trade_event in self.client.stream_token_trades([mint]):
                ref = extract_price_ref(trade_event)
                if ref is not None:
                    return ref
            return None

        try:
            return await asyncio.wait_for(_one(), timeout=self.fresh_ref_timeout_sec)
        except asyncio.TimeoutError:
            return None

    async def _buy(self, mint: str, event: dict, added_ts: float) -> bool:
        """Returns True once this candidate is DONE - either bought, or
        terminally rejected (already tracked / risk-blocked / market-cap /
        concentration / momentum) - the caller (_poll_once) removes it from
        the watchlist. Returns False only when holder_count is the reason
        it didn't buy THIS poll and the candidate could still qualify
        later - user-requested: holder count used to be checked exactly
        ONCE, the instant socials were first detected, often as early as
        ~3-20s after launch when almost no real token has accumulated many
        holders yet regardless of the threshold (confirmed live: 83
        candidates checked over ~14 min, none cleared even a lowered
        min_holder_count=70, highest was 51). Now it's rechecked on every
        subsequent poll instead, up until the watch window itself expires
        (see _poll_once's existing expiry logic - this doesn't extend how
        long a candidate is watched, just how many chances its holder count
        gets within that same window)."""
        name = event.get("name", "?")
        symbol = event.get("symbol", "?")
        liquidity_sol = event.get("vSolInBondingCurve")

        if self.outcome_tracker is not None and self.outcome_tracker.is_tracking(mint):
            # already held (e.g. sniper bought this same mint and hasn't
            # exited yet) - buying again would spend real SOL on a position
            # nothing would ever track or exit, since OutcomeTracker keys by
            # mint alone
            logger.info("Social-watch: %s wordt al gevolgd, sla over.", mint)
            return True

        open_positions_count = (
            self.outcome_tracker.open_position_count(strategy="social_watch")
            if self.outcome_tracker is not None else None
        )
        ok, reason = self.risk.can_trade(
            self.trade_size_sol, liquidity_sol, open_positions_count=open_positions_count,
            max_open_positions_override=self.cfg.max_open_positions,
            max_sol_per_trade_override=self.trade_size_sol,
        )
        if not ok:
            logger.info("Social-watch: trade geblokkeerd door risk manager: %s", reason)
            return True

        # a candidate can get bought on the very FIRST poll cycle if socials
        # were already present at launch - too soon for either signal below
        # to be trustworthy yet. Top up to whichever wait the ACTIVE filters
        # need before trusting their numbers, measured from the token's own
        # launch (added_ts). Two different reasons, two different windows:
        #
        # - holder count: an RPC indexing lag (getProgramAccounts hasn't
        #   caught up yet) - confirmed by re-checking a live "0 holders" read
        #   minutes later and finding real holders that were always there.
        #   User-requested speed tradeoff: skipped entirely when
        #   min_holder_count <= 1, since at that threshold the count barely
        #   gates anything (see holder_count.py's INDEXING_DELAY_SEC).
        # - top-10 concentration: not an indexing lag at all - right after
        #   launch essentially all supply just IS still with the deployer/
        #   curve, because no one has traded yet. A short settling window
        #   gives real trades a chance to happen first (see
        #   holder_concentration.py's SETTLING_DELAY_SEC).
        #
        # Only ever wait once, for whichever of these is longer - not both
        # sequentially.
        required_delay = 0.0
        if self.cfg.min_holder_count > 1:
            required_delay = max(required_delay, HOLDER_COUNT_INDEXING_DELAY_SEC)
        if self.cfg.max_top10_concentration_pct > 0:
            required_delay = max(required_delay, CONCENTRATION_SETTLING_DELAY_SEC)
        if required_delay > 0:
            elapsed_since_launch = time.time() - added_ts
            if elapsed_since_launch < required_delay:
                # too early to trust holder-count/concentration numbers yet -
                # retry on the next poll instead of blocking this whole poll
                # cycle asleep (holder_count is now retried across polls
                # anyway, see this method's docstring)
                return False

        # unlike sniper's instant buy, social_watch already tolerates real
        # delay - fetch these synchronously so the values are accurate AT
        # the decision point, instead of a delayed best-effort background log
        (
            entry_ref, holder_count, market_cap_usd, top10_concentration_pct, price_changes_pct,
        ) = await asyncio.gather(
            self._fetch_fresh_ref(mint),
            fetch_holder_count(mint, self.client.rpc_http_url),
            fetch_market_cap_usd(mint),
            fetch_top10_concentration_pct(mint, self.client.rpc_http_url),
            fetch_price_changes_pct(mint),
        )
        # only m5 gates the buy decision (user-requested, see dexscreener.py) -
        # h1/h6/h24 are logged below with the trade but not enforced yet
        price_change_5m_pct = price_changes_pct.get("m5") if price_changes_pct is not None else None
        if entry_ref is None:
            entry_ref = extract_price_ref(event)
        if holder_count is None:
            # couldn't verify - not evidence the token is bad, just that the
            # RPC lookup itself failed. Retryable - might succeed next poll.
            logger.debug("Social-watch: kon holder count niet verifiëren voor %s, probeer opnieuw.", mint)
            if self.cfg.min_holder_count > 0:
                # a real minimum is set (auto-tuned from evidence, see
                # holder_count_tuning.py) - an unverifiable count can't be
                # confirmed to clear that bar, so don't buy blind this poll
                return False
        elif holder_count < self.cfg.min_holder_count:
            logger.info(
                "Social-watch: %s heeft %d holders, onder de min_holder_count van %d, "
                "probeer opnieuw volgende poll.", mint, holder_count, self.cfg.min_holder_count,
            )
            return False

        if self.cfg.min_market_cap_usd > 0:
            # user-requested: thin market caps correlated with the worst
            # stop-loss overshoots live - a single sell can crater an
            # illiquid curve 30-50% in one trade tick, and a low market cap
            # is the clearest available signal for that risk
            if market_cap_usd is None:
                logger.info("Social-watch: market cap onbekend voor %s, sla over.", mint)
                return True
            if market_cap_usd < self.cfg.min_market_cap_usd:
                logger.info(
                    "Social-watch: %s heeft $%.0f market cap, onder de min_market_cap_usd "
                    "van $%.0f, sla over.", mint, market_cap_usd, self.cfg.min_market_cap_usd,
                )
                return True

        if self.cfg.max_top10_concentration_pct > 0:
            # user-requested: a manufactured/bundled launch concentrates the
            # float in a handful of wallets that can dump together - same
            # failure mode behind the deep stop-loss overshoots (see
            # holder_concentration.py)
            if top10_concentration_pct is None:
                logger.info("Social-watch: top-10 concentratie onbekend voor %s, sla over.", mint)
                return True
            if top10_concentration_pct > self.cfg.max_top10_concentration_pct:
                logger.info(
                    "Social-watch: %s heeft %.0f%% top-10 concentratie, boven de "
                    "max_top10_concentration_pct van %.0f%%, sla over.",
                    mint, top10_concentration_pct, self.cfg.max_top10_concentration_pct,
                )
                return True

        if self.cfg.require_positive_momentum_5m:
            # user-requested "movers"-style filter: only buy candidates
            # already gaining, not just fresh with good fundamentals - see
            # dexscreener.py for why this is the ToS-clean data source
            # instead of scraping pump.fun's own Movers tab
            if price_change_5m_pct is None:
                logger.info("Social-watch: 5m prijsverandering onbekend voor %s, sla over.", mint)
                return True
            if price_change_5m_pct <= 0:
                logger.info(
                    "Social-watch: %s heeft %.1f%% prijsverandering (5m), geen positief "
                    "momentum, sla over.", mint, price_change_5m_pct,
                )
                return True

        if self.cfg.max_price_change_5m_pct > 0:
            # user-requested, evidence-based (real dry-run outcomes tonight):
            # losing trades averaged 216% momentum at buy vs 134% for
            # winners, and win rate drops sharply above ~100% (80% win at
            # 0-50% momentum down to ~25-31% above 150%) - buying a token
            # already up 100%+ correlates with buying near a local top, not
            # catching a real move early
            if price_change_5m_pct is None:
                logger.info("Social-watch: 5m prijsverandering onbekend voor %s, sla over.", mint)
                return True
            if price_change_5m_pct > self.cfg.max_price_change_5m_pct:
                logger.info(
                    "Social-watch: %s heeft %.1f%% prijsverandering (5m), boven de "
                    "max_price_change_5m_pct van %.0f%% (waarschijnlijk al over de piek), "
                    "sla over.", mint, price_change_5m_pct, self.cfg.max_price_change_5m_pct,
                )
                return True

        mcap_str = f"${market_cap_usd:.0f}" if market_cap_usd is not None else "?"
        momentum_str = f", {price_change_5m_pct:+.1f}% (5m)" if price_change_5m_pct is not None else ""
        await self.alerter.send(
            f"👥 Social-watch kandidaat: {name} ({symbol}) - {mint} "
            f"(socials gevonden, {holder_count if holder_count is not None else '?'} holders, "
            f"{mcap_str} mcap{momentum_str})"
        )

        # user-requested: log every available momentum window (not just the
        # m5 one that actually gates the buy) so the best window can be
        # picked from real outcome data later instead of guessing
        momentum_meta = {
            f"price_change_{window}_pct": (price_changes_pct or {}).get(window)
            for window in ("m5", "h1", "h6", "h24")
        }
        # 1m/2m come from our own buffered trade ticks, not DexScreener -
        # that API doesn't expose anything shorter than m5 (see
        # candidate_price_tracker.py)
        if self.price_tracker is not None:
            momentum_meta["price_change_1m_pct"] = self.price_tracker.price_change_pct(mint, 60)
            momentum_meta["price_change_2m_pct"] = self.price_tracker.price_change_pct(mint, 120)

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            bot_state.log_trade(
                "social_watch", "buy", mint, self.trade_size_sol, dry_run=True,
                meta={
                    "liquidity_sol": liquidity_sol, "has_socials": True, "holder_count": holder_count,
                    **momentum_meta,
                },
            )
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="social_watch",
                    take_profit_ladder=self.cfg.take_profit_ladder,
                )
            if self.scaled_exit_simulator is not None and entry_ref is not None:
                self.scaled_exit_simulator.track(mint, entry_ref, self.trade_size_sol)
            return True

        try:
            result = await self.client.build_and_send_trade(
                action="buy", mint=mint, amount_sol=self.trade_size_sol,
                slippage_pct=self.slippage_pct,
            )
            self.risk.register_trade_opened(self.trade_size_sol)
            await self.risk.report_buy_result(success=True)
            creator = event.get("traderPublicKey")
            bot_state.log_trade(
                "social_watch", "buy", mint, self.trade_size_sol,
                dry_run=False, tx_signature=result["signature"],
                meta={
                    "liquidity_sol": liquidity_sol, "has_socials": True,
                    "creator": creator, "holder_count": holder_count,
                    **momentum_meta,
                },
            )
            await self.alerter.send(f"✅ Gekocht (social-watch): {symbol} | tx: {result['signature']}")
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="social_watch",
                    take_profit_ladder=self.cfg.take_profit_ladder,
                )
            if self.scaled_exit_simulator is not None and entry_ref is not None:
                self.scaled_exit_simulator.track(mint, entry_ref, self.trade_size_sol)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Social-watch buy mislukt voor %s: %s", mint, exc)
            await self.risk.report_buy_result(success=False)
            await self.alerter.send(f"❌ Social-watch buy mislukt voor {symbol}: {exc}")
        return True
