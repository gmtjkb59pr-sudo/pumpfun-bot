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
from ..config import SocialWatchConfig
from ..holder_count import INDEXING_DELAY_SEC as HOLDER_COUNT_INDEXING_DELAY_SEC
from ..holder_count import fetch_holder_count
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
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
                    "Social-watch: %s kreeg geen socials binnen %ds, laten gaan.",
                    mint, self.cfg.watch_window_sec,
                )

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
                # may already be gone (expired or bought concurrently)
                # between the snapshot above and now
                if mint not in self._watching:
                    continue
                self._watching.pop(mint, None)
            await self._buy(mint, info["event"], info["added_ts"])

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

    async def _buy(self, mint: str, event: dict, added_ts: float) -> None:
        name = event.get("name", "?")
        symbol = event.get("symbol", "?")
        liquidity_sol = event.get("vSolInBondingCurve")

        if self.outcome_tracker is not None and self.outcome_tracker.is_tracking(mint):
            # already held (e.g. sniper bought this same mint and hasn't
            # exited yet) - buying again would spend real SOL on a position
            # nothing would ever track or exit, since OutcomeTracker keys by
            # mint alone
            logger.info("Social-watch: %s wordt al gevolgd, sla over.", mint)
            return

        open_positions_count = (
            self.outcome_tracker.open_position_count() if self.outcome_tracker is not None else None
        )
        ok, reason = self.risk.can_trade(
            self.trade_size_sol, liquidity_sol, open_positions_count=open_positions_count,
        )
        if not ok:
            logger.info("Social-watch: trade geblokkeerd door risk manager: %s", reason)
            return

        # a candidate can get bought on the very FIRST poll cycle if socials
        # were already present at launch - that can be well under
        # HOLDER_COUNT_INDEXING_DELAY_SEC after the token's own creation,
        # too soon for getProgramAccounts' index to have caught up (same lag
        # confirmed for sniper, just measured from token launch here instead
        # of our own buy). Top up to that minimum before trusting the count -
        # confirmed by re-checking a live "0 holders" read minutes later and
        # finding real holders that were always there, just not indexed yet.
        elapsed_since_launch = time.time() - added_ts
        remaining_delay = HOLDER_COUNT_INDEXING_DELAY_SEC - elapsed_since_launch
        if remaining_delay > 0:
            await asyncio.sleep(remaining_delay)

        # unlike sniper's instant buy, social_watch already tolerates real
        # delay - fetch these synchronously so the values are accurate AT
        # the decision point, instead of a delayed best-effort background log
        entry_ref, holder_count = await asyncio.gather(
            self._fetch_fresh_ref(mint),
            fetch_holder_count(mint, self.client.rpc_http_url),
        )
        if entry_ref is None:
            entry_ref = extract_price_ref(event)
        if holder_count is None:
            # couldn't verify - not evidence the token is bad, just that the
            # RPC lookup itself failed
            logger.debug("Social-watch: kon holder count niet verifiëren voor %s.", mint)
            if self.cfg.min_holder_count > 0:
                # a real minimum is set (auto-tuned from evidence, see
                # holder_count_tuning.py) - an unverifiable count can't be
                # confirmed to clear that bar, so don't buy blind
                logger.info("Social-watch: holder count onbekend voor %s, sla over.", mint)
                return
        elif holder_count < self.cfg.min_holder_count:
            logger.info(
                "Social-watch: %s heeft %d holders, onder de min_holder_count van %d, sla over.",
                mint, holder_count, self.cfg.min_holder_count,
            )
            return

        await self.alerter.send(
            f"👥 Social-watch kandidaat: {name} ({symbol}) - {mint} "
            f"(socials gevonden, {holder_count if holder_count is not None else '?'} holders)"
        )

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            bot_state.log_trade(
                "social_watch", "buy", mint, self.trade_size_sol, dry_run=True,
                meta={"liquidity_sol": liquidity_sol, "has_socials": True, "holder_count": holder_count},
            )
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    take_profit_pct=self.cfg.take_profit_pct,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                )
            return

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
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Social-watch buy mislukt voor %s: %s", mint, exc)
            await self.risk.report_buy_result(success=False)
            await self.alerter.send(f"❌ Social-watch buy mislukt voor {symbol}: {exc}")
