"""
User-requested: a deliberately different bet from every other strategy in
this bot. sniper/social_watch/birdeye_movers/coingecko_movers all cut
losses fast and lock in profit early (tight stop_loss/take_profit, 15-min
max hold) - a risk-managed approach aiming for consistent small wins. This
one instead aims at the rare 100-1000x pump.fun outlier: wide stop-loss,
a take-profit ladder that only starts taking (small) profit at huge
multiples so most of the position keeps riding, and a hold time measured
in days/weeks instead of minutes.

Discovers candidates the SAME way SocialWatchStrategy does (watch new
launches via subscribeNewToken, wait for socials within watch_window_sec)
- reuses the same holder_count/holder_concentration/dexscreener building
blocks, just with much stricter/inverted thresholds. See
MoonshotHunterConfig's docstring in config.py for the full reasoning,
including the honest caveat: there's no reliable early signal for a true
viral 1000x, this is a small, deliberately isolated lottery-ticket
allocation, not a strategy expected to have a proven edge like
social_watch's evidence-based filters.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..alerts import Alerter
from ..config import MoonshotHunterConfig
from ..dexscreener import fetch_price_changes_pct
from ..holder_concentration import SETTLING_DELAY_SEC as CONCENTRATION_SETTLING_DELAY_SEC
from ..holder_concentration import fetch_top10_concentration_pct
from ..holder_count import INDEXING_DELAY_SEC as HOLDER_COUNT_INDEXING_DELAY_SEC
from ..holder_count import fetch_holder_count
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref_with_field
from ..pumpportal_client import PumpPortalClient
from ..risk import RiskManager
from ..scam_social_check import evaluate_social_links, record_scam_links
from ..social_metadata import fetch_social_links
from ..state import bot_state

logger = logging.getLogger("pumpfun_bot.moonshot_hunter")


class MoonshotHunterStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: MoonshotHunterConfig,
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
        # user-requested: don't let this fire back-to-back bets - a real
        # cooldown between attempts, not just max_open_positions (which
        # only limits how many are open AT ONCE, not how often a new one
        # starts once the last one closed)
        self._last_buy_ts: float = 0.0

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        logger.info(
            "Moonshot-hunter strategie gestart (dry_run=%s, watch_window=%ds, "
            "poll_interval=%ds, min_holder_count=%d, min_price_change_5m_pct=%.0f%%).",
            self.dry_run, self.cfg.watch_window_sec, self.cfg.poll_interval_sec,
            self.cfg.min_holder_count, self.cfg.min_price_change_5m_pct,
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
                    "Moonshot-hunter: %s kwalificeerde niet binnen %ds, laten gaan.",
                    mint, self.cfg.watch_window_sec,
                )

        still_watching = {m: i for m, i in candidates.items() if m not in expired}
        if not still_watching:
            return

        results = await asyncio.gather(*(
            fetch_social_links(info["event"].get("uri")) for info in still_watching.values()
        ))

        for (mint, info), links in zip(still_watching.items(), results):
            if not links:
                continue
            # user-requested: check BEFORE spending real money, not after -
            # reuses this same metadata fetch (no extra network round trip)
            # rather than the separate post-buy check in outcome_tracker.py
            # (still runs too, as a safety net for anything that changes
            # between here and the buy actually landing)
            is_sus, reason = await evaluate_social_links(links)
            if is_sus:
                logger.warning(
                    "Moonshot-hunter: %s verdachte socials, koop niet: %s", mint, reason,
                )
                record_scam_links([v for v in links.values() if v])
                async with self._lock:
                    self._watching.pop(mint, None)
                continue
            async with self._lock:
                if mint not in self._watching:
                    continue
            done = await self._buy(mint, info["event"], info["added_ts"])
            if not done:
                # same retry-across-polls behavior as social_watch - see
                # its _buy() docstring for why a one-shot holder-count
                # check produced almost no real buys
                continue
            async with self._lock:
                self._watching.pop(mint, None)

    async def _fetch_fresh_ref(self, mint: str) -> tuple[float | None, str | None]:
        """See social_watch.py's identical method for the full reasoning -
        a fresh trade tick right before buying, not the stale launch-time
        snapshot, and which PumpPortal field supplied it (see price_ref.py's
        module docstring for why a later WS tick must only ever re-extract
        that same field)."""
        async def _one() -> tuple[float | None, str | None]:
            async for trade_event in self.client.stream_token_trades([mint]):
                ref, field = extract_price_ref_with_field(trade_event)
                if ref is not None:
                    return ref, field
            return None, None

        try:
            return await asyncio.wait_for(_one(), timeout=self.fresh_ref_timeout_sec)
        except asyncio.TimeoutError:
            return None, None

    async def _buy(self, mint: str, event: dict, added_ts: float) -> bool:
        """Returns True once this candidate is DONE (bought or terminally
        rejected) - see social_watch.py's _buy() docstring for the
        retry-across-polls design this mirrors for min_holder_count."""
        name = event.get("name", "?")
        symbol = event.get("symbol", "?")
        liquidity_sol = event.get("vSolInBondingCurve")

        if self.outcome_tracker is not None and self.outcome_tracker.is_tracking(mint):
            logger.info("Moonshot-hunter: %s wordt al gevolgd, sla over.", mint)
            return True

        seconds_since_last_buy = time.time() - self._last_buy_ts
        if seconds_since_last_buy < self.cfg.min_seconds_between_buys:
            logger.debug(
                "Moonshot-hunter: cooldown actief (%.0fs/%ds sinds vorige bet), sla %s over.",
                seconds_since_last_buy, self.cfg.min_seconds_between_buys, mint,
            )
            return True

        open_positions_count = (
            self.outcome_tracker.open_position_count(strategy="moonshot_hunter")
            if self.outcome_tracker is not None else None
        )
        ok, reason = self.risk.can_trade(
            self.trade_size_sol, liquidity_sol, open_positions_count=open_positions_count,
            max_open_positions_override=self.cfg.max_open_positions,
            max_sol_per_trade_override=self.trade_size_sol,
        )
        if not ok:
            logger.info("Moonshot-hunter: trade geblokkeerd door risk manager: %s", reason)
            return True

        # see social_watch.py's identical wait for the full reasoning - top
        # up once, for whichever active filter (holder count or
        # concentration) needs the longer settling window, measured from
        # the token's own launch
        required_delay = 0.0
        if self.cfg.min_holder_count > 1:
            required_delay = max(required_delay, HOLDER_COUNT_INDEXING_DELAY_SEC)
        if self.cfg.max_top10_concentration_pct > 0:
            required_delay = max(required_delay, CONCENTRATION_SETTLING_DELAY_SEC)
        if required_delay > 0:
            elapsed_since_launch = time.time() - added_ts
            if elapsed_since_launch < required_delay:
                return False

        (
            (entry_ref, price_ref_field), holder_count, top10_concentration_pct, price_changes_pct,
        ) = await asyncio.gather(
            self._fetch_fresh_ref(mint),
            fetch_holder_count(mint, self.client.rpc_http_url),
            fetch_top10_concentration_pct(mint, self.client.rpc_http_url),
            fetch_price_changes_pct(mint),
        )
        price_change_5m_pct = price_changes_pct.get("m5") if price_changes_pct is not None else None
        if entry_ref is None:
            entry_ref, price_ref_field = extract_price_ref_with_field(event)

        if holder_count is None:
            logger.debug("Moonshot-hunter: kon holder count niet verifiëren voor %s, probeer opnieuw.", mint)
            if self.cfg.min_holder_count > 0:
                return False
        elif holder_count < self.cfg.min_holder_count:
            logger.info(
                "Moonshot-hunter: %s heeft %d holders, onder de min_holder_count van %d, "
                "probeer opnieuw volgende poll.", mint, holder_count, self.cfg.min_holder_count,
            )
            return False

        if self.cfg.max_top10_concentration_pct > 0:
            if top10_concentration_pct is None:
                logger.info("Moonshot-hunter: top-10 concentratie onbekend voor %s, sla over.", mint)
                return True
            if top10_concentration_pct > self.cfg.max_top10_concentration_pct:
                logger.info(
                    "Moonshot-hunter: %s heeft %.0f%% top-10 concentratie, boven de "
                    "max_top10_concentration_pct van %.0f%%, sla over.",
                    mint, top10_concentration_pct, self.cfg.max_top10_concentration_pct,
                )
                return True

        if self.cfg.min_price_change_5m_pct > 0:
            # INVERTED from every other strategy's ceiling - see
            # MoonshotHunterConfig's docstring for why: this REQUIRES
            # already-explosive momentum as the closest available proxy
            # for "this might be going viral right now"
            if price_change_5m_pct is None:
                logger.info("Moonshot-hunter: 5m prijsverandering onbekend voor %s, sla over.", mint)
                return True
            if price_change_5m_pct < self.cfg.min_price_change_5m_pct:
                logger.info(
                    "Moonshot-hunter: %s heeft %.1f%% prijsverandering (5m), onder de "
                    "min_price_change_5m_pct van %.0f%% (niet expliciet genoeg), sla over.",
                    mint, price_change_5m_pct, self.cfg.min_price_change_5m_pct,
                )
                return True

        await self.alerter.send(
            f"🚀 Moonshot-kandidaat: {name} ({symbol}) - {mint} "
            f"({holder_count} holders, {price_change_5m_pct:+.1f}% (5m)) - "
            f"zet in op {self.trade_size_sol} SOL, mikt op honderden x."
        )

        meta = {
            "liquidity_sol": liquidity_sol, "has_socials": True, "holder_count": holder_count,
            "price_change_m5_pct": price_change_5m_pct,
        }

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen (moonshot-hunter): %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            self._last_buy_ts = time.time()
            bot_state.log_trade("moonshot_hunter", "buy", mint, self.trade_size_sol, dry_run=True, meta=meta)
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="moonshot_hunter",
                    take_profit_ladder=self.cfg.take_profit_ladder,
                    price_ref_field=price_ref_field,
                    max_hold_sec=self.cfg.max_hold_sec,
                    stale_price_timeout_sec=self.cfg.stale_price_timeout_sec,
                )
            return True

        try:
            result = await self.client.build_and_send_trade(
                action="buy", mint=mint, amount_sol=self.trade_size_sol,
                slippage_pct=self.slippage_pct,
            )
            self.risk.register_trade_opened(self.trade_size_sol)
            await self.risk.report_buy_result(success=True)
            self._last_buy_ts = time.time()
            creator = event.get("traderPublicKey")
            bot_state.log_trade(
                "moonshot_hunter", "buy", mint, self.trade_size_sol,
                dry_run=False, tx_signature=result["signature"],
                meta={**meta, "creator": creator},
            )
            await self.alerter.send(f"✅ Gekocht (moonshot-hunter): {symbol} | tx: {result['signature']}")
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                    stop_loss_pct=self.cfg.stop_loss_pct,
                    trailing_activation_pct=self.cfg.trailing_activation_pct,
                    trailing_stop_pct=self.cfg.trailing_stop_pct,
                    strategy="moonshot_hunter",
                    take_profit_ladder=self.cfg.take_profit_ladder,
                    price_ref_field=price_ref_field,
                    max_hold_sec=self.cfg.max_hold_sec,
                    stale_price_timeout_sec=self.cfg.stale_price_timeout_sec,
                    metadata_uri=event.get("uri"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Moonshot-hunter buy mislukt voor %s: %s", mint, exc)
            await self.risk.report_buy_result(success=False)
            await self.alerter.send(f"❌ Moonshot-hunter buy mislukt voor {symbol}: {exc}")
        return True
