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
            await self._buy(mint, info["event"])

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

    async def _buy(self, mint: str, event: dict) -> None:
        name = event.get("name", "?")
        symbol = event.get("symbol", "?")
        liquidity_sol = event.get("vSolInBondingCurve")

        ok, reason = self.risk.can_trade(self.trade_size_sol, liquidity_sol)
        if not ok:
            logger.info("Social-watch: trade geblokkeerd door risk manager: %s", reason)
            return

        await self.alerter.send(
            f"👥 Social-watch kandidaat: {name} ({symbol}) - {mint} (socials gevonden)"
        )
        entry_ref = await self._fetch_fresh_ref(mint)
        if entry_ref is None:
            entry_ref = extract_price_ref(event)

        if self.dry_run:
            logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
            self.risk.register_trade_opened(self.trade_size_sol)
            bot_state.log_trade(
                "social_watch", "buy", mint, self.trade_size_sol, dry_run=True,
                meta={"liquidity_sol": liquidity_sol, "has_socials": True},
            )
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                )
            return

        try:
            result = await self.client.build_and_send_trade(
                action="buy", mint=mint, amount_sol=self.trade_size_sol,
                slippage_pct=self.slippage_pct,
            )
            self.risk.register_trade_opened(self.trade_size_sol)
            creator = event.get("traderPublicKey")
            bot_state.log_trade(
                "social_watch", "buy", mint, self.trade_size_sol,
                dry_run=False, tx_signature=result["signature"],
                meta={"liquidity_sol": liquidity_sol, "has_socials": True, "creator": creator},
            )
            await self.alerter.send(f"✅ Gekocht (social-watch): {symbol} | tx: {result['signature']}")
            if self.outcome_tracker is not None:
                await self.outcome_tracker.track(
                    mint, name, symbol, entry_ref, trade_size_sol=self.trade_size_sol,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Social-watch buy mislukt voor %s: %s", mint, exc)
            await self.alerter.send(f"❌ Social-watch buy mislukt voor {symbol}: {exc}")
