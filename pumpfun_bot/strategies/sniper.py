"""
Sniper: luistert naar nieuwe token-launches op pump.fun en koopt tokens die
door een paar simpele filters komen. Deze filters vangen NIET alle scams/
rugpulls - pump.fun launches zijn per ontwerp permissionless en risicovol.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from ..activity_log import DATA_LOG_PATH
from ..alerts import Alerter
from ..config import SniperConfig
from ..fees import PRIORITY_FEE_SOL_PER_LEG
from ..holder_concentration import fetch_top10_concentration_pct
from ..holder_count import record_holder_count
from ..outcome_tracker import OutcomeTracker
from ..price_ref import extract_price_ref_with_field
from ..pumpportal_client import OnChainTransactionError, PumpPortalClient
from ..risk import RiskManager
from .. import sniper_model
from ..state import bot_state
from ..wallet_reputation import blocked_wallets

logger = logging.getLogger("pumpfun_bot.sniper")

WALLET_BLOCKLIST_REFRESH_SEC = 60

# user-requested, real finding 2026-08-23: bought "Rogue Rocket (ROGROC)",
# and ~35s later a DIFFERENT mint launched under the exact same name and
# symbol (confirmed via logs - the second one was correctly rejected by
# min_buys_in_window, but only because it happened to have zero activity;
# nothing before this stopped the bot from buying a copycat with the same
# name outright). Reusing an identical name/symbol combo is a real, free,
# no-RPC-call scam signal - a legitimate project doesn't relaunch under
# its own name from scratch, but a copycat/rug-kit reusing a recognizable
# or trending name to catch bots/humans does.
#
# PERSISTED - confirmed live the SAME day: "Rogue Wizard" (ROGWIZ) was
# bought, and a THIRD relaunch of that exact name appeared ~55 HOURS after
# the first two (which themselves were only ~90s apart) - a scam kit
# clearly resurfaces a name over days, not just minutes, so an in-memory/
# short-window check isn't enough. Also noticed "Rogue Rocket" and "Rogue
# Wizard" share a "Rogue ___" template, consistent with the same actor/kit
# behind both.
SEEN_LAUNCH_NAMES_PATH = Path("data/sniper_seen_launch_names.json")
# user-requested 2026-08-23: loosened from no-expiry - confirmed live this
# was already the single biggest rejection category (38% of everything
# sniper saw in one short window), and a permanently-growing store means
# any common/generic meme name that gets coincidentally reused by
# unrelated people days or weeks later starts getting blocked too, not
# just deliberate copycats. 72h comfortably covers the confirmed 55h
# ROGWIZ gap with margin, while still letting old entries age out.
SEEN_LAUNCH_NAME_WINDOW_SEC = 72 * 3600

# Standard total supply of a pump.fun token - used to turn initialBuy (how
# many tokens the creator already bought in the creation tx itself) into a
# %. Ported from an earlier version of this bot's sniper - verify against a
# live launch if pump.fun ever changes this.
DEFAULT_TOTAL_SUPPLY = 1_000_000_000


def _load_seen_launch_names() -> dict[str, float]:
    if not SEEN_LAUNCH_NAMES_PATH.exists():
        return {}
    try:
        data = json.loads(SEEN_LAUNCH_NAMES_PATH.read_text())
        return dict(data) if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        logger.debug("Kon sniper_seen_launch_names.json niet lezen, start leeg.")
        return {}


def _record_seen_launch_name(key: str, existing: dict[str, float], now: float) -> None:
    existing[key] = now
    # prune anything past SEEN_LAUNCH_NAME_WINDOW_SEC on every write - bounded
    # file size without a separate cleanup pass
    expired = [k for k, ts in existing.items() if now - ts >= SEEN_LAUNCH_NAME_WINDOW_SEC]
    for k in expired:
        del existing[k]
    SEEN_LAUNCH_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_LAUNCH_NAMES_PATH.write_text(json.dumps(existing, indent=2))


class SniperStrategy:
    def __init__(
        self,
        client: PumpPortalClient,
        cfg: SniperConfig,
        risk: RiskManager,
        alerter: Alerter,
        trade_size_sol: float,
        slippage_pct: float,
        dry_run: bool,
        outcome_tracker: OutcomeTracker | None = None,
    ):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.alerter = alerter
        self.trade_size_sol = trade_size_sol
        self.slippage_pct = slippage_pct
        self.dry_run = dry_run
        self.outcome_tracker = outcome_tracker
        # wallets that our own trade history shows repeatedly launch losing
        # tokens - see wallet_reputation.py. Only ever grows (tighten-only,
        # same philosophy as auto_tuner.py), refreshed periodically below.
        self.blocked_wallets: set[str] = set()
        # (name, symbol) -> last-seen timestamp, persisted to disk with a
        # SEEN_LAUNCH_NAME_WINDOW_SEC expiry - see SEEN_LAUNCH_NAMES_PATH's
        # docstring
        self._seen_names: dict[str, float] = _load_seen_launch_names()
        # user-requested 2026-08-24: real buy-count observed by the most
        # recent _pre_buy_activity_check() call, for sniper_model.py's
        # shadow-mode scoring - None when the check was skipped (both
        # enable_bundle_check and min_buys_in_window disabled) or failed,
        # distinct from a genuinely-observed 0. See that method's docstring
        # for why this is a separate attribute rather than a return value -
        # avoids touching _pre_buy_activity_check's existing str|None return
        # type and the many tests asserting against it directly.
        self._last_activity_window_buy_count: int | None = None
        # user-requested 2026-08-24: cached for _passes_model_score_gate()
        # (see that method's docstring) - starts empty/None like
        # blocked_wallets above, populated by _refresh_background_state_loop
        # within WALLET_BLOCKLIST_REFRESH_SEC of startup. Rebuilding
        # creator_win_rates from the full activity log on EVERY candidate
        # (as _log_shadow_model_score already does, but only for successful
        # buys - much rarer) would cost real time on sniper's much
        # higher-volume pre-buy path, working against its whole speed edge.
        self._creator_win_rates: dict[str, float] = {}
        self._model: dict | None = None

    def _passes_filters(self, event: dict) -> bool:
        # PumpPortal new-token events bevatten o.a. mint, name, symbol, initial
        # liquidity, en (soms) social links. Veldnamen kunnen wijzigen - check dit
        # tegen een paar live events voordat je live gaat.
        liquidity_sol = event.get("vSolInBondingCurve") or event.get("initialBuy") or 0
        if liquidity_sol and liquidity_sol < self.risk.cfg.min_liquidity_sol:
            return False

        if self.cfg.require_socials:
            has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
            if not has_socials:
                return False

        # user-requested, ported from an earlier version of this bot's
        # sniper: free filter (no extra RPC call) - rejects a launch where
        # the creator's own initialBuy already claims too much of the
        # supply in the SAME transaction as the token's creation, often a
        # sign of a planned dump.
        if self.cfg.max_initial_buy_pct > 0:
            initial_buy_tokens = event.get("initialBuy")
            if initial_buy_tokens:
                initial_buy_pct = (initial_buy_tokens / DEFAULT_TOTAL_SUPPLY) * 100
                if initial_buy_pct > self.cfg.max_initial_buy_pct:
                    return False

        creator = event.get("traderPublicKey")
        if creator and creator in self.blocked_wallets:
            return False

        mint = event.get("mint")
        if (
            mint and self.outcome_tracker is not None
            and self.outcome_tracker.is_tracking(mint)
        ):
            # already held (e.g. social_watch bought this same mint) -
            # buying again would spend real SOL on a position nothing would
            # ever track or exit, since OutcomeTracker keys by mint alone
            return False

        return True

    def _pre_buy_model_score(self, meta: dict) -> float | None:
        """User-requested 2026-08-24: real gate on sniper_model.py's win-
        probability score, once corrected real trade data showed it beats
        baseline (66.13% vs 60.64% holdout) and that dead-on-arrival tokens
        (stale_price) are the single biggest real loss category. Only
        active when cfg.model_score_min_to_buy > 0 (default 0/off) - see
        that field's docstring in config.py for the full reasoning and the
        caution about still-modest training data.

        Uses the CACHED self._model/self._creator_win_rates (refreshed
        every WALLET_BLOCKLIST_REFRESH_SEC by _refresh_background_state_loop)
        rather than sniper_model.load_model()/build_creator_win_rates()
        fresh on every call, unlike _log_shadow_model_score below - this
        runs on every candidate that reaches this point (much higher
        volume than "successful buys only"), and rebuilding creator_win_rates
        from the whole activity log on each one would cost real time on
        sniper's speed-critical path.

        Returns None (never gates) if no model is cached yet or the
        candidate is missing a required raw feature - fails OPEN, same
        philosophy as sniper's other checks (holder_concentration,
        activity-window)."""
        if self._model is None:
            return None
        return sniper_model.score(meta, self._creator_win_rates, model=self._model)

    def _log_shadow_model_score(self, mint: str, symbol: str, meta: dict) -> None:
        """User-requested: computes and logs sniper_model.py's win-
        probability score for this real buy, WITHOUT touching the buy
        decision itself (already made and executed by the time this
        runs) - see sniper_model.py's module docstring for why this stays
        shadow-mode-only until there's enough labeled real-trade history
        to validate it against the existing hard filters. Any failure here
        (no model trained yet, a bad read) must never affect the strategy
        - caught and logged, not raised."""
        try:
            model = sniper_model.load_model()
            if model is None:
                return
            creator_win_rates = sniper_model.build_creator_win_rates(DATA_LOG_PATH)
            win_probability = sniper_model.score(meta, creator_win_rates, model=model)
            if win_probability is None:
                return
            logger.info(
                "Sniper: model score voor %s (%s) = %.2f (schaduwmodus, "
                "heeft geen invloed op de koopbeslissing).",
                symbol, mint, win_probability,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Kon shadow-mode model score niet berekenen voor %s.", mint)

    async def _refresh_background_state_loop(self) -> None:
        while True:
            await asyncio.sleep(WALLET_BLOCKLIST_REFRESH_SEC)
            newly_blocked = blocked_wallets(DATA_LOG_PATH) - self.blocked_wallets
            if newly_blocked:
                self.blocked_wallets |= newly_blocked
                message = (
                    f"🧠 Wallet-reputatie: {len(newly_blocked)} launcher-wallet(s) geblokkeerd "
                    f"na herhaalde verliezen: {', '.join(sorted(newly_blocked))}"
                )
                logger.warning(message)
                await self.alerter.send(message)

            if self.cfg.model_score_min_to_buy > 0:
                try:
                    self._creator_win_rates = sniper_model.build_creator_win_rates(DATA_LOG_PATH)
                    self._model = sniper_model.load_model()
                except Exception:  # noqa: BLE001
                    logger.exception("Kon model/creator win-rates niet verversen voor de model-gate.")

    async def _holder_concentration_flags_risk(self, mint: str) -> bool:
        """user-requested, ported from an earlier version of this bot's
        sniper. CAUTION (see max_top10_concentration_pct's docstring in
        config.py): sniper checks this within seconds of launch, before
        holder_concentration.py's own SETTLING_DELAY_SEC has had a chance to
        pass - unlike social_watch/birdeye_movers/coingecko_movers, which
        only ever reach this check well after that delay. Costs one extra
        RPC round-trip; only called at all when enabled via a > 0 threshold."""
        if self.cfg.max_top10_concentration_pct <= 0:
            return False
        top10_concentration_pct = await fetch_top10_concentration_pct(mint, self.client.rpc_http_url)
        if top10_concentration_pct is None:
            return False
        return top10_concentration_pct > self.cfg.max_top10_concentration_pct

    def _is_duplicate_name(self, name: str, symbol: str) -> bool:
        """See SEEN_LAUNCH_NAMES_PATH's docstring - a free, no-RPC-call
        check against every (name, symbol) sniper has seen in the last
        SEEN_LAUNCH_NAME_WINDOW_SEC, persisted to disk (a scam kit reuses a
        name across days, not just minutes - a short in-memory window
        isn't enough, confirmed live). Records the CURRENT candidate
        regardless of outcome (even a rejected one) so a run of copycats
        reusing the same name all get caught, not just the first repeat."""
        key = f"{name.strip().lower()}|{symbol.strip().lower()}"
        # a missing name/symbol (event.get(..., "?") in run()) must never
        # match itself across different real launches - that would flag
        # every subsequent placeholder-named candidate as a false duplicate
        if name.strip() == "?" or symbol.strip() == "?" or not name.strip() or not symbol.strip():
            return False
        now = time.time()
        last_seen = self._seen_names.get(key)
        is_duplicate = last_seen is not None and now - last_seen < SEEN_LAUNCH_NAME_WINDOW_SEC
        _record_seen_launch_name(key, self._seen_names, now)
        return is_duplicate

    async def _pre_buy_activity_check(self, mint: str) -> str | None:
        """Watches the token's live trade stream for bundle_check_window_ms
        right after launch, in ONE pass, for two opposite failure modes:

        - too MANY buys (bundle_check_max_buys) - user-requested, ported
          from an earlier version of this bot's sniper: a sign of
          coordinated insider wallets all buying at once.
        - too FEW buys (min_buys_in_window) - user-requested, real finding
          2026-08-23: 57.1% of everything sniper bought went stale_price
          (zero real trade activity) within seconds, and min_liquidity_sol
          can't catch this since every pump.fun launch starts at
          essentially the same bonding-curve liquidity (confirmed live: 391
          of ~401 real buys all fell in the same 30-40 SOL band - a
          constant, not a signal). Checked in the SAME window-watch instead
          of a second one, so this doesn't cost sniper a second window's
          worth of real time on top of the bundle check.

        Returns None if the candidate passes, or a short reason string if
        it should be rejected. Deliberately costs the whole window in real
        time when either check is enabled - a conscious trade of sniper's
        speed advantage for these signals, which is why both are off by
        default."""
        self._last_activity_window_buy_count = None
        if not self.cfg.enable_bundle_check and self.cfg.min_buys_in_window <= 0:
            return None

        count = 0
        deadline = time.monotonic() + (self.cfg.bundle_check_window_ms / 1000)
        stream = self.client.stream_token_trades([mint])
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    event = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except (asyncio.TimeoutError, StopAsyncIteration):
                    break
                if event.get("txType") == "buy":
                    count += 1
                if self.cfg.enable_bundle_check and count > self.cfg.bundle_check_max_buys:
                    self._last_activity_window_buy_count = count
                    return "gebundeld (te veel snelle buys)"
        except Exception:  # noqa: BLE001
            logger.exception("Activiteit-check mislukt voor %s - filter wordt overgeslagen.", mint)
            return None
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass

        self._last_activity_window_buy_count = count
        if self.cfg.enable_bundle_check and count > self.cfg.bundle_check_max_buys:
            return "gebundeld (te veel snelle buys)"
        if self.cfg.min_buys_in_window > 0 and count < self.cfg.min_buys_in_window:
            return f"geen echte koopactiviteit ({count}/{self.cfg.min_buys_in_window} buys in het venster)"
        return None

    async def run(self) -> None:
        if not self.cfg.enabled:
            return
        logger.info("Sniper strategie gestart (dry_run=%s).", self.dry_run)
        if self.cfg.model_score_min_to_buy > 0:
            try:
                self._creator_win_rates = sniper_model.build_creator_win_rates(DATA_LOG_PATH)
                self._model = sniper_model.load_model()
            except Exception:  # noqa: BLE001
                logger.exception("Kon model/creator win-rates niet initieel laden voor de model-gate.")
        asyncio.create_task(self._refresh_background_state_loop())

        async for event in self.client.stream_new_tokens():
            mint = event.get("mint")
            name = event.get("name", "?")
            symbol = event.get("symbol", "?")
            if not mint:
                continue

            if not self._passes_filters(event):
                logger.debug("Token %s (%s) afgewezen door filters.", symbol, mint)
                continue

            if self._is_duplicate_name(name, symbol):
                logger.info(
                    "Sniper: %s (%s) geweerd - zelfde naam/symbool recent al gezien "
                    "(mogelijk kopie/scam).", name, symbol,
                )
                continue

            reject_reason = await self._pre_buy_activity_check(mint)
            if reject_reason is not None:
                logger.info("Sniper: %s (%s) geweerd - %s.", name, symbol, reject_reason)
                continue

            if await self._holder_concentration_flags_risk(mint):
                logger.info("Sniper: %s (%s) geweerd - houder-concentratie te hoog.", name, symbol)
                continue

            liquidity_sol = event.get("vSolInBondingCurve")
            entry_ref, price_ref_field = extract_price_ref_with_field(event)
            open_positions_count = (
                self.outcome_tracker.open_position_count() if self.outcome_tracker is not None else None
            )
            # user-requested: logged for sniper_model.py's shadow-mode
            # scoring - the SAME free, no-RPC-call signal
            # max_initial_buy_pct already computes above, just not
            # previously logged anywhere past the filter check itself
            initial_buy_tokens = event.get("initialBuy")
            initial_buy_pct = (
                (initial_buy_tokens / DEFAULT_TOTAL_SUPPLY) * 100 if initial_buy_tokens else None
            )
            creator = event.get("traderPublicKey")

            if self.cfg.model_score_min_to_buy > 0:
                score = self._pre_buy_model_score({
                    "liquidity_sol": liquidity_sol, "creator": creator,
                    "initial_buy_pct": initial_buy_pct,
                    "activity_window_buy_count": self._last_activity_window_buy_count,
                })
                if score is not None and score < self.cfg.model_score_min_to_buy:
                    logger.info(
                        "Sniper: %s (%s) geweerd - model score te laag (%.2f < %.2f).",
                        name, symbol, score, self.cfg.model_score_min_to_buy,
                    )
                    continue

            ok, reason = self.risk.can_trade(
                self.trade_size_sol, liquidity_sol, open_positions_count=open_positions_count,
            )
            if not ok:
                logger.info("Sniper: trade geblokkeerd door risk manager: %s", reason)
                continue

            msg = f"🎯 Snipe kandidaat: {name} ({symbol}) - {mint}"
            await self.alerter.send(msg)

            if self.dry_run:
                logger.info("[DRY RUN] Zou kopen: %s SOL van %s", self.trade_size_sol, mint)
                self.risk.register_trade_opened(self.trade_size_sol)
                has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
                bot_state.log_trade(
                    "sniper", "buy", mint, self.trade_size_sol, dry_run=True,
                    meta={
                        "liquidity_sol": liquidity_sol, "has_socials": has_socials,
                        "creator": creator, "initial_buy_pct": initial_buy_pct,
                        "activity_window_buy_count": self._last_activity_window_buy_count,
                    },
                )
                asyncio.create_task(record_holder_count(mint, self.client.rpc_http_url))
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, entry_ref,
                        trade_size_sol=self.trade_size_sol,
                        take_profit_pct=self.cfg.take_profit_pct,
                        stop_loss_pct=self.cfg.stop_loss_pct,
                        trailing_activation_pct=self.cfg.trailing_activation_pct,
                        trailing_stop_pct=self.cfg.trailing_stop_pct,
                        take_profit_ladder=self.cfg.take_profit_ladder,
                        price_ref_field=price_ref_field,
                    )
                continue

            try:
                result = await self.client.build_and_send_trade(
                    action="buy",
                    mint=mint,
                    amount_sol=self.trade_size_sol,
                    slippage_pct=self.slippage_pct,
                )
                self.risk.register_trade_opened(self.trade_size_sol)
                await self.risk.report_buy_result(success=True)
                has_socials = any(event.get(k) for k in ("twitter", "telegram", "website"))
                bot_state.log_trade(
                    "sniper", "buy", mint, self.trade_size_sol,
                    dry_run=False, tx_signature=result["signature"],
                    meta={
                        "liquidity_sol": liquidity_sol, "has_socials": has_socials,
                        "creator": creator, "initial_buy_pct": initial_buy_pct,
                        "activity_window_buy_count": self._last_activity_window_buy_count,
                    },
                )
                asyncio.create_task(record_holder_count(mint, self.client.rpc_http_url))
                self._log_shadow_model_score(mint, symbol, {
                    "liquidity_sol": liquidity_sol, "creator": creator,
                    "initial_buy_pct": initial_buy_pct,
                    "activity_window_buy_count": self._last_activity_window_buy_count,
                })
                await self.alerter.send(f"✅ Gekocht: {symbol} | tx: {result['signature']}")
                if self.outcome_tracker is not None:
                    await self.outcome_tracker.track(
                        mint, name, symbol, entry_ref,
                        trade_size_sol=self.trade_size_sol,
                        take_profit_pct=self.cfg.take_profit_pct,
                        stop_loss_pct=self.cfg.stop_loss_pct,
                        trailing_activation_pct=self.cfg.trailing_activation_pct,
                        trailing_stop_pct=self.cfg.trailing_stop_pct,
                        take_profit_ladder=self.cfg.take_profit_ladder,
                        price_ref_field=price_ref_field,
                        metadata_uri=event.get("uri"),
                        buy_tx_signature=result["signature"],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Snipe buy mislukt voor %s: %s", mint, exc)
                await self.risk.report_buy_result(success=False)
                if isinstance(exc, OnChainTransactionError):
                    # user-requested 2026-08-24 ("the actual profit is not
                    # right"): a buy that fails ON-CHAIN (not e.g. a
                    # network/RPC error before the transaction ever landed)
                    # still pays a real priority fee - register_trade_opened
                    # was never called (still inside the try above it), so
                    # there's no exposure to release, only a real cost to
                    # record. Estimate (the configured priority fee, not the
                    # exact on-chain fee - avoids an extra RPC round-trip on
                    # the failure path) - realized_pnl_sol never accounted
                    # for this at all before, looking better than reality by
                    # the sum of every failed buy's fee.
                    self.risk.register_trade_closed(0.0, -PRIORITY_FEE_SOL_PER_LEG)
                await self.alerter.send(f"❌ Snipe mislukt voor {symbol}: {exc}")
