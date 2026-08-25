"""
Wrapper rond de (community) PumpPortal API voor pump.fun.

LET OP: dit is een third-party service, niet officieel van pump.fun of Solana.
Endpoints en payload-formaten kunnen wijzigen - check altijd de actuele docs op
https://pumpportal.fun/ voordat je hiermee live handelt. Dit bestand is een
werkend startpunt, geen garantie dat elk veld exact klopt met de huidige API.

De "Local Trading API" laat je zelf de transactie signen en submitten (jouw
private key verlaat je machine nooit richting PumpPortal), wat veiliger is dan
de "Lightning API" waarbij zij namens jou signen.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import AsyncIterator, Callable

import aiohttp
import websockets
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

from . import jito
from .fees import PRIORITY_FEE_SOL_PER_LEG

logger = logging.getLogger("pumpfun_bot.pumpportal")


class OnChainTransactionError(RuntimeError):
    """Raised by _confirm_transaction when a transaction reverted on-chain
    - a plain RuntimeError subclass (existing `except Exception`/
    `assertRaises(RuntimeError)` callers are unaffected), but carries the
    parsed Anchor custom error code (e.g. 6005) when the failure has one,
    so a caller that cares about a SPECIFIC on-chain reason (see
    build_and_send_full_sell's pool="pump-amm" retry) doesn't have to
    string-match the stringified error dict."""

    def __init__(self, message: str, custom_error_code: int | None = None):
        super().__init__(message)
        self.custom_error_code = custom_error_code


def _extract_custom_error_code(err) -> int | None:
    if not isinstance(err, dict):
        return None
    instr_err = err.get("InstructionError")
    if not (isinstance(instr_err, list) and len(instr_err) == 2 and isinstance(instr_err[1], dict)):
        return None
    return instr_err[1].get("Custom")


def authenticated_ws_url(ws_url: str, api_key: str) -> str:
    # subscribeTokenTrade/subscribeAccountTrade need the key on the connection
    # URL itself (?api-key=...), not in a message - see
    # https://pumpportal.fun/data-api/real-time
    if not api_key:
        return ws_url
    separator = "&" if "?" in ws_url else "?"
    return f"{ws_url}{separator}api-key={api_key}"


class PumpPortalClient:
    def __init__(
        self,
        ws_url: str,
        trade_api_url: str,
        rpc_http_url: str,
        keypair: Keypair,
        api_key: str = "",
    ):
        self.ws_url = ws_url
        self.trade_api_url = trade_api_url
        self.rpc_http_url = rpc_http_url
        self.keypair = keypair
        self.api_key = api_key

    # ---------- Data feed (WebSocket) ----------

    async def stream_new_tokens(self) -> AsyncIterator[dict]:
        """Yield events voor elk nieuw gelanceerd token op pump.fun."""
        async for event in self._stream(subscribe_method="subscribeNewToken"):
            yield event

    async def stream_wallet_trades(self, wallets: list[str]) -> AsyncIterator[dict]:
        """Yield events wanneer een van de gevolgde wallets een trade doet."""
        async for event in self._stream(
            subscribe_method="subscribeAccountTrade", keys=wallets
        ):
            yield event

    async def stream_token_trades(self, mints: list[str]) -> AsyncIterator[dict]:
        """Yield events voor trades op specifieke tokens (handig voor market making)."""
        async for event in self._stream(
            subscribe_method="subscribeTokenTrade", keys=mints
        ):
            yield event

    async def _stream(
        self, subscribe_method: str, keys: list[str] | None = None
    ) -> AsyncIterator[dict]:
        payload: dict = {"method": subscribe_method}
        if keys:
            payload["keys"] = keys

        ws_url = authenticated_ws_url(self.ws_url, self.api_key)
        async for websocket in websockets.connect(ws_url, ping_interval=20):
            try:
                await websocket.send(json.dumps(payload))
                logger.info("WS verbonden en geabonneerd op %s", subscribe_method)
                async for raw in websocket:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug("Kon WS bericht niet parsen: %s", raw)
                        continue
                    yield data
            except websockets.ConnectionClosed:
                logger.warning("WebSocket verbinding verbroken, reconnect...")
                continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Onverwachte WS fout: %s", exc)
                continue

    # ---------- Trading (Local Trading API) ----------

    async def build_and_send_trade(
        self,
        action: str,          # "buy" of "sell"
        mint: str,
        amount_sol: float,
        slippage_pct: float,
        priority_fee_sol: float = PRIORITY_FEE_SOL_PER_LEG,
        pool: str = "auto",
    ) -> dict:
        """
        Vraagt een ongesigneerde transactie op bij PumpPortal, signeert lokaal met
        onze eigen keypair, en stuurt hem naar de Solana RPC. Private key verlaat
        nooit dit process.

        amount_sol is altijd SOL-gedenomineerd (denominatedInSol: true) - voor
        buys "koop voor X SOL", voor sells "verkoop tokens ter waarde van X SOL".
        Voor een volledige exit (verkoop 100% van de holding) gebruik je
        build_and_send_full_sell() - "amount: X SOL" bij een sell verkoopt NIET
        per se de hele positie, alleen tokens ter waarde van X SOL tegen de
        actuele prijs.

        pool="auto" (was hardcoded "pump" - bonding curve only): PumpPortal
        supports "pump" (bonding curve), "pump-amm" (PumpSwap, where
        pump.fun tokens graduate to by default since March 2025), and
        several external DEXes, with "auto" routing to wherever the token
        actually trades right now - a real limitation of hardcoding "pump"
        for any position that outlives its own migration. Tested directly
        against PumpPortal tonight, though: this was NOT what caused the
        "400 Bad Request" on birdeye_movers' Truth Coin/OpenAI PreStocks
        candidates - every pool value failed identically for those two.
        Root cause turned out to be simpler: neither mint ends in "pump"
        (pump.fun's vanity-address convention for every mint it launches),
        meaning Birdeye's Solana-wide trending list surfaced two tokens
        that were never pump.fun tokens at all - PumpPortal correctly has
        no route for them regardless of pool. See birdeye_movers.py's
        is_pump_fun_mint() for the actual fix. "auto" is kept anyway since
        it's a strict superset of "pump" per PumpPortal's own docs and
        costs nothing, but it does not explain tonight's failures.
        """
        body = {
            "publicKey": str(self.keypair.pubkey()),
            "action": action,
            "mint": mint,
            "amount": amount_sol,
            "denominatedInSol": "true",
            "slippage": slippage_pct,
            "priorityFee": priority_fee_sol,
            "pool": pool,
        }
        sig = await self._sign_and_send_with_migration_retry(body)
        # only fetched for sells - a buy is speed-critical (sniper's whole
        # edge depends on it) and doesn't need this, see _fetch_real_sol_
        # delta's docstring for what it's actually used for (real, ground-
        # truth pnl on a SELL, not buy-side accounting)
        real_sol_delta = await self._fetch_real_sol_delta(sig) if action == "sell" else None
        return {
            "signature": sig, "action": action, "mint": mint, "amount_sol": amount_sol,
            "real_sol_delta": real_sol_delta,
        }

    async def build_and_send_trade_via_jito_bundle(
        self,
        action: str,
        mint: str,
        amount_sol: float,
        slippage_pct: float,
        priority_fee_sol: float = PRIORITY_FEE_SOL_PER_LEG,
        tip_sol: float = jito.RECOMMENDED_TIP_SOL,
        pool: str = "auto",
        block_engine_url: str = jito.DEFAULT_BLOCK_ENGINE_URL,
        bundle_timeout_sec: float = jito.BUNDLE_STATUS_TIMEOUT_SEC,
    ) -> dict:
        """
        User-requested 2026-08-25 ("how can i execute the bot faster" ->
        "both") - same real trade as build_and_send_trade, but submitted
        as a Jito bundle. Mirrors build_and_send_full_sell_via_jito_bundle
        exactly (see that method's docstring for the full story of how
        this bundle-submission approach was arrived at - PumpPortal's own
        array-mode endpoint 400s on real trades, so this fetches the
        unsigned tx through the SAME proven single-object endpoint
        build_and_send_trade already uses, signs it locally, and submits
        it alongside a second, independent tip-only transaction built by
        jito.build_tip_transaction - never touching or modifying
        PumpPortal's own transaction).

        tip_sol is a real, separate cost from priority_fee_sol, paid only
        when this path is used - see priority_fee_sol_for_sell's sibling
        reasoning in fees.py. Sized the same as the sell-side default
        (jito.RECOMMENDED_TIP_SOL, empirically found live to actually
        land a bundle) since the tip requirement is a Jito protocol
        behavior, not specific to which side of a trade it's attached to.

        Raises RuntimeError/OnChainTransactionError on any failure, same
        contract as build_and_send_full_sell_via_jito_bundle - the
        caller's existing retry/failure handling treats this the same as
        any other failed buy attempt.
        """
        body = {
            "publicKey": str(self.keypair.pubkey()),
            "action": action,
            "mint": mint,
            "amount": amount_sol,
            "denominatedInSol": "true",
            "slippage": slippage_pct,
            "priorityFee": priority_fee_sol,
            "pool": pool,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.trade_api_url, json=body, timeout=15) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"PumpPortal trade-local fout ({resp.status}): {text}")
                raw_tx_bytes = await resp.read()

        tx = VersionedTransaction.from_bytes(raw_tx_bytes)
        signed_tx = VersionedTransaction(tx.message, [self.keypair])
        sig = str(signed_tx.signatures[0])

        tip_lamports = round(tip_sol * 1_000_000_000)
        tip_tx = jito.build_tip_transaction(self.keypair, tip_lamports, tx.message.recent_blockhash)

        bundle_id = await jito.send_bundle([bytes(signed_tx), bytes(tip_tx)], block_engine_url)
        if bundle_id is None:
            raise RuntimeError(f"Jito bundle-verzending mislukt voor {mint} ({action}).")

        status = await jito.poll_bundle_until_landed(
            bundle_id, block_engine_url, timeout_sec=bundle_timeout_sec,
        )
        if status is None:
            raise RuntimeError(
                f"Jito bundle {bundle_id} niet bevestigd binnen {bundle_timeout_sec}s voor {mint} "
                f"({action}) - onder-getipt, of geen validator heeft hem opgepikt."
            )
        err = status.get("err")
        if err not in (None, {"Ok": None}):
            raise OnChainTransactionError(
                f"Jito bundle {bundle_id} is gefaald on-chain: {err}",
                custom_error_code=_extract_custom_error_code(err),
            )

        real_sol_delta = await self._fetch_real_sol_delta(sig) if action == "sell" else None
        return {
            "signature": sig, "action": action, "mint": mint, "amount_sol": amount_sol,
            "real_sol_delta": real_sol_delta, "bundle_id": bundle_id,
        }

    async def build_and_send_full_sell(
        self,
        mint: str,
        slippage_pct: float,
        priority_fee_sol: float = PRIORITY_FEE_SOL_PER_LEG,
        pool: str = "auto",
        amount_pct: float = 100,
    ) -> dict:
        """
        Verkoopt amount_pct% van wat deze wallet aan `mint` in bezit heeft -
        voor een volledige exit (take-profit/stop-loss/timeout), niet een
        gedeeltelijke SOL-gedenomineerde sell. Gebruikt amount: "{pct}%" met
        denominatedInSol: "false", zoals PumpPortal's Local Trading API docs
        beschrijven (https://pumpportal.fun/local-trading-api/trading-api).

        pool="auto" (was hardcoded "pump") - see build_and_send_trade's
        docstring for the full reasoning and what it does/doesn't explain.
        Tested directly against the 3 real long-stuck positions tonight
        (585UP, TOXCOM, $MAMA): PumpPortal builds a valid transaction for
        all three regardless of "pump" vs "auto" - both succeed at the
        build step identically, so the Custom 6022/6024 on-chain failures
        those hit are a separate, still-unexplained issue, NOT pool
        routing. Kept "auto" as the default anyway (strictly more capable,
        costs nothing), just not overselling what it fixes.

        amount_pct: default 100 (unchanged behavior). Confirmed live via
        getTransaction logs that these same 3 stuck positions ALL fail with
        an AnchorError thrown in programs/pump/src/lib.rs:801 ("Overflow",
        Custom 6024) right after a successful GetFees sub-call - consistent
        with an edge case in fully-liquidating a position, not a slippage
        or balance-index problem (see outcome_tracker.py's _exit(), which
        retries at amount_pct=99 once a position hits
        MAX_CONSECUTIVE_SELL_FAILURES at 100%, before giving up entirely).
        """
        amount = f"{amount_pct:g}%"
        body = {
            "publicKey": str(self.keypair.pubkey()),
            "action": "sell",
            "mint": mint,
            "amount": amount,
            "denominatedInSol": "false",
            "slippage": slippage_pct,
            "priorityFee": priority_fee_sol,
            "pool": pool,
        }
        sig = await self._sign_and_send_with_migration_retry(body)
        real_sol_delta = await self._fetch_real_sol_delta(sig)
        return {
            "signature": sig, "action": "sell", "mint": mint, "amount": amount,
            "real_sol_delta": real_sol_delta,
        }

    async def build_and_send_full_sell_via_jito_bundle(
        self,
        mint: str,
        slippage_pct: float,
        priority_fee_sol: float = PRIORITY_FEE_SOL_PER_LEG,
        tip_sol: float = jito.RECOMMENDED_TIP_SOL,
        pool: str = "auto",
        amount_pct: float = 100,
        block_engine_url: str = jito.DEFAULT_BLOCK_ENGINE_URL,
        bundle_timeout_sec: float = jito.BUNDLE_STATUS_TIMEOUT_SEC,
    ) -> dict:
        """
        User-requested 2026-08-24 ("is there still autonomous learning" ->
        ... -> "yes build if you think it will make the bot better") -
        same real sell as build_and_send_full_sell, but submitted as a
        Jito bundle instead of a normal RPC sendTransaction, so it either
        lands atomically in the slot a validator accepts the tip for, or
        doesn't land at all - closing the trigger-to-landed-fill gap this
        session's own real data (see fees.py's
        DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON) showed is the dominant
        real cost on this bot's trades.

        tip_sol is a SEPARATE cost from priority_fee_sol (user-requested
        2026-08-24, "try" -> "build it": originally reused priority_fee_sol
        for both, confirmed live that a real bundle only landed at 0.01 SOL,
        4x the normal take_profit priority fee - conflating the two would
        have meant either overpaying the sell's own priority fee to get a
        usable tip, or under-tipping to keep the fee normal. priority_fee_sol
        still sets the sell transaction's own compute-budget priority fee,
        needed regardless of Jito; tip_sol is only spent when this path is
        actually used, and not optional - an under-tipped bundle simply
        never gets included by any validator, silently losing the trade to
        a normal sendTransaction competitor instead of costing extra.

        Real bug found live 2026-08-24, hours after this shipped: initially
        requested the unsigned tx via PumpPortal's documented array-mode
        /api/trade-local (their own Jito-bundles docs show this - post an
        array of trade objects instead of one, response is base58-encoded
        instead of raw bytes). Confirmed live on 2 real trades ("80K Bull",
        "BULL Token") this consistently 400'd - added real retry delay to
        exactly the exits meant to be fast, before falling back to the
        existing 99% normal-path fallback. Reproduced directly: PumpPortal's
        array-mode endpoint 400s on action="sell" unconditionally - any
        amount format, any pool value, a REAL held mint, even a bare 2-item
        self-bundle - while action="buy" in array mode returns 200 fine.
        An undocumented limitation/bug on PumpPortal's side, not fixable by
        changing what we send them.

        Fixed by not needing their bundle endpoint AT ALL: Jito's sendBundle
        only needs already-signed transactions, with zero requirement on
        how they were built. Fetches the unsigned tx through the SAME
        proven single-object endpoint build_and_send_full_sell already
        uses (confirmed live: reliably 200s for sells), signs it locally
        exactly the same way, and submits that one signed tx to Jito
        directly as a bundle of one - PumpPortal's involvement ends at
        "hand me an unsigned transaction," same as every other real-sell
        path in this client.

        Raises RuntimeError if PumpPortal's own build step failed, the
        bundle was never even accepted for submission, never confirmed
        within bundle_timeout_sec, or landed with an on-chain error - the
        caller's existing retry/cooldown handling (see outcome_tracker.py's
        _exit) already treats any exception here as "sell failed, try
        again shortly", same as every other real-sell path in this client.
        """
        amount = f"{amount_pct:g}%"
        body = {
            "publicKey": str(self.keypair.pubkey()),
            "action": "sell",
            "mint": mint,
            "amount": amount,
            "denominatedInSol": "false",
            "slippage": slippage_pct,
            "priorityFee": priority_fee_sol,
            "pool": pool,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.trade_api_url, json=body, timeout=15) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"PumpPortal trade-local fout ({resp.status}): {text}")
                raw_tx_bytes = await resp.read()

        tx = VersionedTransaction.from_bytes(raw_tx_bytes)
        signed_tx = VersionedTransaction(tx.message, [self.keypair])
        sig = str(signed_tx.signatures[0])

        # user-requested 2026-08-24 ("built" - the real fix after Jito
        # rejected an untipped bundle live) - a second, from-scratch
        # transaction whose only job is paying the tip, same recent
        # blockhash as the sell tx above, submitted together in one
        # bundle. See jito.py's module docstring for why this doesn't
        # touch the sell transaction itself.
        tip_lamports = round(tip_sol * 1_000_000_000)
        tip_tx = jito.build_tip_transaction(
            self.keypair, tip_lamports, tx.message.recent_blockhash,
        )

        bundle_id = await jito.send_bundle([bytes(signed_tx), bytes(tip_tx)], block_engine_url)
        if bundle_id is None:
            raise RuntimeError(f"Jito bundle-verzending mislukt voor {mint} (verkoop).")

        status = await jito.poll_bundle_until_landed(
            bundle_id, block_engine_url, timeout_sec=bundle_timeout_sec,
        )
        if status is None:
            raise RuntimeError(
                f"Jito bundle {bundle_id} niet bevestigd binnen {bundle_timeout_sec}s voor {mint} "
                f"(verkoop) - onder-getipt, of geen validator heeft hem opgepikt."
            )
        err = status.get("err")
        if err not in (None, {"Ok": None}):
            raise OnChainTransactionError(
                f"Jito bundle {bundle_id} is gefaald on-chain: {err}",
                custom_error_code=_extract_custom_error_code(err),
            )

        real_sol_delta = await self._fetch_real_sol_delta(sig)
        return {
            "signature": sig, "action": "sell", "mint": mint, "amount": amount,
            "real_sol_delta": real_sol_delta, "bundle_id": bundle_id,
        }

    async def _sign_and_send_with_migration_retry(self, body: dict) -> str:
        """User-requested, real finding 2026-08-23: a sell failed with
        AnchorError BondingCurveComplete (Custom 6005) - "The bonding curve
        has completed and liquidity migrated to raydium." - even with
        pool="auto", which is supposed to route wherever the token
        currently trades. A token can graduate WHILE this bot holds it
        (confirmed live: a real position pumped hard enough to migrate
        mid-hold), and "auto" doesn't always catch a migration that just
        happened. Only retries a SELL (a buy into an already-migrated
        token is a different, pre-buy problem "auto" already claims to
        handle) that isn't already explicitly on pool="pump-amm" - one
        retry, forcing the PumpSwap pool explicitly since we now KNOW
        definitively (from the error itself) that the bonding curve is
        done and the token only trades there now."""
        try:
            return await self._sign_and_send(body)
        except OnChainTransactionError as exc:
            if (
                exc.custom_error_code != 6005
                or body.get("action") != "sell"
                or body.get("pool") == "pump-amm"
            ):
                raise
            logger.warning(
                "Verkoop van %s mislukte met BondingCurveComplete (Custom 6005) - bonding curve "
                "is gemigreerd naar Raydium/PumpSwap, probeer opnieuw met pool=pump-amm.",
                body.get("mint"),
            )
            retry_body = {**body, "pool": "pump-amm"}
            return await self._sign_and_send(retry_body)

    async def _sign_and_send(self, body: dict) -> str:
        # timing + blockhash-validity diagnostics for BlockhashNotFound
        # failures: separates "PumpPortal handed us an already-stale
        # blockhash" from "it went stale during our own processing/submission"
        t_start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.post(self.trade_api_url, json=body, timeout=15) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"PumpPortal trade-local fout ({resp.status}): {text}")
                raw_tx_bytes = await resp.read()
        t_built = time.monotonic()

        tx = VersionedTransaction.from_bytes(raw_tx_bytes)
        blockhash = str(tx.message.recent_blockhash)
        signed_tx = VersionedTransaction(tx.message, [self.keypair])
        t_signed = time.monotonic()

        blockhash_valid = await self._check_blockhash_valid(blockhash)
        t_checked = time.monotonic()

        def log_timing(outcome: str, t_end: float) -> None:
            logger.info(
                "Trade timing (%s): pumpportal_build=%.0fms sign=%.0fms "
                "blockhash_check=%.0fms rest=%.0fms total=%.0fms "
                "blockhash=%s valid_right_after_receiving_it=%s",
                outcome,
                (t_built - t_start) * 1000,
                (t_signed - t_built) * 1000,
                (t_checked - t_signed) * 1000,
                (t_end - t_checked) * 1000,
                (t_end - t_start) * 1000,
                blockhash,
                blockhash_valid,
            )

        try:
            sig = await self._send_raw_transaction(bytes(signed_tx))
            # sendTransaction only confirms the tx was ACCEPTED for
            # processing, not that it actually executed - with skipPreflight
            # on (needed to avoid rejecting valid-but-just-issued blockhashes,
            # see _check_blockhash_valid above) there's no other check left
            # that would catch a transaction that reverts on-chain (e.g.
            # slippage tolerance exceeded), so without this a failed sell
            # would silently be treated as a successful one
            await self._confirm_transaction(sig)
        except Exception:
            log_timing("FAILED", time.monotonic())
            raise
        log_timing("OK", time.monotonic())
        logger.info("Transactie verstuurd: %s", sig)
        return sig

    async def _check_blockhash_valid(self, blockhash: str) -> bool | None:
        """Asks our own RPC whether this blockhash is still valid right now -
        None if the check itself fails (doesn't block the actual trade)."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "isBlockhashValid",
            "params": [blockhash, {"commitment": "processed"}],
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rpc_http_url, json=payload, timeout=10) as resp:
                    data = await resp.json()
                    return (data.get("result") or {}).get("value")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Kon blockhash-geldigheid niet checken: %s", exc)
            return None

    async def _confirm_transaction(
        self, signature: str, timeout_sec: float = 15.0, poll_interval_sec: float = 0.15
    ) -> None:
        """Polls until the transaction actually lands (or definitively
        failed) on-chain. Raises RuntimeError if it reverted, or if it never
        confirms within timeout_sec - never silently assume success.

        Lowered from 30s -> 15s: user-requested after live sessions showed
        a struggling sell attempt (indexing lag, network congestion) could
        take the full 30s before even starting its 15s retry cooldown,
        making a dead token's exit feel slow even after it was correctly
        detected as dead within ~10-15s. Trade-off, not a free win: a
        transaction that's genuinely still confirming past 15s (rare, but
        real) now gets classified "failed" and retried sooner, risking a
        duplicate attempt/fee for something that may have actually landed -
        acceptable given the user explicitly chose faster reaction over
        that small extra margin.

        poll_interval_sec lowered 0.5 -> 0.15, user-requested 2026-08-24
        ("how can it be even faster" -> "yes"): the bot's own real Trade
        timing logs showed the post-send confirmation-polling phase
        ("rest") dominating total trade time (~760-800ms of a ~1100-1300ms
        total, 65-70%) - far more than fetching the unsigned tx from
        PumpPortal (~300-500ms) or the blockhash check (~40-116ms). Solana's
        own block time is ~400-600ms, so at the old 500ms interval a
        transaction that landed on the very next block still waited out
        most of a full poll cycle before the bot even checked again.
        Tightening this doesn't change what gets submitted or add any real
        cost (unlike the priority-fee/Jito levers) - only how often an
        already-sent transaction's status gets checked. Real trade-off:
        more getSignatureStatuses RPC calls per trade, cheap but not free.
        Not lowered further than this - below Solana's own block time,
        extra polling just burns RPC calls without a matching chance of a
        new status existing yet."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignatureStatuses",
            "params": [[signature], {"searchTransactionHistory": True}],
        }
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.rpc_http_url, json=payload, timeout=10) as resp:
                        data = await resp.json()
                        statuses = (data.get("result") or {}).get("value") or [None]
                        status = statuses[0]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Kon transactiestatus niet checken voor %s: %s", signature, exc)
                status = None

            if status is not None:
                if status.get("err") is not None:
                    raise OnChainTransactionError(
                        f"Transactie {signature} is gefaald on-chain: {status['err']}",
                        custom_error_code=_extract_custom_error_code(status["err"]),
                    )
                if status.get("confirmationStatus") in ("confirmed", "finalized"):
                    return

            await asyncio.sleep(poll_interval_sec)

        raise RuntimeError(
            f"Transactie {signature} nog niet bevestigd na {timeout_sec}s - "
            f"status onbekend, behandel als mislukt."
        )

    async def _fetch_real_sol_delta(
        self, signature: str, max_attempts: int = 4, retry_delay_sec: float = 0.5,
    ) -> float | None:
        """Returns the wallet's actual net SOL change from this confirmed
        transaction (postBalance - preBalance for our own account, already
        fee-inclusive since Solana debits the fee from the same balance) -
        None if the lookup itself fails.

        User-requested, real finding 2026-08-23: over one 23-minute live
        session, real wallet balance dropped 0.155 SOL while the bot's own
        fee-model P&L tracker (a flat 1.75%-per-leg assumption, see
        fees.py) showed +0.057 SOL - a real, large gap NOT explained by
        fees (only ~24% of it). The rest is real slippage on fast-dying,
        thin-liquidity tokens the flat fee-model has no way to see. This
        gives _exit()/_partial_exit() the actual on-chain proceeds of a
        sell to compute real pnl_sol from, instead of estimating it.

        Real bug found live 2026-08-24, user-reported ("catecoin this is a
        false log", "wojakius also not correct"): the getTransaction call
        below specified no commitment level, so it defaulted to
        "finalized" - a MUCH stricter, slower level than the "confirmed"
        status _confirm_transaction() already waited for before this is
        ever called (see that method - it returns as soon as
        confirmationStatus is "confirmed", not "finalized"). Called only
        ~100ms after send, "finalized" data for the transaction usually
        isn't available yet, so this silently returned None almost every
        time - falling back to the stale PRE-SELL price-tick estimate in
        outcome_tracker.py's _exit(), instead of failing loudly. Confirmed
        directly via getTransaction on two real trades tonight: Catecoin
        was logged as "+40.2%" (est. from a stale tick) when the real
        on-chain result was roughly -87% (bought 0.012248 SOL, sold back
        only 0.001615 SOL); WOJAKIUS was logged as "+101.9%" when real
        proceeds were a real LOSS (bought 0.014879 SOL, sold back 0.008042
        SOL) - both mislabeled real losses as real gains, which also
        corrupted every downstream consumer of pct_change: the realized
        daily pnl total, and sniper_model.py's training labels (WIN_MARGIN_PCT
        comparison).

        Fixed by requesting "confirmed" explicitly (matching what was
        already waited for), plus a short retry loop - even "confirmed"
        can occasionally lag a beat behind getSignatureStatuses on a
        different RPC replica, and a silent wrong fallback is far more
        costly than one or two extra 500ms retries."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "json", "maxSupportedTransactionVersion": 0,
                    "commitment": "confirmed",
                },
            ],
        }
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.rpc_http_url, json=payload, timeout=10) as resp:
                        data = await resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Kon transactie %s niet ophalen voor real-delta: %s", signature, exc)
                return None

            result = data.get("result")
            if result:
                try:
                    account_keys = result["transaction"]["message"]["accountKeys"]
                    our_key = str(self.keypair.pubkey())
                    idx = account_keys.index(our_key)
                    pre = result["meta"]["preBalances"][idx]
                    post = result["meta"]["postBalances"][idx]
                    return (post - pre) / 1_000_000_000
                except (KeyError, ValueError, IndexError) as exc:
                    logger.debug(
                        "Kon real SOL-delta niet bepalen uit transactie %s: %s", signature, exc,
                    )
                    return None

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay_sec)

        logger.debug(
            "Transactie %s nog niet zichtbaar via getTransaction na %d pogingen - "
            "real-delta niet beschikbaar, valt terug op schatting.", signature, max_attempts,
        )
        return None

    async def _fetch_real_token_amount(
        self, signature: str, mint: str, max_attempts: int = 4, retry_delay_sec: float = 0.5,
    ) -> int | None:
        """Returns the exact number of raw base-unit tokens this wallet
        actually received from a confirmed buy transaction (postAmount -
        preAmount for our own token account, from meta.postTokenBalances/
        preTokenBalances - present regardless of the "json" encoding used
        below, same as meta.postBalances/preBalances in
        _fetch_real_sol_delta). None if the lookup fails or our account
        isn't present in either balance list.

        User-requested 2026-08-25 ("build" - absolute-amount sell) - feeds
        OutcomeTracker._fetch_real_token_amount_bg, which lets a later
        exit sell this EXACT on-chain-confirmed amount (see
        build_and_send_full_sell_by_amount) instead of a PumpPortal-index-
        dependent percentage, sidestepping MIN_SELL_DELAY_SEC entirely.
        Mirrors _fetch_real_sol_delta's retry/commitment reasoning
        exactly - same "confirmed" vs "finalized" bug class is possible
        here too."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "json", "maxSupportedTransactionVersion": 0,
                    "commitment": "confirmed",
                },
            ],
        }
        our_key = str(self.keypair.pubkey())
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.rpc_http_url, json=payload, timeout=10) as resp:
                        data = await resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Kon transactie %s niet ophalen voor token-amount: %s", signature, exc)
                return None

            result = data.get("result")
            if result:
                try:
                    meta = result["meta"]
                    pre_balances = meta.get("preTokenBalances") or []
                    post_balances = meta.get("postTokenBalances") or []
                    pre_amount = next(
                        (
                            int(b["uiTokenAmount"]["amount"]) for b in pre_balances
                            if b.get("owner") == our_key and b.get("mint") == mint
                        ),
                        0,
                    )
                    post_amount = next(
                        (
                            int(b["uiTokenAmount"]["amount"]) for b in post_balances
                            if b.get("owner") == our_key and b.get("mint") == mint
                        ),
                        None,
                    )
                    if post_amount is None:
                        return None
                    return post_amount - pre_amount
                except (KeyError, ValueError, TypeError) as exc:
                    logger.debug(
                        "Kon real token-amount niet bepalen uit transactie %s: %s", signature, exc,
                    )
                    return None

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay_sec)

        logger.debug(
            "Transactie %s nog niet zichtbaar via getTransaction na %d pogingen - "
            "real token-amount niet beschikbaar.", signature, max_attempts,
        )
        return None

    async def build_and_send_full_sell_by_amount(
        self,
        mint: str,
        token_amount: int,
        slippage_pct: float,
        priority_fee_sol: float = PRIORITY_FEE_SOL_PER_LEG,
        pool: str = "auto",
    ) -> dict:
        """Sells an EXACT, absolute token quantity (raw base units,
        integer) - "amount": <int> with denominatedInSol: "false", per
        PumpPortal's Local Trading API docs
        (https://pumpportal.fun/local-trading-api/trading-api - amount
        accepts either a percentage string like "100%" or a raw integer
        quantity). Unlike build_and_send_full_sell's percentage form,
        this does NOT ask PumpPortal to resolve "what does this wallet
        currently hold" against their OWN indexed balance -
        MIN_SELL_DELAY_SEC (outcome_tracker.py) exists ONLY because that
        percentage lookup reliably returns SellZeroAmount before their
        index catches up with a real buy (confirmed live). An absolute
        amount sourced from our OWN on-chain buy result (see
        OutcomeTracker._fetch_real_token_amount_bg) sidesteps that
        dependency, in principle allowing a real sell almost immediately
        after the buy itself confirms.

        User-requested 2026-08-25 ("how can we improve the baseline") -
        real 8h live data found 76% of ALL sniper exits held for exactly
        the 15-18s MIN_SELL_DELAY_SEC floor, by which point a fast-dying
        pump.fun curve had usually already reversed hard - take_profit
        itself (previously the ONE exit reason with a proven real edge,
        +28%/trade over 87 trades) had flipped to -23.2%/trade in that
        window, and 14 of 16 take_profit-labeled exits were actually
        losses once the real fill landed.

        NOT YET CONFIRMED to actually bypass PumpPortal's own internal
        validation - they may still check the requested amount against
        their own index regardless of format, in which case this fails
        exactly like the percentage form does today. The caller
        (OutcomeTracker._exit) must treat any failure here as "not yet
        sellable" and fall back to the proven percentage-based
        build_and_send_full_sell once MIN_SELL_DELAY_SEC has elapsed -
        same "log/test before trusting" discipline as every other speed
        change this session (see jito.py's RECOMMENDED_TIP_SOL docstring
        for the precedent)."""
        body = {
            "publicKey": str(self.keypair.pubkey()),
            "action": "sell",
            "mint": mint,
            "amount": token_amount,
            "denominatedInSol": "false",
            "slippage": slippage_pct,
            "priorityFee": priority_fee_sol,
            "pool": pool,
        }
        sig = await self._sign_and_send_with_migration_retry(body)
        real_sol_delta = await self._fetch_real_sol_delta(sig)
        return {
            "signature": sig, "action": "sell", "mint": mint, "amount": token_amount,
            "real_sol_delta": real_sol_delta,
        }

    async def _send_raw_transaction(self, raw_tx: bytes) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                base64.b64encode(raw_tx).decode("utf-8"),
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 3},
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.rpc_http_url, json=payload, timeout=15) as resp:
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(f"RPC sendTransaction fout: {data['error']}")
                return data["result"]
