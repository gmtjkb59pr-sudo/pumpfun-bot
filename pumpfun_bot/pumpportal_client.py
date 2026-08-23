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

from .fees import PRIORITY_FEE_SOL_PER_LEG

logger = logging.getLogger("pumpfun_bot.pumpportal")


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
        sig = await self._sign_and_send(body)
        return {"signature": sig, "action": action, "mint": mint, "amount_sol": amount_sol}

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
        sig = await self._sign_and_send(body)
        return {"signature": sig, "action": "sell", "mint": mint, "amount": amount}

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
        self, signature: str, timeout_sec: float = 15.0, poll_interval_sec: float = 0.5
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
        that small extra margin."""
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
                    raise RuntimeError(
                        f"Transactie {signature} is gefaald on-chain: {status['err']}"
                    )
                if status.get("confirmationStatus") in ("confirmed", "finalized"):
                    return

            await asyncio.sleep(poll_interval_sec)

        raise RuntimeError(
            f"Transactie {signature} nog niet bevestigd na {timeout_sec}s - "
            f"status onbekend, behandel als mislukt."
        )

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
