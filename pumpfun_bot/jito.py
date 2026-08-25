"""
Submits already-signed transactions to Jito's Block Engine as a bundle,
instead of a normal Solana RPC sendTransaction - user-requested 2026-08-24
("is there still autonomous learning" -> "can you change that" -> ...
-> "yes build if you think it will make the bot better").

Why this exists: this session's own real, sourced data (see fees.py's
DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON) shows the dominant real cost on
this bot's trades is NOT protocol/API fees but execution slippage/staleness
- the gap between a tick-based exit decision and what the sell actually
lands at, on thin, fast-reversing pump.fun bonding curves. A normal
sendTransaction races every other bot on the public mempool; a Jito bundle
guarantees atomic same-slot inclusion (or doesn't land at all) once a
validator accepts the tip, which is the standard technique serious sniper/
sell bots use specifically to close this exact gap.

Confirmed against Jito's own docs 2026-08-24 (docs.jito.wtf/lowlatencytxnsend):
- sendBundle: JSON-RPC method, params = [[base64 signed tx, ...], {"encoding":
  "base64"}] - up to 5 transactions. Base64 explicitly recommended over
  base58 (deprecated for this call, performance). Returns a bundle_id.
- getBundleStatuses: JSON-RPC method, params = [[bundle_id, ...]] (max 5).
  confirmation_status is one of processed/confirmed/finalized; a missing
  bundle (not yet landed, or dropped) returns null for that entry, not an
  error - same "genuinely might just not be there yet" semantics as
  PumpPortalClient._confirm_transaction's getSignatureStatuses polling,
  mirrored here on purpose.

Does NOT handle turning a PumpPortal trade request into a signed
transaction - that stays PumpPortalClient's job (it already fetches an
unsigned tx and signs locally with the bot's own keypair; Jito mode only
changes what happens to the already-signed bytes). This module is a thin,
dependency-light HTTP wrapper around Jito's two RPC methods, matching this
codebase's existing per-concern module style (holder_count.py,
dexscreener.py, etc.) - PLUS build_tip_transaction below, added after a
real live failure.

Real bug found live 2026-08-24, hours after this shipped: submitting just
the signed sell tx alone got rejected by Jito with "Bundles must write
lock at least one tip account to be eligible for the auction." Root
cause: PumpPortal's own bundle-mode endpoint bakes a tip-account transfer
INTO the transaction it builds, but that endpoint 400s on every real sell
(see pumpportal_client.py's build_and_send_full_sell_via_jito_bundle for
that whole story) - the workaround that avoided the 400 (fetching the
sell tx through the normal single-object endpoint instead) also silently
lost the tip instruction, since that endpoint has no reason to include one.

Fixed the RIGHT way per Jito's own docs (confirmed 2026-08-24,
docs.jito.wtf/lowlatencytxnsend): the tip requirement is on the BUNDLE,
not any specific transaction in it - "any instruction, top-level or CPI,
that transfers SOL to one of the 8 tip accounts" satisfies it. Rather
than risk manually decompiling and re-signing PumpPortal's own swap
transaction (confirmed live it uses an address lookup table across 4
instructions - reconstructing correct signer/writable flags for ALT-
resolved accounts by hand is real, easy-to-get-subtly-wrong work against
real money), build_tip_transaction below produces a second, tiny,
from-scratch transaction that ONLY transfers SOL to a tip account -
submitted as the second transaction in the same 2-tx bundle, alongside
the untouched, already-proven sell transaction.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import random
import time

import aiohttp
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

logger = logging.getLogger("pumpfun_bot.jito")

# the 8 official Jito tip accounts, confirmed live against
# docs.jito.wtf/lowlatencytxnsend 2026-08-24 - a tip to ANY one of these
# satisfies a bundle's tip requirement, picked at random per bundle to
# spread load rather than hammering the same one every time
TIP_ACCOUNTS = (
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
)
# confirmed live 2026-08-24: "Jito enforces a minimum tip of 1000 lamports
# for bundles" - a tip below this gets rejected outright, same failure
# mode as no tip at all. This is Jito's own PROTOCOL floor, not a
# recommendation - see RECOMMENDED_TIP_SOL below for what actually gets a
# bundle included in practice, a much bigger real number.
MIN_TIP_LAMPORTS = 1000

# user-requested 2026-08-24 ("try" -> "build it" - size-gating the whole
# feature after finding the tip cost doesn't fit small trades) - real,
# sourced from live testing against this bot's actual real trading wallet:
# a 0.0025 SOL tip (this bot's normal take_profit priority fee, reused
# naively as the tip in the first version of this fix) failed to land
# TWICE across two different attempts (global AND the NY regional block
# engine, 25s and 30s timeouts) - the bundle was accepted but no validator
# ever included it. A 0.01 SOL tip landed on the first try. n=1 success,
# n=2 failure at the smaller size - a real but thin sample, not a
# rigorously-tuned number; revisit if real trades show this is
# consistently too low (or unnecessarily high) once more data exists.
# Deliberately kept SEPARATE from priority_fee_sol_for_sell's boosted fee
# (fees.py) - that's the sell transaction's OWN compute-budget priority
# fee, needed regardless of Jito; this is an independent cost only paid
# when the Jito path is actually used.
RECOMMENDED_TIP_SOL = 0.01


def build_tip_transaction(
    payer: Keypair, tip_lamports: int, recent_blockhash: Hash,
) -> VersionedTransaction:
    """A minimal, single-instruction, from-scratch transaction that only
    transfers tip_lamports (floored at MIN_TIP_LAMPORTS) to a randomly-
    picked Jito tip account - deliberately NOT touching or modifying any
    other transaction (see module docstring for why). Uses the SAME
    recent_blockhash as whatever real transaction it's bundled with, so
    both land in the same slot's validity window."""
    tip_lamports = max(tip_lamports, MIN_TIP_LAMPORTS)
    tip_account = Pubkey.from_string(random.choice(TIP_ACCOUNTS))
    ix = transfer(TransferParams(
        from_pubkey=payer.pubkey(), to_pubkey=tip_account, lamports=tip_lamports,
    ))
    message = MessageV0.try_compile(
        payer=payer.pubkey(), instructions=[ix], address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash,
    )
    return VersionedTransaction(message, [payer])

# "Global" - Jito's own recommended default when no specific region is
# known to be closer; regional endpoints exist (Frankfurt, Tokyo, NY, ...)
# but picking one requires knowing where this bot actually runs, not
# assumed here.
DEFAULT_BLOCK_ENGINE_URL = "https://mainnet.block-engine.jito.wtf"

BUNDLE_STATUS_POLL_INTERVAL_SEC = 1.0
BUNDLE_STATUS_TIMEOUT_SEC = 30.0


async def send_bundle(
    signed_txs: list[bytes], block_engine_url: str = DEFAULT_BLOCK_ENGINE_URL,
    timeout_sec: float = 15.0,
) -> str | None:
    """Submits up to 5 already-signed raw transactions as one atomic
    bundle. Returns the bundle_id, or None if the submission itself
    failed (network error, non-200, malformed response) - never raises,
    matching this codebase's own fail-safe-not-fail-loud convention for
    speculative/best-effort network calls."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "sendBundle",
        "params": [
            [base64.b64encode(tx).decode("utf-8") for tx in signed_txs],
            {"encoding": "base64"},
        ],
    }
    url = f"{block_engine_url}/api/v1/bundles"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Jito sendBundle gaf status %d: %s", resp.status, text)
                    return None
                data = await resp.json()
                if "error" in data:
                    logger.warning("Jito sendBundle fout: %s", data["error"])
                    return None
                return data.get("result")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kon Jito bundle niet versturen: %s", exc)
        return None


async def get_bundle_status(
    bundle_id: str, block_engine_url: str = DEFAULT_BLOCK_ENGINE_URL, timeout_sec: float = 10.0,
) -> dict | None:
    """Returns the raw status entry for this bundle_id ({"confirmation_status",
    "err", "slot", ...}), or None if it hasn't landed yet (or was dropped) -
    NOT an error, a bundle that never gets included is a real, expected
    outcome (someone else's transaction landed first, or no validator
    picked it up in time), not a network failure."""
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getBundleStatuses", "params": [[bundle_id]],
    }
    url = f"{block_engine_url}/api/v1/bundles"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout_sec) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if "error" in data:
                    return None
                values = ((data.get("result") or {}).get("value")) or []
                return values[0] if values else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon Jito bundle-status niet checken voor %s: %s", bundle_id, exc)
        return None


async def poll_bundle_until_landed(
    bundle_id: str,
    block_engine_url: str = DEFAULT_BLOCK_ENGINE_URL,
    timeout_sec: float = BUNDLE_STATUS_TIMEOUT_SEC,
    poll_interval_sec: float = BUNDLE_STATUS_POLL_INTERVAL_SEC,
) -> dict | None:
    """Polls until the bundle reaches "confirmed"/"finalized", errors
    on-chain, or timeout_sec elapses with no result. Returns the final
    status dict (check status["err"] for on-chain failure) or None if it
    never showed up in time - same ambiguity PumpPortalClient's own
    _confirm_transaction accepts for a plain sendTransaction: "never
    confirmed within the timeout" and "definitely failed" are different
    things, and this can only ever tell you the former without more
    aggressive (and costly) resubmission logic this doesn't attempt."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = await get_bundle_status(bundle_id, block_engine_url)
        if status is not None and status.get("confirmation_status") in ("confirmed", "finalized"):
            return status
        await asyncio.sleep(poll_interval_sec)
    return None
