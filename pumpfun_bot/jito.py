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
dexscreener.py, etc.).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time

import aiohttp

logger = logging.getLogger("pumpfun_bot.jito")

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
