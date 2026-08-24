"""
Retroactively corrects pct_change for every real (dry_run=false) exit in
data/activity_log.jsonl, using the wallet's TRUE on-chain SOL delta for
the sell transaction instead of whatever the bot logged at the time.

Real bug found live 2026-08-24 (user-reported: "catecoin this is a false
log with what happened really", "wojakius also not correct"):
PumpPortalClient._fetch_real_sol_delta queried getTransaction with no
commitment level, defaulting to "finalized" - far stricter/slower than
the "confirmed" status already waited for a moment earlier. Called only
~100ms after send, it almost always silently returned None, and
outcome_tracker.py's _exit() fell back to a stale PRE-SELL price-tick
estimate instead. Audited one live session (25 real exits): every single
one was overstated, total real pnl was -0.029 SOL vs a believed
+0.129 SOL - this wasn't an occasional flake, it was failing close to
100% of the time. Fixed going forward in pumpfun_bot/pumpportal_client.py
(see that file's _fetch_real_sol_delta docstring); this script corrects
the HISTORICAL record so sniper_model.py's training labels reflect what
actually happened on-chain, not what got silently misreported.

Does NOT rewrite data/activity_log.jsonl in place - that file is an
append-only event log the live bot continuously writes to, so mutating
it concurrently is unsafe. Instead writes a separate correction index,
data/real_pnl_corrections.json, keyed by sell tx_signature - consulted
by scripts/train_sniper_model.py in preference to the originally-logged
pct_change whenever a correction exists.

Uses the SAME convention as outcome_tracker.py's _exit(): cost basis is
the position's nominal trade_size_sol (bookkeeping), not a re-fetched
real buy-side delta - only the sell side was ever wrong (the buy side's
real_sol_delta lookup was never used for anything), so this only needs
ONE getTransaction call per exit, not two.

Usage:
    ./venv/bin/python scripts/audit_real_pnl.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import base58
from solders.keypair import Keypair

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pumpfun_bot.config import load_config  # noqa: E402

ACTIVITY_LOG_PATH = Path("data/activity_log.jsonl")
CORRECTIONS_PATH = Path("data/real_pnl_corrections.json")
MAX_CONCURRENT_LOOKUPS = 1
MAX_ATTEMPTS = 5
RETRY_DELAY_SEC = 1.5
# user's shared Helius RPC key visibly rate-limits well below what its
# error responses would suggest - confirmed live: signatures this script
# marked "failed" at concurrency=2 resolved perfectly fine when re-queried
# standalone seconds later. Fully sequential, paced, is what actually
# gets a truthful result instead of a fast but wrong one.
PACE_DELAY_SEC = 0.4


async def _fetch_real_delta(
    session: aiohttp.ClientSession, rpc_url: str, our_key: str, signature: str,
) -> float | None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {"encoding": "json", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"},
        ],
    }
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
        except Exception:  # noqa: BLE001
            return None
        result = data.get("result")
        if result:
            try:
                account_keys = result["transaction"]["message"]["accountKeys"]
                idx = account_keys.index(our_key)
                pre = result["meta"]["preBalances"][idx]
                post = result["meta"]["postBalances"][idx]
                return (post - pre) / 1_000_000_000
            except (KeyError, ValueError, IndexError):
                return None
        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY_SEC)
    return None


def _load_real_exits(activity_log_path: Path) -> list[dict]:
    exits = []
    with activity_log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("type") == "exit" and record.get("dry_run") is False
                and record.get("tx_signature")
            ):
                exits.append(record)
    return exits


async def main() -> None:
    if not ACTIVITY_LOG_PATH.exists():
        print(f"Geen {ACTIVITY_LOG_PATH} gevonden - niets te auditen.")
        return

    cfg = load_config()
    our_key = str(Keypair.from_bytes(base58.b58decode(cfg.private_key)).pubkey())
    rpc_url = cfg.rpc_http_url

    exits = _load_real_exits(ACTIVITY_LOG_PATH)
    print(f"{len(exits)} echte exits gevonden - real on-chain delta ophalen...")

    corrections: dict[str, dict] = {}
    if CORRECTIONS_PATH.exists():
        try:
            corrections = json.loads(CORRECTIONS_PATH.read_text())
        except Exception:  # noqa: BLE001
            corrections = {}

    to_fetch = [e for e in exits if e["tx_signature"] not in corrections]
    print(f"{len(to_fetch)} nieuw (rest al eerder gecorrigeerd in {CORRECTIONS_PATH}).")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LOOKUPS)
    total = len(to_fetch)
    done = 0

    async with aiohttp.ClientSession() as session:
        async def _process(exit_record: dict) -> None:
            nonlocal done
            async with semaphore:
                delta = await _fetch_real_delta(
                    session, rpc_url, our_key, exit_record["tx_signature"],
                )
                await asyncio.sleep(PACE_DELAY_SEC)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} opgehaald...")
            if delta is None:
                return
            trade_size = exit_record.get("trade_size_sol") or 0.0
            if not trade_size:
                return
            corrected_pct = round((delta / trade_size - 1) * 100, 2)
            real_pnl = round(delta - trade_size, 6)
            corrections[exit_record["tx_signature"]] = {
                "mint": exit_record.get("mint"),
                "name": exit_record.get("name"),
                "reason": exit_record.get("reason"),
                "ts": exit_record.get("ts"),
                "trade_size_sol": trade_size,
                "original_pct_change": exit_record.get("pct_change"),
                "corrected_pct_change": corrected_pct,
                "real_pnl_sol": real_pnl,
                "real_sol_delta": round(delta, 9),
            }

        await asyncio.gather(*(_process(e) for e in to_fetch))

    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_PATH.write_text(json.dumps(corrections, indent=2))

    resolved = sum(1 for e in exits if e["tx_signature"] in corrections)
    print(f"\n{resolved}/{len(exits)} exits hebben nu een gecorrigeerde real pct_change.")
    print(f"Opgeslagen naar {CORRECTIONS_PATH}")

    total_original_est = 0.0
    total_real = 0.0
    big_mismatches = 0
    for sig, c in corrections.items():
        orig = c.get("original_pct_change")
        trade_size = c["trade_size_sol"]
        if orig is not None:
            total_original_est += trade_size * (orig / 100)
        total_real += c["real_pnl_sol"]
        if orig is not None and abs(c["corrected_pct_change"] - orig) > 15:
            big_mismatches += 1

    print(f"\nTotaal 'geloofd' pnl (uit gelogde pct_change, ruwe schatting): {total_original_est:+.4f} SOL")
    print(f"Totaal ECHTE pnl (uit on-chain delta):                        {total_real:+.4f} SOL")
    print(f"Exits met >15 procentpunt afwijking: {big_mismatches}/{len(corrections)}")


if __name__ == "__main__":
    asyncio.run(main())
