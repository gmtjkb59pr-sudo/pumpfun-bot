"""
Retroactively corrects pct_change for every real (dry_run=false) exit in
data/activity_log.jsonl, using the wallet's TRUE on-chain SOL delta for
BOTH the buy and sell transactions instead of whatever the bot logged at
the time.

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
(see that file's _fetch_real_sol_delta docstring).

Second real bug found live the same night (user: "buy side cost can you
fix that aswell"): even after the sell-side fix, the pnl COST BASIS was
still the nominal trade_size_sol bookkeeping value, not what was really
spent on the buy - real buy-side slippage meant these differ (confirmed
live: WOJAKIUS's real buy cost was ~24% over nominal). Fixed going
forward in outcome_tracker.py's _fetch_real_buy_cost(); this script does
the equivalent retroactive correction for the historical record, using
each exit's matched buy's real on-chain cost as the denominator instead
of nominal trade_size_sol wherever that buy's real cost can be resolved.

Does NOT rewrite data/activity_log.jsonl in place - that file is an
append-only event log the live bot continuously writes to, so mutating
it concurrently is unsafe. Instead writes a separate correction index,
data/real_pnl_corrections.json, keyed by sell tx_signature - consulted
by sniper_model.py's retrain_and_save() in preference to the
originally-logged pct_change whenever a correction exists.

Resumable/idempotent - safe to re-run. Each phase only fetches what the
existing corrections file doesn't already have (a sell-side delta, then
separately a buy-side cost), so a re-run after this script gained the
buy-side phase only does the NEW work, not the already-done sell-side
lookups again.

Usage:
    ./venv/bin/python scripts/audit_real_pnl.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
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


def _load_buys_by_mint(activity_log_path: Path) -> dict[str, list[tuple[float, str]]]:
    """mint -> [(ts, tx_signature), ...] sorted by ts, for every real
    (dry_run=false) buy across every strategy - used to find each exit's
    matching buy (the most recent one at or before the exit's own ts)."""
    buys: dict[str, list[tuple[float, str]]] = defaultdict(list)
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
                record.get("type") == "trade" and record.get("action") == "buy"
                and record.get("dry_run") is False and record.get("tx_signature")
            ):
                buys[record["mint"]].append((record.get("ts", 0.0), record["tx_signature"]))
    for mint_buys in buys.values():
        mint_buys.sort(key=lambda pair: pair[0])
    return buys


def _find_matching_buy_signature(
    buys_by_mint: dict[str, list[tuple[float, str]]], mint: str, exit_ts: float,
) -> str | None:
    candidates = buys_by_mint.get(mint) or []
    matched = None
    for ts, sig in candidates:
        if ts <= exit_ts:
            matched = sig
        else:
            break
    return matched


def _recompute(entry: dict) -> None:
    """Recomputes corrected_pct_change/real_pnl_sol from whatever's
    currently known - real buy cost if resolved, nominal trade_size_sol
    otherwise (unchanged from before the buy-side phase existed)."""
    sell_delta = entry.get("real_sol_delta")
    if sell_delta is None:
        return
    cost_basis = entry.get("real_buy_cost_sol") or entry["trade_size_sol"]
    if not cost_basis:
        return
    entry["corrected_pct_change"] = round((sell_delta / cost_basis - 1) * 100, 2)
    entry["real_pnl_sol"] = round(sell_delta - cost_basis, 6)


async def main() -> None:
    if not ACTIVITY_LOG_PATH.exists():
        print(f"Geen {ACTIVITY_LOG_PATH} gevonden - niets te auditen.")
        return

    cfg = load_config()
    our_key = str(Keypair.from_bytes(base58.b58decode(cfg.private_key)).pubkey())
    rpc_url = cfg.rpc_http_url

    exits = _load_real_exits(ACTIVITY_LOG_PATH)
    print(f"{len(exits)} echte exits gevonden.")

    corrections: dict[str, dict] = {}
    if CORRECTIONS_PATH.exists():
        try:
            corrections = json.loads(CORRECTIONS_PATH.read_text())
        except Exception:  # noqa: BLE001
            corrections = {}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LOOKUPS)

    # --- Phase 1: sell-side real delta (unchanged from before) ---
    to_fetch_sell = [e for e in exits if e["tx_signature"] not in corrections]
    print(f"\nFase 1 (sell-side): {len(to_fetch_sell)} nieuw, {len(exits) - len(to_fetch_sell)} al bekend.")
    total_1, done_1 = len(to_fetch_sell), 0

    async with aiohttp.ClientSession() as session:
        async def _process_sell(exit_record: dict) -> None:
            nonlocal done_1
            async with semaphore:
                delta = await _fetch_real_delta(session, rpc_url, our_key, exit_record["tx_signature"])
                await asyncio.sleep(PACE_DELAY_SEC)
            done_1 += 1
            if done_1 % 50 == 0 or done_1 == total_1:
                print(f"  {done_1}/{total_1} opgehaald...")
            trade_size = exit_record.get("trade_size_sol") or 0.0
            entry = {
                "mint": exit_record.get("mint"),
                "name": exit_record.get("name"),
                "reason": exit_record.get("reason"),
                "ts": exit_record.get("ts"),
                "trade_size_sol": trade_size,
                "original_pct_change": exit_record.get("pct_change"),
                "real_sol_delta": round(delta, 9) if delta is not None else None,
                "real_buy_cost_sol": None,
                "corrected_pct_change": exit_record.get("pct_change"),
                "real_pnl_sol": 0.0,
            }
            _recompute(entry)
            if delta is not None and trade_size:
                corrections[exit_record["tx_signature"]] = entry

        await asyncio.gather(*(_process_sell(e) for e in to_fetch_sell))

        # --- Phase 2: buy-side real cost (new) ---
        buys_by_mint = _load_buys_by_mint(ACTIVITY_LOG_PATH)
        to_fetch_buy = [
            (sig, entry) for sig, entry in corrections.items()
            if entry.get("real_buy_cost_sol") is None
        ]
        print(f"\nFase 2 (buy-side): {len(to_fetch_buy)} exits missen nog een real buy-cost.")
        total_2, done_2 = len(to_fetch_buy), 0

        async def _process_buy(sig: str, entry: dict) -> None:
            nonlocal done_2
            buy_sig = _find_matching_buy_signature(buys_by_mint, entry.get("mint"), entry.get("ts", 0.0))
            if buy_sig is not None:
                async with semaphore:
                    delta = await _fetch_real_delta(session, rpc_url, our_key, buy_sig)
                    await asyncio.sleep(PACE_DELAY_SEC)
                if delta is not None and delta < 0:
                    entry["real_buy_cost_sol"] = round(-delta, 9)
                    _recompute(entry)
            done_2 += 1
            if done_2 % 50 == 0 or done_2 == total_2:
                print(f"  {done_2}/{total_2} opgehaald...")

        await asyncio.gather(*(_process_buy(sig, entry) for sig, entry in to_fetch_buy))

    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_PATH.write_text(json.dumps(corrections, indent=2))

    resolved = sum(1 for e in exits if e["tx_signature"] in corrections)
    with_buy_cost = sum(1 for c in corrections.values() if c.get("real_buy_cost_sol") is not None)
    print(f"\n{resolved}/{len(exits)} exits hebben een gecorrigeerde real pct_change.")
    print(f"{with_buy_cost}/{resolved} daarvan gebruiken ook de echte buy-side cost (i.p.v. nominaal).")
    print(f"Opgeslagen naar {CORRECTIONS_PATH}")

    total_original_est = 0.0
    total_real = 0.0
    big_mismatches = 0
    for c in corrections.values():
        orig = c.get("original_pct_change")
        trade_size = c["trade_size_sol"]
        if orig is not None:
            total_original_est += trade_size * (orig / 100)
        total_real += c["real_pnl_sol"]
        if orig is not None and abs(c["corrected_pct_change"] - orig) > 15:
            big_mismatches += 1

    print(f"\nTotaal 'geloofd' pnl (uit gelogde pct_change, ruwe schatting): {total_original_est:+.4f} SOL")
    print(f"Totaal ECHTE pnl (uit on-chain delta, nu incl. buy-side):       {total_real:+.4f} SOL")
    print(f"Exits met >15 procentpunt afwijking: {big_mismatches}/{len(corrections)}")


if __name__ == "__main__":
    asyncio.run(main())
