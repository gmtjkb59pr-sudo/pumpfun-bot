"""
Finds real, evidence-based copytrade candidate wallets by looking at who
actually profited on the tokens this bot has bought - not by guessing.

Built after a first manual pass tonight found that the wallets repeatedly
profitable across our bought tokens were almost all Birdeye-tagged
"sniper"/"bundler" - professional launch-bots, or the manufactured-supply
wallets behind the exact bundled/rug pattern this bot's own quality
filters (holder_concentration.py) exist to avoid buying alongside, not
wallets worth mirroring. A sniper's edge is raw speed; by the time this
bot reacts to a signal, that edge is already gone - copying them would
mean copying the pattern we're trying to screen out, not real alpha.

Excludes those tags client-side (EXCLUDED_TAGS) and looks for wallets that
show up as a genuinely profitable trader across MULTIPLE of our bought
tokens - a single big number on one token could be luck; repeat
performance across many different, unrelated tokens is a real, if still
imperfect, skill signal.

Persists one snapshot per mint queried to its own cache file
(WALLET_DISCOVERY_CACHE_PATH), separate from activity_log.jsonl (that
file is for trade/exit/alert events, not this kind of one-off lookup), so
repeated runs never re-query a mint already analyzed - each Birdeye call
costs real, budget-limited CU (see birdeye.py's module docstring: the
free tier is ~1,000 calls/month, and birdeye_movers' own regular polling
already uses most of that). CALLS_PER_RUN_DEFAULT caps any single run
small; the dataset is meant to grow incrementally across many runs
instead of one expensive sweep.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from .birdeye import fetch_top_traders

logger = logging.getLogger("pumpfun_bot.wallet_discovery")

WALLET_DISCOVERY_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "wallet_discovery_cache.jsonl"

# confirmed live: every repeat "profitable" wallet found in a first manual
# pass carried one of these tags - not organic buyers worth mirroring
EXCLUDED_TAGS = frozenset({"sniper", "bundler", "dev"})

# a single token's PnL could be luck - this is the minimum for "shows up
# as profitable across multiple of our tokens" to mean anything
MIN_APPEARANCES_FOR_CANDIDATE = 2

CALLS_PER_RUN_DEFAULT = 20
CALL_DELAY_SEC_DEFAULT = 1.2  # Birdeye's free tier is 1 request/second


def load_cached_snapshots() -> dict[str, dict]:
    """Returns {mint: {"mint":..., "ts":..., "traders": [...]}} for every
    mint ever queried, across all past runs of this module."""
    snapshots: dict[str, dict] = {}
    if not WALLET_DISCOVERY_CACHE_PATH.exists():
        return snapshots
    with open(WALLET_DISCOVERY_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            mint = record.get("mint")
            if mint:
                snapshots[mint] = record  # a later record for the same mint wins
    return snapshots


def _append_snapshot(mint: str, traders: list[dict] | None) -> None:
    WALLET_DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WALLET_DISCOVERY_CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"mint": mint, "ts": time.time(), "traders": traders}) + "\n")


async def fetch_and_cache_new_mints(
    mints: list[str],
    api_key: str,
    *,
    calls_per_run: int = CALLS_PER_RUN_DEFAULT,
    delay_sec: float = CALL_DELAY_SEC_DEFAULT,
) -> int:
    """Queries Birdeye for whichever of `mints` aren't already cached, up
    to calls_per_run - real API calls, real CU cost, paced at delay_sec
    between calls to stay under the free tier's 1 req/sec limit. Returns
    how many were actually fetched (including failed lookups, which are
    still cached as None so a bad response doesn't get retried forever on
    every run - see the module docstring on why every call matters)."""
    cached = load_cached_snapshots()
    to_fetch = [m for m in mints if m not in cached][:calls_per_run]
    fetched = 0
    for mint in to_fetch:
        traders = await fetch_top_traders(mint, api_key)
        _append_snapshot(mint, traders)
        fetched += 1
        if traders is None:
            logger.debug("Wallet-discovery: kon top traders niet ophalen voor %s.", mint)
        await asyncio.sleep(delay_sec)
    return fetched


def is_candidate_wallet(trader: dict) -> bool:
    """A trader entry is disqualified if it carries any excluded tag (see
    module docstring) - never returns True for a wallet we can't confirm
    is clean of those tags."""
    tags = set(trader.get("tags") or [])
    return not (tags & EXCLUDED_TAGS)


def aggregate_wallet_performance(snapshots: dict[str, dict]) -> dict[str, dict]:
    """Combines every cached snapshot into a per-wallet view: which of our
    bought tokens they showed up as a top trader on (excluding sniper/
    bundler/dev-tagged appearances), total PnL across those, and which
    tokens. Only meaningful for wallets with >= MIN_APPEARANCES_FOR_
    CANDIDATE appearances - see repeat_candidates()."""
    by_wallet: dict[str, dict] = {}
    for mint, record in snapshots.items():
        traders = record.get("traders") or []
        for trader in traders:
            if not is_candidate_wallet(trader):
                continue
            owner = trader.get("owner")
            if not owner:
                continue
            pnl = trader.get("totalPnl", 0) or 0
            entry = by_wallet.setdefault(owner, {"appearances": [], "total_pnl": 0.0})
            entry["appearances"].append({"mint": mint, "pnl": pnl})
            entry["total_pnl"] += pnl
    return by_wallet


def repeat_candidates(
    by_wallet: dict[str, dict], min_appearances: int = MIN_APPEARANCES_FOR_CANDIDATE,
) -> list[tuple[str, dict]]:
    """Wallets with real repeat performance across multiple of our bought
    tokens, ranked by total PnL descending - the actual output of this
    module, meant as a starting shortlist for copytrade.watched_wallets,
    not an auto-applied config change."""
    candidates = [
        (wallet, info) for wallet, info in by_wallet.items()
        if len(info["appearances"]) >= min_appearances
    ]
    candidates.sort(key=lambda item: -item[1]["total_pnl"])
    return candidates
