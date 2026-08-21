"""
Tracks per-launcher-wallet performance from our own trade history, so the
sniper can skip future launches from a wallet that has repeatedly produced
losing tokens - a well-known pump.fun pattern (the same wallet deploying one
rug/dud after another under a new name each time).

Deliberately conservative, matching auto_tuner.py's philosophy:
- Only ever blocks, never un-blocks automatically - once a wallet has shown a
  clear pattern of bad launches, there's no evidence-based case for trusting
  it again later.
- Requires a minimum number of OUR OWN completed trades against that wallet
  as launcher before acting, to avoid blocking on a single bad launch.
- Uses the net-after-fees median (not mean) for the same fat-tailed-
  distribution reason as stats.py/auto_tuner.py - a couple of huge winners
  shouldn't mask a wallet that mostly launches duds.
- This can only ever narrow what we buy, never widen it, so it can't increase
  live risk exposure.
"""
from __future__ import annotations

import json
import statistics

from .activity_log import DATA_LOG_PATH
from .fees import net_pct_change_after_fees

MIN_WALLET_SAMPLES = 3
BLOCK_MEDIAN_THRESHOLD_PCT = 0.0  # block once the median outcome is a net loss


def compute_wallet_stats(log_path=DATA_LOG_PATH) -> dict[str, dict]:
    """Returns {creator_wallet: {"count": int, "median_pct_change": float}},
    built by joining our own buy records (which carry the launcher wallet in
    meta.creator) with our own exit records, by mint."""
    creators: dict[str, str] = {}
    outcomes: dict[str, float] = {}
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("action") == "buy":
                    creator = (d.get("meta") or {}).get("creator")
                    mint = d.get("mint")
                    if creator and mint:
                        creators[mint] = creator
                elif d.get("type") == "exit":
                    mint = d.get("mint")
                    pct = d.get("pct_change")
                    if mint and pct is not None:
                        outcomes[mint] = pct
    except FileNotFoundError:
        return {}

    by_wallet: dict[str, list[float]] = {}
    for mint, pct in outcomes.items():
        creator = creators.get(mint)
        if not creator:
            continue
        by_wallet.setdefault(creator, []).append(net_pct_change_after_fees(pct))

    return {
        wallet: {
            "count": len(pct_changes),
            "median_pct_change": round(statistics.median(pct_changes), 2),
        }
        for wallet, pct_changes in by_wallet.items()
    }


def blocked_wallets(
    log_path=DATA_LOG_PATH,
    min_samples: int = MIN_WALLET_SAMPLES,
    median_threshold: float = BLOCK_MEDIAN_THRESHOLD_PCT,
) -> set[str]:
    """Wallets with enough of our own trades against them, and a net-loss
    median outcome, that future launches from them should be skipped."""
    stats = compute_wallet_stats(log_path)
    return {
        wallet
        for wallet, s in stats.items()
        if s["count"] >= min_samples and s["median_pct_change"] < median_threshold
    }
