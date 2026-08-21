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
# even ONE confirmed-unsellable token from a wallet is enough to block it -
# unlike a mere loss, "sell_paused" (see outcome_tracker.py) already only
# fires after a token survives BOTH a real 5-attempt failure cap AND a
# follow-up on-chain re-confirmation that it's still genuinely stuck (not a
# confirmation-timeout false alarm - confirmed live: most "paused" positions
# tonight turned out to have actually sold fine and self-resolved before
# ever reaching this signal). That's already a much higher bar than a single
# losing trade, so it doesn't need MIN_WALLET_SAMPLES-style repetition.
MIN_UNSELLABLE_SAMPLES = 1


def compute_wallet_stats(log_path=DATA_LOG_PATH) -> dict[str, dict]:
    """Returns {creator_wallet: {"count": int, "median_pct_change": float | None,
    "unsellable_count": int}}, built by joining our own buy records (which
    carry the launcher wallet in meta.creator) with our own exit records and
    sell_paused records, by mint. "count"/"median_pct_change" only reflect
    real measured exits, same as before - "unsellable_count" is tracked
    separately since a stuck position never produces an exit at all."""
    creators: dict[str, str] = {}
    outcomes: dict[str, float] = {}
    unsellable_mints: set[str] = set()
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
                elif d.get("type") == "sell_paused":
                    mint = d.get("mint")
                    if mint:
                        unsellable_mints.add(mint)
    except FileNotFoundError:
        return {}

    by_wallet: dict[str, list[float]] = {}
    for mint, pct in outcomes.items():
        creator = creators.get(mint)
        if not creator:
            continue
        by_wallet.setdefault(creator, []).append(net_pct_change_after_fees(pct))

    unsellable_by_wallet: dict[str, int] = {}
    for mint in unsellable_mints:
        creator = creators.get(mint)
        if not creator:
            continue
        unsellable_by_wallet[creator] = unsellable_by_wallet.get(creator, 0) + 1

    stats: dict[str, dict] = {}
    for wallet in set(by_wallet) | set(unsellable_by_wallet):
        pct_changes = by_wallet.get(wallet)
        stats[wallet] = {
            "count": len(pct_changes) if pct_changes else 0,
            "median_pct_change": round(statistics.median(pct_changes), 2) if pct_changes else None,
            "unsellable_count": unsellable_by_wallet.get(wallet, 0),
        }
    return stats


def blocked_wallets(
    log_path=DATA_LOG_PATH,
    min_samples: int = MIN_WALLET_SAMPLES,
    median_threshold: float = BLOCK_MEDIAN_THRESHOLD_PCT,
    min_unsellable_samples: int = MIN_UNSELLABLE_SAMPLES,
) -> set[str]:
    """Wallets with enough of our own trades against them, and a net-loss
    median outcome, that future launches from them should be skipped - OR
    at least min_unsellable_samples confirmed-unsellable tokens, which is
    its own, stronger signal (see MIN_UNSELLABLE_SAMPLES)."""
    stats = compute_wallet_stats(log_path)
    return {
        wallet
        for wallet, s in stats.items()
        if (
            s["median_pct_change"] is not None
            and s["count"] >= min_samples
            and s["median_pct_change"] < median_threshold
        )
        or s["unsellable_count"] >= min_unsellable_samples
    }
