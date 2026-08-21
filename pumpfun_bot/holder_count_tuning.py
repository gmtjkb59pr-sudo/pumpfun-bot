"""
Looks at social_watch's own buy/exit history to decide whether real evidence
supports raising min_holder_count - same conservative philosophy as
auto_tuner.py:
- Only ever tightens (raises the bar), never loosens.
- Requires a real minimum sample size in both the candidate bucket and the
  baseline before acting, to avoid reacting to noise on a handful of trades.
- Only applies to social_watch, not sniper - sniper buys instantly, before
  holder count is even real (confirmed: checking immediately after a buy
  reliably reads 0), so there's no point in the pipeline where sniper could
  ever act on this signal. social_watch already waits, so it can.

Deliberately separate from auto_tuner.py's own liquidity/socials logic
rather than bolted onto it - this looks at a different strategy's own
trades (social_watch, not sniper) and a config field that only exists on
SocialWatchConfig.
"""
from __future__ import annotations

import json
import statistics

from .activity_log import DATA_LOG_PATH
from .fees import net_pct_change_after_fees

MIN_SAMPLES = 15
MARGIN_PCT = 10.0

# ascending order matters - decide_min_holder_count picks the highest
# threshold that clears the bar, so later entries must have larger min_count
HOLDER_COUNT_BUCKETS = (
    ("10+", 10),
    ("16+", 16),
    ("25+", 25),
)


def compute_holder_count_outcomes(log_path=DATA_LOG_PATH) -> list[tuple[int, float]]:
    """Returns [(holder_count, net_pct_change_after_fees), ...] for every
    social_watch buy that has both a logged holder_count and a completed
    exit with a measured pct_change."""
    social_watch_buys: dict[str, int] = {}
    exits: dict[str, float] = {}
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
                if d.get("type") == "trade" and d.get("action") == "buy":
                    if d.get("strategy") != "social_watch":
                        continue
                    hc = (d.get("meta") or {}).get("holder_count")
                    mint = d.get("mint")
                    if hc is not None and mint:
                        social_watch_buys[mint] = hc
                elif d.get("type") == "exit":
                    mint = d.get("mint")
                    pct = d.get("pct_change")
                    if mint and pct is not None:
                        exits[mint] = pct
    except FileNotFoundError:
        return []

    return [
        (hc, net_pct_change_after_fees(exits[mint]))
        for mint, hc in social_watch_buys.items()
        if mint in exits
    ]


def decide_min_holder_count(
    log_path=DATA_LOG_PATH,
    current_min_holder_count: int = 0,
    min_samples: int = MIN_SAMPLES,
    margin_pct: float = MARGIN_PCT,
) -> list[dict]:
    """Pure decision logic, kept separate from any async loop so it's easy
    to test without needing real time to pass."""
    pairs = compute_holder_count_outcomes(log_path)
    if not pairs:
        return []

    overall_median = statistics.median(pct for _, pct in pairs)

    best_label, best_min_count, best_median, best_count = None, None, None, None
    for label, min_count in HOLDER_COUNT_BUCKETS:
        if min_count <= current_min_holder_count:
            continue  # wouldn't actually tighten anything
        bucket = [pct for hc, pct in pairs if hc >= min_count]
        if len(bucket) < min_samples:
            continue
        median = statistics.median(bucket)
        if best_median is None or median > best_median:
            best_label, best_min_count, best_median, best_count = label, min_count, median, len(bucket)

    if best_min_count is None or best_median - overall_median < margin_pct:
        return []

    return [{
        "field": "min_holder_count",
        "from": current_min_holder_count,
        "to": best_min_count,
        "reason": (
            f"holder_count {best_label}: {round(best_median, 2)}% mediaan (N={best_count}) vs "
            f"algemeen mediaan {round(overall_median, 2)}% (N={len(pairs)})"
        ),
    }]
