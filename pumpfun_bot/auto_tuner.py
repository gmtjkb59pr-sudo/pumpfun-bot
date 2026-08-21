"""
Looks at the outcome stats (see stats.py) and, when the evidence is strong
enough, tightens the sniper's own filters - e.g. requiring socials, or
raising the minimum liquidity threshold - so the bot's behavior actually
adapts to what has worked so far instead of just reporting on it.

Deliberately conservative:
- Only ever TIGHTENS a filter (cuts a worse-performing segment out), never
  loosens one. There's no evidence-based case for automatically relaxing a
  quality filter, only for narrowing based on what's underperformed.
- Requires a minimum sample size per compared bucket before acting, to avoid
  reacting to noise on a handful of trades.
- Every change is logged both to the persistent activity log and as a
  regular alert (so it shows up in the dashboard's live feed like anything
  else the bot does) - nothing changes silently.
- Changes only live in the running process's config objects, never written
  back to config.yaml - a restart reverts to your own configured values.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .activity_log import DATA_LOG_PATH, append_jsonl
from .alerts import Alerter
from .config import SniperConfig
from .risk import RiskManager
from .stats import compute_stats

logger = logging.getLogger("pumpfun_bot.auto_tuner")

MIN_SAMPLES = 15
MARGIN_PCT = 10.0
CHECK_INTERVAL_SEC = 60
DEFAULT_CHECKPOINT_SEC = 300

# liquidity bucket label (from stats.py) -> the min_liquidity_sol value that
# bucket represents the lower edge of
LIQUIDITY_BUCKET_THRESHOLDS = {
    "5-20 SOL": 5.0,
    "20+ SOL": 20.0,
}


def decide_adjustments(
    stats: dict,
    current_min_liquidity_sol: float,
    current_require_socials: bool,
    checkpoint_sec: int = DEFAULT_CHECKPOINT_SEC,
) -> list[dict]:
    """Pure decision logic, kept separate from the async loop so it's easy to
    test without needing real time to pass or a live websocket."""
    changes: list[dict] = []
    cp = (stats.get("by_checkpoint") or {}).get(str(checkpoint_sec))
    if not cp:
        return changes

    overall = cp.get("overall") or {}
    overall_avg = overall.get("avg_pct_change")

    socials = cp.get("by_socials") or {}
    true_bucket = socials.get("true") or {}
    false_bucket = socials.get("false") or {}
    if (
        not current_require_socials
        and (true_bucket.get("count") or 0) >= MIN_SAMPLES
        and (false_bucket.get("count") or 0) >= MIN_SAMPLES
        and true_bucket.get("avg_pct_change") is not None
        and false_bucket.get("avg_pct_change") is not None
        and true_bucket["avg_pct_change"] - false_bucket["avg_pct_change"] >= MARGIN_PCT
    ):
        changes.append({
            "field": "require_socials",
            "from": False,
            "to": True,
            "reason": (
                f"met socials: {true_bucket['avg_pct_change']}% gem. (N={true_bucket['count']}) vs "
                f"zonder socials: {false_bucket['avg_pct_change']}% gem. (N={false_bucket['count']}) "
                f"op {checkpoint_sec}s checkpoint"
            ),
        })

    by_liquidity = cp.get("by_liquidity") or {}
    best_key, best_avg, best_count = None, None, None
    for bucket_key, threshold in LIQUIDITY_BUCKET_THRESHOLDS.items():
        bucket = by_liquidity.get(bucket_key) or {}
        if threshold <= current_min_liquidity_sol:
            continue  # wouldn't actually tighten anything
        if (bucket.get("count") or 0) < MIN_SAMPLES or bucket.get("avg_pct_change") is None:
            continue
        if best_avg is None or bucket["avg_pct_change"] > best_avg:
            best_key, best_avg, best_count = bucket_key, bucket["avg_pct_change"], bucket["count"]

    if best_key is not None and overall_avg is not None and best_avg - overall_avg >= MARGIN_PCT:
        changes.append({
            "field": "min_liquidity_sol",
            "from": current_min_liquidity_sol,
            "to": LIQUIDITY_BUCKET_THRESHOLDS[best_key],
            "reason": (
                f"liquiditeit {best_key}: {best_avg}% gem. (N={best_count}) vs "
                f"algemeen gemiddelde {overall_avg}% op {checkpoint_sec}s checkpoint"
            ),
        })

    return changes


class AutoTuner:
    def __init__(
        self,
        sniper_cfg: SniperConfig,
        risk: RiskManager,
        alerter: Alerter,
        log_path=DATA_LOG_PATH,
        interval_sec: int = CHECK_INTERVAL_SEC,
        checkpoint_sec: int = DEFAULT_CHECKPOINT_SEC,
    ):
        self.sniper_cfg = sniper_cfg
        self.risk = risk
        self.alerter = alerter
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.checkpoint_sec = checkpoint_sec

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_sec)
            stats = compute_stats(self.log_path)
            changes = decide_adjustments(
                stats,
                current_min_liquidity_sol=self.risk.cfg.min_liquidity_sol,
                current_require_socials=self.sniper_cfg.require_socials,
                checkpoint_sec=self.checkpoint_sec,
            )
            for change in changes:
                await self._apply(change)

    async def _apply(self, change: dict) -> None:
        field = change["field"]
        if field == "require_socials":
            self.sniper_cfg.require_socials = change["to"]
        elif field == "min_liquidity_sol":
            self.risk.cfg.min_liquidity_sol = change["to"]
        else:
            logger.warning("Onbekend auto-tune veld genegeerd: %s", field)
            return

        message = (
            f"🧠 Auto-tune: {field} {change['from']} -> {change['to']} "
            f"({change['reason']})"
        )
        logger.warning(message)
        append_jsonl({"type": "autotune_change", "ts": time.time(), **change})
        await self.alerter.send(message)
