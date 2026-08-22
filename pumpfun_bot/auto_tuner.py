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
from .holder_count_tuning import decide_min_holder_count
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

EXIT_MIN_SAMPLES = 15
EXIT_MARGIN_PCT = 10.0
MIN_TAKE_PROFIT_PCT = 10.0  # sanity floor - never auto-tune below this


def decide_exit_adjustments(stats: dict, current_take_profit_pct: float) -> list[dict]:
    """Only tunes take_profit_pct, and only downward (lock in gains sooner).

    Stop-loss can't be safely auto-tuned from this data: once a position
    exits on stop-loss we stop watching it, so there's no way to know
    whether a tighter stop would have avoided a bigger loss or cut off a
    recovery - that would need tracking price after a hypothetical earlier
    exit, which isn't built. Take-profit is different: timeout exits (still
    open at 15 min) tell us directly what gains were actually achievable
    without ever hitting the current target, which is a real signal.
    """
    changes: list[dict] = []
    timeout = ((stats.get("exits") or {}).get("by_reason") or {}).get("timeout") or {}
    count = timeout.get("count") or 0
    median = timeout.get("median_pct_change")
    win_rate = timeout.get("win_rate_pct")

    if (
        count >= EXIT_MIN_SAMPLES
        and median is not None
        and win_rate is not None
        and win_rate >= 60.0
        and MIN_TAKE_PROFIT_PCT <= median <= current_take_profit_pct - EXIT_MARGIN_PCT
    ):
        new_take_profit = round(median, 1)
        changes.append({
            "field": "take_profit_pct",
            "from": current_take_profit_pct,
            "to": new_take_profit,
            "reason": (
                f"timeout-exits: {median}% mediaan winst (N={count}, win_rate={win_rate}%) "
                f"blijft ruim onder huidige take_profit_pct ({current_take_profit_pct}%) - "
                f"win eerder vast in plaats van te wachten op een doel dat zelden geraakt wordt"
            ),
        })
    return changes


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

    # median, not mean - pump.fun outcomes are extremely fat-tailed, a
    # couple of huge winners drag the mean up 1000%+ while the typical
    # trade is flat or negative, so mean alone gives false signals here
    overall = cp.get("overall") or {}
    overall_median = overall.get("median_pct_change")

    socials = cp.get("by_socials") or {}
    true_bucket = socials.get("true") or {}
    false_bucket = socials.get("false") or {}
    if (
        not current_require_socials
        and (true_bucket.get("count") or 0) >= MIN_SAMPLES
        and (false_bucket.get("count") or 0) >= MIN_SAMPLES
        and true_bucket.get("median_pct_change") is not None
        and false_bucket.get("median_pct_change") is not None
        and true_bucket["median_pct_change"] - false_bucket["median_pct_change"] >= MARGIN_PCT
    ):
        changes.append({
            "field": "require_socials",
            "from": False,
            "to": True,
            "reason": (
                f"met socials: {true_bucket['median_pct_change']}% mediaan (N={true_bucket['count']}) vs "
                f"zonder socials: {false_bucket['median_pct_change']}% mediaan (N={false_bucket['count']}) "
                f"op {checkpoint_sec}s checkpoint"
            ),
        })

    by_liquidity = cp.get("by_liquidity") or {}
    best_key, best_median, best_count = None, None, None
    for bucket_key, threshold in LIQUIDITY_BUCKET_THRESHOLDS.items():
        bucket = by_liquidity.get(bucket_key) or {}
        if threshold <= current_min_liquidity_sol:
            continue  # wouldn't actually tighten anything
        if (bucket.get("count") or 0) < MIN_SAMPLES or bucket.get("median_pct_change") is None:
            continue
        if best_median is None or bucket["median_pct_change"] > best_median:
            best_key, best_median, best_count = bucket_key, bucket["median_pct_change"], bucket["count"]

    if best_key is not None and overall_median is not None and best_median - overall_median >= MARGIN_PCT:
        changes.append({
            "field": "min_liquidity_sol",
            "from": current_min_liquidity_sol,
            "to": LIQUIDITY_BUCKET_THRESHOLDS[best_key],
            "reason": (
                f"liquiditeit {best_key}: {best_median}% mediaan (N={best_count}) vs "
                f"algemeen mediaan {overall_median}% op {checkpoint_sec}s checkpoint"
            ),
        })

    return changes


class AutoTuner:
    def __init__(
        self,
        sniper_cfg: SniperConfig,
        risk: RiskManager,
        alerter: Alerter,
        outcome_tracker=None,
        social_watch_cfg=None,
        birdeye_movers_cfg=None,
        log_path=DATA_LOG_PATH,
        interval_sec: int = CHECK_INTERVAL_SEC,
        checkpoint_sec: int = DEFAULT_CHECKPOINT_SEC,
    ):
        self.sniper_cfg = sniper_cfg
        self.risk = risk
        self.alerter = alerter
        # optional so entry-filter tuning still works without it - only the
        # take-profit adjustment needs a live OutcomeTracker to mutate
        self.outcome_tracker = outcome_tracker
        # optional - only social_watch's/birdeye_movers' min_holder_count
        # tuning needs these, since that field doesn't exist on every
        # strategy's config (see holder_count_tuning.py for why this can't
        # apply to sniper at all)
        self.social_watch_cfg = social_watch_cfg
        self.birdeye_movers_cfg = birdeye_movers_cfg
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
            if self.outcome_tracker is not None:
                changes += decide_exit_adjustments(
                    stats, current_take_profit_pct=self.outcome_tracker.take_profit_pct
                )
            if self.social_watch_cfg is not None:
                changes += decide_min_holder_count(
                    self.log_path,
                    current_min_holder_count=self.social_watch_cfg.min_holder_count,
                    strategy="social_watch",
                )
            if self.birdeye_movers_cfg is not None:
                changes += decide_min_holder_count(
                    self.log_path,
                    current_min_holder_count=self.birdeye_movers_cfg.min_holder_count,
                    strategy="birdeye_movers",
                    field_name="birdeye_movers_min_holder_count",
                )
            for change in changes:
                await self._apply(change)

    async def _apply(self, change: dict) -> None:
        field = change["field"]
        if field == "require_socials":
            self.sniper_cfg.require_socials = change["to"]
        elif field == "min_liquidity_sol":
            self.risk.cfg.min_liquidity_sol = change["to"]
        elif field == "take_profit_pct" and self.outcome_tracker is not None:
            self.outcome_tracker.take_profit_pct = change["to"]
        elif field == "min_holder_count" and self.social_watch_cfg is not None:
            self.social_watch_cfg.min_holder_count = change["to"]
        elif field == "birdeye_movers_min_holder_count" and self.birdeye_movers_cfg is not None:
            self.birdeye_movers_cfg.min_holder_count = change["to"]
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
