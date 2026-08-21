"""
Aggregates activity_log.jsonl into simple learning stats: how simulated
snipes performed at each outcome checkpoint, broken down by the entry-time
features logged alongside each trade. Pure function over the log file so it's
easy to test and reuse from both the dashboard API and a standalone script.
"""
from __future__ import annotations

import json
from pathlib import Path

from .outcome_tracker import CHECKPOINTS_SEC


def _liquidity_bucket(liquidity_sol: float | None) -> str:
    if liquidity_sol is None:
        return "unknown"
    if liquidity_sol < 5:
        return "<5 SOL"
    if liquidity_sol < 20:
        return "5-20 SOL"
    return "20+ SOL"


def _empty_bucket() -> dict:
    return {"count": 0, "sum_pct_change": 0.0, "wins": 0}


def _summarize(bucket: dict) -> dict:
    count = bucket["count"]
    if count == 0:
        return {"count": 0, "avg_pct_change": None, "win_rate_pct": None}
    return {
        "count": count,
        "avg_pct_change": round(bucket["sum_pct_change"] / count, 2),
        "win_rate_pct": round(100 * bucket["wins"] / count, 1),
    }


def compute_stats(log_path: str | Path) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        return {"total_trades": 0, "total_outcomes": 0, "total_unmeasured": 0, "by_checkpoint": {}}

    trade_meta_by_mint: dict[str, dict] = {}
    outcomes = []
    unmeasured_count = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") == "trade" and record.get("dry_run"):
                trade_meta_by_mint[record["mint"]] = record.get("meta") or {}
            elif record.get("type") == "outcome":
                if record.get("pct_change") is None:
                    unmeasured_count += 1
                else:
                    outcomes.append(record)

    by_checkpoint: dict[str, dict] = {}
    for cp in CHECKPOINTS_SEC:
        cp_key = str(cp)
        overall = _empty_bucket()
        by_socials = {"true": _empty_bucket(), "false": _empty_bucket()}
        by_liquidity: dict[str, dict] = {}

        for outcome in outcomes:
            if outcome.get("checkpoint_sec") != cp:
                continue
            pct = outcome.get("pct_change")
            if pct is None:
                continue
            meta = trade_meta_by_mint.get(outcome.get("mint"), {})

            for bucket in (overall,):
                bucket["count"] += 1
                bucket["sum_pct_change"] += pct
                if pct > 0:
                    bucket["wins"] += 1

            socials_key = "true" if meta.get("has_socials") else "false"
            bucket = by_socials[socials_key]
            bucket["count"] += 1
            bucket["sum_pct_change"] += pct
            if pct > 0:
                bucket["wins"] += 1

            liq_key = _liquidity_bucket(meta.get("liquidity_sol"))
            bucket = by_liquidity.setdefault(liq_key, _empty_bucket())
            bucket["count"] += 1
            bucket["sum_pct_change"] += pct
            if pct > 0:
                bucket["wins"] += 1

        by_checkpoint[cp_key] = {
            "overall": _summarize(overall),
            "by_socials": {k: _summarize(v) for k, v in by_socials.items()},
            "by_liquidity": {k: _summarize(v) for k, v in by_liquidity.items()},
        }

    return {
        "total_trades": len(trade_meta_by_mint),
        "total_outcomes": len(outcomes),
        "total_unmeasured": unmeasured_count,
        "by_checkpoint": by_checkpoint,
    }
