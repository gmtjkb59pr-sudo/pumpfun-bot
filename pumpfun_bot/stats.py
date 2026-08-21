"""
Aggregates activity_log.jsonl into simple learning stats: how simulated
snipes performed at each outcome checkpoint (price drift, informational
only), broken down by the entry-time features logged alongside each trade,
plus the realized P&L from actual exits (take-profit/stop-loss/timeout).

Reports both mean and median: pump.fun outcomes are extremely fat-tailed
(a couple of huge winners can drag the mean up 1000%+ while the typical
trade is flat or negative), so mean alone is actively misleading.

Pure function over the log file so it's easy to test and reuse from both
the dashboard API and a standalone script.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .fees import net_pct_change_after_fees
from .outcome_tracker import CHECKPOINTS_SEC


def _liquidity_bucket(liquidity_sol: float | None) -> str:
    if liquidity_sol is None:
        return "unknown"
    if liquidity_sol < 5:
        return "<5 SOL"
    if liquidity_sol < 20:
        return "5-20 SOL"
    return "20+ SOL"


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_pct_change": None, "median_pct_change": None, "win_rate_pct": None}
    return {
        "count": len(values),
        "mean_pct_change": round(statistics.mean(values), 2),
        "median_pct_change": round(statistics.median(values), 2),
        "win_rate_pct": round(100 * sum(1 for v in values if v > 0) / len(values), 1),
    }


def compute_stats(log_path: str | Path) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        return {
            "total_trades": 0,
            "total_outcomes": 0,
            "total_unmeasured": 0,
            "by_checkpoint": {},
            "exits": {
                "total": 0,
                "total_realized_pnl_sol": 0.0,
                "total_realized_pnl_sol_after_fees": 0.0,
                "by_reason": {},
                "by_reason_after_fees": {},
            },
        }

    trade_meta_by_mint: dict[str, dict] = {}
    outcomes = []
    exits = []
    post_exit_checks = []
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
            elif record.get("type") == "exit":
                exits.append(record)
            elif record.get("type") == "post_exit_check":
                if record.get("vs_realized_pct") is not None:
                    post_exit_checks.append(record)

    by_checkpoint: dict[str, dict] = {}
    for cp in CHECKPOINTS_SEC:
        cp_key = str(cp)
        overall_vals: list[float] = []
        socials_vals = {"true": [], "false": []}
        liquidity_vals: dict[str, list[float]] = {}

        for outcome in outcomes:
            if outcome.get("checkpoint_sec") != cp:
                continue
            pct = outcome.get("pct_change")
            if pct is None:
                continue
            meta = trade_meta_by_mint.get(outcome.get("mint"), {})

            overall_vals.append(pct)
            socials_vals["true" if meta.get("has_socials") else "false"].append(pct)
            liquidity_vals.setdefault(_liquidity_bucket(meta.get("liquidity_sol")), []).append(pct)

        by_checkpoint[cp_key] = {
            "overall": _summarize(overall_vals),
            "by_socials": {k: _summarize(v) for k, v in socials_vals.items()},
            "by_liquidity": {k: _summarize(v) for k, v in liquidity_vals.items()},
        }

    # "gross" = raw price movement, same number the rest of this file has
    # always shown. "net" = gross with pump.fun's 1.25%/trade + PumpPortal's
    # 0.5%/trade round-trip fee deducted (see fees.py for sources) - still
    # missing slippage, which has no fixed published rate to apply.
    exits_by_reason: dict[str, list[float]] = {}
    exits_by_reason_net: dict[str, list[float]] = {}
    total_realized_pnl_sol = 0.0
    total_realized_pnl_sol_after_fees = 0.0
    for exit_record in exits:
        reason = exit_record.get("reason", "unknown")
        pct = exit_record.get("pct_change")
        trade_size = exit_record.get("trade_size_sol") or 0.0
        if pct is not None:
            exits_by_reason.setdefault(reason, []).append(pct)
            net_pct = net_pct_change_after_fees(pct)
            exits_by_reason_net.setdefault(reason, []).append(net_pct)
            if trade_size:
                total_realized_pnl_sol += trade_size * (pct / 100)
                total_realized_pnl_sol_after_fees += trade_size * (net_pct / 100)

    # vs_realized_pct: positive = holding past the exit would have made MORE
    # than the exit strategy actually realized (exiting was premature);
    # negative = exiting was the right call. Broken down by exit reason and
    # how long after the exit we're looking, same (60/300/900s) cadence.
    counterfactual_by_checkpoint: dict[str, dict] = {}
    for cp in CHECKPOINTS_SEC:
        by_reason_vals: dict[str, list[float]] = {}
        for check in post_exit_checks:
            if check.get("checkpoint_sec_after_exit") != cp:
                continue
            reason = check.get("exit_reason", "unknown")
            by_reason_vals.setdefault(reason, []).append(check["vs_realized_pct"])
        counterfactual_by_checkpoint[str(cp)] = {
            reason: _summarize(vals) for reason, vals in by_reason_vals.items()
        }

    return {
        "total_trades": len(trade_meta_by_mint),
        "total_outcomes": len(outcomes),
        "total_unmeasured": unmeasured_count,
        "by_checkpoint": by_checkpoint,
        "exits": {
            "total": len(exits),
            "total_realized_pnl_sol": round(total_realized_pnl_sol, 6),
            "total_realized_pnl_sol_after_fees": round(total_realized_pnl_sol_after_fees, 6),
            "by_reason": {k: _summarize(v) for k, v in exits_by_reason.items()},
            "by_reason_after_fees": {k: _summarize(v) for k, v in exits_by_reason_net.items()},
        },
        "counterfactual_hold": counterfactual_by_checkpoint,
    }
