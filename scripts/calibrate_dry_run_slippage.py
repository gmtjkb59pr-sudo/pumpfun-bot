"""
Regenerates the per-exit-reason slippage calibration in
pumpfun_bot/fees.py's DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON.

User-requested 2026-08-24 ("how can i let the sniper run closer to the dry
run when going live" -> "calibrate"): dry-run pnl has never modeled
slippage at all (see fees.py's own docstring) - it just trusts whatever
pct_change the WS price tick showed at the moment an exit condition fired.
Real corrected data (data/real_pnl_corrections.json, produced by
audit_real_pnl.py from real on-chain SOL deltas) already has BOTH that
original tick-based pct_change AND what the position actually netted for
every real exit - the gap between them, averaged per exit reason, is a
real, sourced slippage/staleness penalty instead of a guess.

Run this again whenever data/real_pnl_corrections.json has grown
meaningfully (more real trades closed, or after running audit_real_pnl.py
again) and paste the printed dict back into fees.py - deliberately not
computed at import time in fees.py itself, so the live bot's behavior
doesn't silently shift every time this file changes size; this is a
periodic manual calibration step, not a live-updating model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CORRECTIONS_PATH = Path("data/real_pnl_corrections.json")


def main() -> None:
    with CORRECTIONS_PATH.open() as f:
        data = json.load(f)

    by_reason: dict[str, list[float]] = defaultdict(list)
    for record in data.values():
        orig = record.get("original_pct_change")
        corr = record.get("corrected_pct_change")
        reason = record.get("reason")
        if orig is None or corr is None or reason is None:
            continue
        by_reason[reason].append(corr - orig)

    print(f"Bron: {CORRECTIONS_PATH} ({len(data)} records)\n")
    print("DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON = {")
    for reason, gaps in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        avg_gap = sum(gaps) / len(gaps)
        print(f'    "{reason}": {avg_gap:.2f},  # n={len(gaps)}')
    print("}")


if __name__ == "__main__":
    main()
