"""
Real, sourced trading costs applied to simulated P&L so the numbers reflect
what a trade would actually cost, not just raw price movement.

Sourced fees (bonding-curve tokens - what this bot trades; PumpSwap/graduated
tokens have a different, lower fee schedule this bot doesn't currently
handle):

- pump.fun protocol fee: 1.25% per trade (buy or sell) on the bonding curve.
  Confirmed via https://froglabs.io/blog/pump-fun-fees-explained/ and
  corroborated by multiple independent sources; not found stated on
  pump.fun's own docs page in a fetchable form, so treat as "widely
  reported," not "pump.fun's own text."
- PumpPortal Local Trading API fee: 0.5% per trade. Confirmed directly from
  PumpPortal's own docs: https://pumpportal.fun/fees ("We take a 0.5% fee
  on each Local trade.")
- Priority fee: a flat SOL amount attached to every transaction we submit
  (see pumpportal_client.py's priority_fee_sol default) to get included
  faster - this is OUR OWN configured value, not looked up externally, but
  it's a real cost paid on-chain on both the buy and the sell leg.

Slippage was deliberately NOT modeled here at first - unlike the fees
above, there's no published fixed slippage rate to import, and actual
slippage depends on trade size vs. bonding-curve depth at the moment of
execution. config.yaml's default_slippage_pct (10%) is a MAXIMUM tolerance
the trade fails beyond, not a typical/average cost.

User-requested 2026-08-24 ("how can i let the sniper run closer to the
dry run when going live" -> "calibrate"): this bot now DOES have real
historical data for it - data/real_pnl_corrections.json (produced by
audit_real_pnl.py) records both the tick-based pct_change that decided
each real exit AND what it actually netted on-chain, for every real trade
this bot has ever closed. See DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON and
apply_dry_run_slippage_penalty below - a real, sourced correction instead
of a guess, though still an average, not a per-trade prediction.
"""
from __future__ import annotations

PUMPFUN_FEE_PCT = 1.25
PUMPPORTAL_LOCAL_API_FEE_PCT = 0.5
FEE_PCT_PER_LEG = PUMPFUN_FEE_PCT + PUMPPORTAL_LOCAL_API_FEE_PCT  # 1.75%, paid on buy AND sell

# must match the priority_fee_sol default in pumpportal_client.py - single
# source of truth so P&L tracking can't silently drift from what's actually
# submitted on-chain
PRIORITY_FEE_SOL_PER_LEG = 0.0005
ROUND_TRIP_PRIORITY_FEE_SOL = PRIORITY_FEE_SOL_PER_LEG * 2


def net_pct_change_after_fees(gross_pct_change: float) -> float:
    """Compounds the round-trip fee drag onto a gross (pre-fee) percentage
    change: pay FEE_PCT_PER_LEG on the buy, let the position move by
    gross_pct_change, pay FEE_PCT_PER_LEG again on the sell. Does not
    include slippage - see apply_dry_run_slippage_penalty below for that,
    applied separately BEFORE this."""
    leg = FEE_PCT_PER_LEG / 100
    gross_multiplier = 1 + gross_pct_change / 100
    net_multiplier = (1 - leg) * gross_multiplier * (1 - leg)
    return round((net_multiplier - 1) * 100, 2)


# Sourced 2026-08-24 from data/real_pnl_corrections.json (813 real corrected
# trades at time of computation) via scripts/calibrate_dry_run_slippage.py -
# re-run that script and update this dict as more real trades accumulate.
# Each value is the average (real corrected_pct_change - original tick-based
# pct_change) actually observed for that exit reason on THIS bot - i.e. how
# far real execution has actually run from the naive WS-tick estimate,
# historically, not a theoretical slippage model. Smaller-n reasons
# (take_profit_ladder n=6, timeout n=4) are noisier and more likely to shift
# with more data than the larger ones (scam_socials n=263, stale_price
# n=217). Any reason not in this dict (e.g. timeout_unmeasured, where
# pct_change is None and there's nothing to correct) gets no adjustment.
DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON = {
    "scam_socials": -26.44,
    "stale_price": -21.53,
    "take_profit": -87.47,
    "trailing_stop": -49.88,
    "stop_loss": -7.53,
    "stale_price_unmeasured": -21.01,
    "take_profit_ladder": -133.47,
    "timeout": -40.84,
}


def apply_dry_run_slippage_penalty(pct_change: float, reason: str) -> float:
    """Subtracts the real, empirically-measured average gap between a
    tick-based exit trigger and what real execution actually returned, for
    this specific exit reason - see DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON.
    Meant for the estimate-only pnl path only (dry-run, or a live trade
    whose real on-chain delta couldn't be fetched) - a real trade with a
    real on-chain SOL delta already reflects whatever slippage actually
    happened and must never have this subtracted on top."""
    return pct_change + DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON.get(reason, 0.0)
