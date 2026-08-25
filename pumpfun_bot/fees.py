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

# user-requested 2026-08-24 ("how can we fix that problem" -> "build 3",
# picking the "faster execution on take_profit sells" option from the
# slippage-calibration finding above): take_profit/take_profit_ladder have
# by far the worst real slippage gap of any exit reason (-87/-133 points),
# and the mechanism behind it is specifically TIME - these exits fire on a
# tick that's already near a local peak on a thin bonding curve that can
# crash within seconds, so the real fill depends heavily on how fast the
# sell actually lands. A bigger priority fee is the direct lever for
# landing speed. The multiplier itself (5x) is a provisional placeholder,
# NOT derived from real measured landing-time-vs-fee data (no real trade
# has ever used a boosted fee yet, since real money isn't currently at
# risk) - revisit once live trades with this fee actually exist. Applied
# only to the SELL leg of take_profit/take_profit_ladder exits (see
# priority_fee_sol_for_sell below) - the buy leg and every other exit
# reason are unaffected.
TAKE_PROFIT_PRIORITY_FEE_SOL_PER_LEG = PRIORITY_FEE_SOL_PER_LEG * 5
_BOOSTED_SELL_PRIORITY_FEE_REASONS = frozenset({"take_profit", "take_profit_ladder"})

# user-requested 2026-08-24 ("how can i make the bot faster" -> "yes"):
# sniper's own buy call never overrode priority_fee_sol at all, so every
# real buy used the flat default below - a real, unaddressed gap given
# sniper's whole documented edge is being first. Same "provisional
# placeholder" epistemic status as TAKE_PROFIT_PRIORITY_FEE_SOL_PER_LEG
# above - no real landing-time-vs-fee data exists for buys yet either.
# Deliberately a SMALLER multiplier (3x, not 5x) than the take_profit
# boost: a buy fee is paid on EVERY candidate sniper attempts, win or
# lose, while take_profit only fires on an already-winning position - the
# same multiplier compounds far more often here, so start more
# conservative and revisit with real data. Scoped to sniper only:
# social_watch already tolerates real delay by design (its watch_window_
# sec), speed isn't its edge the way it is sniper's.
SNIPER_BUY_PRIORITY_FEE_SOL_PER_LEG = PRIORITY_FEE_SOL_PER_LEG * 3


def priority_fee_sol_for_sell(reason: str) -> float:
    """The priority fee to attach to a real sell transaction for this exit
    reason - boosted for take_profit/take_profit_ladder (see
    TAKE_PROFIT_PRIORITY_FEE_SOL_PER_LEG's docstring), the normal flat fee
    for everything else."""
    if reason in _BOOSTED_SELL_PRIORITY_FEE_REASONS:
        return TAKE_PROFIT_PRIORITY_FEE_SOL_PER_LEG
    return PRIORITY_FEE_SOL_PER_LEG


def round_trip_priority_fee_sol_for_reason(reason: str) -> float:
    """Buy leg always pays the normal flat fee (the boost only applies to
    the SELL decision, made after already knowing which exit fired) - sell
    leg pays whatever priority_fee_sol_for_sell says for this reason. Used
    by the dry-run/estimate-only pnl path so its assumed cost matches what
    a real trade with this exit reason would actually submit, instead of
    the flat ROUND_TRIP_PRIORITY_FEE_SOL every other reason still uses."""
    return PRIORITY_FEE_SOL_PER_LEG + priority_fee_sol_for_sell(reason)


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
