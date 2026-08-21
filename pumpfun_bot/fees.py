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

Slippage is deliberately NOT modeled here. Unlike the fees above, there is
no published fixed slippage rate to import - actual slippage depends on
trade size vs. bonding-curve depth at the moment of execution, which this
bot doesn't have historical data for. config.yaml's default_slippage_pct
(10%) is a MAXIMUM tolerance the trade fails beyond, not a typical/average
cost - treat any fee-adjusted number here as still missing that cost.
"""
from __future__ import annotations

PUMPFUN_FEE_PCT = 1.25
PUMPPORTAL_LOCAL_API_FEE_PCT = 0.5
FEE_PCT_PER_LEG = PUMPFUN_FEE_PCT + PUMPPORTAL_LOCAL_API_FEE_PCT  # 1.75%, paid on buy AND sell


def net_pct_change_after_fees(gross_pct_change: float) -> float:
    """Compounds the round-trip fee drag onto a gross (pre-fee) percentage
    change: pay FEE_PCT_PER_LEG on the buy, let the position move by
    gross_pct_change, pay FEE_PCT_PER_LEG again on the sell. Does not
    include slippage - see module docstring."""
    leg = FEE_PCT_PER_LEG / 100
    gross_multiplier = 1 + gross_pct_change / 100
    net_multiplier = (1 - leg) * gross_multiplier * (1 - leg)
    return round((net_multiplier - 1) * 100, 2)
