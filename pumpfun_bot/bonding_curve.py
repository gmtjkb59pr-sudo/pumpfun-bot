"""
pump.fun bonding-curve math: constant-product AMM on virtual SOL/token reserves.

Official curve (pump.fun docs / Global account):
    k = virtual_sol * virtual_token
    spot (SOL per token) = virtual_sol / virtual_token

Initial virtual reserves (UI units, matching PumpPortal's vSolInBondingCurve /
vTokensInBondingCurve — not lamports / raw base units):
    virtual SOL    = 30
    virtual tokens = 1_073_000_000

A buy of sol_in SOL:
    tokens_out = (sol_in * virtual_token) / (virtual_sol + sol_in)
A sell of tokens_in tokens:
    sol_out = (tokens_in * virtual_sol) / (virtual_token + tokens_in)

The market-maker used to treat last-price * (1 ± spacing%) as the grid. That
ignores price impact: a real fill of order_size_sol walks the curve, so the
executable average price is strictly worse than spot. Grid levels here are
the actual average fill at that target spot for the configured order size.
"""
from __future__ import annotations

from .pumpportal_client import MissingPumpPortalFieldError, require_event_float

# pump.fun Global account, converted from raw (lamports / 6-decimal tokens)
# to the same UI scale PumpPortal's WS fields use.
INITIAL_VIRTUAL_SOL = 30.0
INITIAL_VIRTUAL_TOKENS = 1_073_000_000.0
INITIAL_REAL_TOKENS = 793_100_000.0


class BondingCurve:
    def __init__(self, virtual_sol: float, virtual_token: float):
        if virtual_sol <= 0 or virtual_token <= 0:
            raise ValueError("virtual reserves moeten positief zijn")
        self.virtual_sol = float(virtual_sol)
        self.virtual_token = float(virtual_token)

    @classmethod
    def initial(cls) -> "BondingCurve":
        return cls(INITIAL_VIRTUAL_SOL, INITIAL_VIRTUAL_TOKENS)

    @classmethod
    def from_pumpportal_event(cls, event: dict) -> "BondingCurve":
        """Fails loud if vSolInBondingCurve / vTokensInBondingCurve are
        missing — never fall back to last-price or marketCapSol, which are
        a different scale and produced the naive grid this module replaces."""
        vsol = require_event_float(event, "vSolInBondingCurve")
        vtok = require_event_float(event, "vTokensInBondingCurve")
        if vsol <= 0 or vtok <= 0:
            raise MissingPumpPortalFieldError(
                "PumpPortal bonding-curve reserves zijn 0 of negatief — "
                "geen geldige curve, niet behandelen als last-price."
            )
        return cls(vsol, vtok)

    @property
    def k(self) -> float:
        return self.virtual_sol * self.virtual_token

    def spot_price(self) -> float:
        """SOL per token (not market-cap)."""
        return self.virtual_sol / self.virtual_token

    def tokens_out_for_sol_in(self, sol_in: float) -> float:
        if sol_in <= 0:
            raise ValueError("sol_in moet positief zijn")
        return (sol_in * self.virtual_token) / (self.virtual_sol + sol_in)

    def sol_out_for_tokens_in(self, tokens_in: float) -> float:
        if tokens_in <= 0:
            raise ValueError("tokens_in moet positief zijn")
        return (tokens_in * self.virtual_sol) / (self.virtual_token + tokens_in)

    def avg_buy_price(self, sol_in: float) -> float:
        """Average SOL/token paid when buying with sol_in SOL. Always above spot."""
        tokens = self.tokens_out_for_sol_in(sol_in)
        return sol_in / tokens

    def avg_sell_price(self, tokens_in: float) -> float:
        """Average SOL/token received when selling tokens_in. Always below spot."""
        sol_out = self.sol_out_for_tokens_in(tokens_in)
        return sol_out / tokens_in

    def at_spot(self, target_spot: float) -> "BondingCurve":
        """Same k, reserves moved so spot equals target_spot.
        virtual_sol' = sqrt(k * target_spot)."""
        if target_spot <= 0:
            raise ValueError("target_spot moet positief zijn")
        virtual_sol = (self.k * target_spot) ** 0.5
        virtual_token = self.k / virtual_sol
        return BondingCurve(virtual_sol, virtual_token)

    def apply_buy(self, sol_in: float) -> float:
        tokens = self.tokens_out_for_sol_in(sol_in)
        self.virtual_sol += sol_in
        self.virtual_token -= tokens
        return tokens

    def apply_sell(self, tokens_in: float) -> float:
        sol_out = self.sol_out_for_tokens_in(tokens_in)
        self.virtual_token += tokens_in
        self.virtual_sol -= sol_out
        return sol_out
