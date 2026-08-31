import unittest

from pumpfun_bot.bonding_curve import (
    INITIAL_VIRTUAL_SOL,
    INITIAL_VIRTUAL_TOKENS,
    BondingCurve,
)
from pumpfun_bot.pumpportal_client import MissingPumpPortalFieldError
from pumpfun_bot.strategies.market_maker import GridState


class BondingCurveMathTests(unittest.TestCase):
    def test_initial_spot_matches_official_reserves(self):
        curve = BondingCurve.initial()
        self.assertEqual(curve.virtual_sol, INITIAL_VIRTUAL_SOL)
        self.assertEqual(curve.virtual_token, INITIAL_VIRTUAL_TOKENS)
        expected = INITIAL_VIRTUAL_SOL / INITIAL_VIRTUAL_TOKENS
        self.assertAlmostEqual(curve.spot_price(), expected)

    def test_k_is_conserved_across_a_buy_and_sell(self):
        curve = BondingCurve.initial()
        k0 = curve.k
        tokens = curve.apply_buy(1.0)
        self.assertAlmostEqual(curve.k, k0, places=6)
        curve.apply_sell(tokens)
        self.assertAlmostEqual(curve.k, k0, places=6)
        self.assertAlmostEqual(curve.virtual_sol, INITIAL_VIRTUAL_SOL, places=9)

    def test_tokens_out_matches_constant_product_formula(self):
        curve = BondingCurve(30.0, 1_073_000_000.0)
        sol_in = 0.05
        expected = (sol_in * curve.virtual_token) / (curve.virtual_sol + sol_in)
        self.assertAlmostEqual(curve.tokens_out_for_sol_in(sol_in), expected)

    def test_buy_raises_spot_sell_lowers_spot(self):
        curve = BondingCurve.initial()
        spot0 = curve.spot_price()
        curve.apply_buy(1.0)
        self.assertGreater(curve.spot_price(), spot0)
        curve.apply_sell(1_000_000.0)
        self.assertLess(curve.spot_price(), curve.spot_price() + 1)  # sanity
        self.assertLess(curve.spot_price(), spot0 * 2)

    def test_avg_buy_price_is_worse_than_spot(self):
        curve = BondingCurve.initial()
        self.assertGreater(curve.avg_buy_price(1.0), curve.spot_price())

    def test_avg_sell_price_is_worse_than_spot(self):
        curve = BondingCurve.initial()
        tokens = 1_000_000.0
        self.assertLess(curve.avg_sell_price(tokens), curve.spot_price())

    def test_at_spot_preserves_k_and_hits_target(self):
        curve = BondingCurve.initial()
        target = curve.spot_price() * 0.98
        moved = curve.at_spot(target)
        self.assertAlmostEqual(moved.k, curve.k, places=4)
        self.assertAlmostEqual(moved.spot_price(), target, places=12)

    def test_rejects_non_positive_reserves(self):
        with self.assertRaises(ValueError):
            BondingCurve(0, 1_000_000)
        with self.assertRaises(ValueError):
            BondingCurve(30, 0)


class FromPumpportalEventTests(unittest.TestCase):
    def test_builds_from_virtual_reserves(self):
        curve = BondingCurve.from_pumpportal_event({
            "vSolInBondingCurve": 32.5,
            "vTokensInBondingCurve": 1_000_000_000,
        })
        self.assertEqual(curve.virtual_sol, 32.5)
        self.assertEqual(curve.virtual_token, 1_000_000_000)

    def test_missing_vsol_fails_loud_not_as_zero(self):
        with self.assertRaises(MissingPumpPortalFieldError):
            BondingCurve.from_pumpportal_event({"vTokensInBondingCurve": 1e9})

    def test_missing_vtokens_fails_loud(self):
        with self.assertRaises(MissingPumpPortalFieldError):
            BondingCurve.from_pumpportal_event({"vSolInBondingCurve": 30})

    def test_zero_reserves_fail_loud(self):
        with self.assertRaises(MissingPumpPortalFieldError):
            BondingCurve.from_pumpportal_event({
                "vSolInBondingCurve": 0, "vTokensInBondingCurve": 1e9,
            })

    def test_does_not_fall_back_to_market_cap_sol(self):
        with self.assertRaises(MissingPumpPortalFieldError):
            BondingCurve.from_pumpportal_event({"marketCapSol": 35.0, "price": 1e-8})


class GridExecutablePriceTests(unittest.TestCase):
    """The whole point vs the old last_price * (1 ± spacing%) grid."""

    def test_buy_level_executable_price_is_not_naive_spot_times_spacing(self):
        curve = BondingCurve.initial()
        grid = GridState(curve, levels=5, spacing_pct=2.0, order_size_sol=1.0)
        naive = grid.center_spot * 0.98  # level -1, 2% below
        executable = grid.level_executable_price(-1)
        # a 1 SOL buy has real impact on a 30 SOL virtual reserve
        self.assertNotAlmostEqual(executable, naive, places=12)
        self.assertGreater(executable, naive)

    def test_small_order_executable_price_is_close_to_target_spot(self):
        curve = BondingCurve.initial()
        grid = GridState(curve, levels=5, spacing_pct=2.0, order_size_sol=0.01)
        target = grid.level_spot(-1)
        executable = grid.level_executable_price(-1)
        # 0.01 SOL on 30 SOL reserves: impact is tiny
        self.assertAlmostEqual(executable / target, 1.0, places=3)

    def test_level_zero_is_center(self):
        curve = BondingCurve.initial()
        grid = GridState(curve, levels=5, spacing_pct=2.0, order_size_sol=0.01)
        self.assertAlmostEqual(grid.level_spot(0), grid.center_spot)

    def test_sell_level_executable_price_is_below_target_spot(self):
        curve = BondingCurve.initial()
        grid = GridState(curve, levels=5, spacing_pct=2.0, order_size_sol=1.0)
        target = grid.level_spot(1)
        executable = grid.level_executable_price(1)
        self.assertLess(executable, target)
