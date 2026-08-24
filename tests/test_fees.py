import unittest

from pumpfun_bot.fees import (
    DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON,
    FEE_PCT_PER_LEG,
    apply_dry_run_slippage_penalty,
    net_pct_change_after_fees,
)


class NetPctChangeAfterFeesTests(unittest.TestCase):
    def test_zero_gross_change_still_loses_the_round_trip_fee(self):
        # even at 0% price movement, you still paid the fee twice
        net = net_pct_change_after_fees(0.0)
        self.assertLess(net, 0.0)
        expected = ((1 - FEE_PCT_PER_LEG / 100) ** 2 - 1) * 100
        self.assertAlmostEqual(net, round(expected, 2), places=2)

    def test_positive_gross_change_shrinks_after_fees(self):
        net = net_pct_change_after_fees(50.0)
        self.assertLess(net, 50.0)
        self.assertGreater(net, 44.0)  # fees are a few percent, not huge

    def test_negative_gross_change_gets_slightly_worse(self):
        net = net_pct_change_after_fees(-25.0)
        self.assertLess(net, -25.0)

    def test_known_round_trip_fee_is_3_5_pct_per_leg_total(self):
        self.assertEqual(FEE_PCT_PER_LEG, 1.75)


class ApplyDryRunSlippagePenaltyTests(unittest.TestCase):
    def test_a_known_reason_gets_its_calibrated_penalty_subtracted(self):
        result = apply_dry_run_slippage_penalty(51.0, "take_profit")
        expected = 51.0 + DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON["take_profit"]
        self.assertAlmostEqual(result, expected)

    def test_an_unknown_reason_gets_no_penalty(self):
        result = apply_dry_run_slippage_penalty(20.0, "some_future_exit_reason")
        self.assertEqual(result, 20.0)

    def test_every_calibrated_penalty_is_negative(self):
        # this is a real, sourced correction toward what execution actually
        # costs, not a random adjustment - every reason here has real
        # trades showing execution landed worse than the trigger tick, so
        # none of these should ever help pnl
        for reason, penalty in DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON.items():
            with self.subTest(reason=reason):
                self.assertLess(penalty, 0.0)

    def test_stop_loss_has_the_smallest_penalty(self):
        # real, sourced finding: stop_loss reacts fastest (tightest
        # threshold, exits at the first sign of trouble) so it has less
        # time to go stale before the real sell lands, unlike trailing_stop/
        # take_profit which wait for a peak first
        stop_loss_penalty = DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON["stop_loss"]
        for reason, penalty in DRY_RUN_SLIPPAGE_PENALTY_PCT_BY_REASON.items():
            if reason == "stop_loss":
                continue
            with self.subTest(reason=reason):
                self.assertLess(penalty, stop_loss_penalty)


if __name__ == "__main__":
    unittest.main()
