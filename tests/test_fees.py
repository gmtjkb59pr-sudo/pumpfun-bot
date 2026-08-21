import unittest

from pumpfun_bot.fees import FEE_PCT_PER_LEG, net_pct_change_after_fees


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


if __name__ == "__main__":
    unittest.main()
