import unittest

from pumpfun_bot.state import BotState


class RealPnlTrackingTests(unittest.TestCase):
    """Ground-truth wallet balance vs. the bot's own per-trade fee model
    (realized_pnl_sol) - confirmed live: the model read positive while the
    real wallet balance dropped far more in the same window."""

    def test_real_pnl_is_none_before_any_balance_is_recorded(self):
        state = BotState()
        snap = state.snapshot()
        self.assertIsNone(snap["real_pnl_sol"])
        self.assertIsNone(snap["real_pnl_usd"])

    def test_computes_real_pnl_relative_to_session_start(self):
        state = BotState()
        state.set_session_start_balance(1.0, 150.0)
        state.update_real_balance(1.2, 180.0)

        snap = state.snapshot()
        self.assertAlmostEqual(snap["real_pnl_sol"], 0.2)
        self.assertAlmostEqual(snap["real_pnl_usd"], 30.0)

    def test_negative_real_pnl_when_balance_drops(self):
        state = BotState()
        state.set_session_start_balance(1.0, 150.0)
        state.update_real_balance(0.7, 105.0)

        snap = state.snapshot()
        self.assertAlmostEqual(snap["real_pnl_sol"], -0.3)
        self.assertAlmostEqual(snap["real_pnl_usd"], -45.0)

    def test_current_balance_recorded_even_without_a_session_start(self):
        # e.g. the very first startup lookup failed, but a later periodic
        # check succeeds - the raw balance should still be visible even
        # though there's no baseline yet to compute a delta against
        state = BotState()
        state.update_real_balance(0.5, 75.0)

        snap = state.snapshot()
        self.assertEqual(snap["current_balance_sol"], 0.5)
        self.assertIsNone(snap["real_pnl_sol"])

    def test_real_pnl_sol_still_computed_when_usd_price_unavailable(self):
        state = BotState()
        state.set_session_start_balance(1.0, 150.0)
        state.update_real_balance(1.1, None)  # price lookup failed this cycle

        snap = state.snapshot()
        self.assertAlmostEqual(snap["real_pnl_sol"], 0.1)
        self.assertIsNone(snap["real_pnl_usd"])  # can't compute without a price


class UntrackedHoldingsCountTests(unittest.TestCase):
    """open_exposure_sol only ever sums positions the bot tracked opening
    itself - confirmed live: the wallet held real balances of several
    leftover mints never counted there at all, understating how much
    capital is actually tied up. This is the visible flag for that gap."""

    def test_defaults_to_zero(self):
        state = BotState()
        self.assertEqual(state.snapshot()["untracked_holdings_count"], 0)

    def test_reflects_the_latest_count(self):
        state = BotState()
        state.set_untracked_holdings_count(4)
        self.assertEqual(state.snapshot()["untracked_holdings_count"], 4)


if __name__ == "__main__":
    unittest.main()
