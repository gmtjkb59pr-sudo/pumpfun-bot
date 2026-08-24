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


class ExternalTransferAccountingTests(unittest.TestCase):
    """User-requested 2026-08-24 ("the actual profit is not right because
    i topped up the bot with 10 dollar it is actually down 3") -
    real_pnl_sol was just current_balance - session_start_balance, with no
    way to know a manual deposit/withdrawal had landed mid-session. A
    $10 top-up 23 seconds after session start showed up entirely as
    "trading profit" until this."""

    def test_net_external_transfers_defaults_to_zero(self):
        state = BotState()
        self.assertEqual(state.snapshot()["net_external_transfers_sol"], 0.0)

    def test_a_deposit_is_excluded_from_real_pnl(self):
        state = BotState()
        state.set_session_start_balance(0.10, 10.0)
        state.update_real_balance(0.20, 20.0)  # naive delta would read +0.10
        state.add_external_transfer(0.10634)  # the deposit that explains most of it

        snap = state.snapshot()
        # real trading result: (0.20 - 0.10) - 0.10634 = a real loss
        self.assertAlmostEqual(snap["real_pnl_sol"], 0.20 - 0.10 - 0.10634, places=6)
        self.assertLess(snap["real_pnl_sol"], 0)
        self.assertAlmostEqual(snap["net_external_transfers_sol"], 0.10634)

    def test_a_withdrawal_is_added_back_to_real_pnl(self):
        state = BotState()
        state.set_session_start_balance(1.0, 150.0)
        state.update_real_balance(0.5, 75.0)  # naive delta would read -0.5
        state.add_external_transfer(-0.4)  # the user withdrew 0.4 SOL

        snap = state.snapshot()
        # real trading result: (0.5 - 1.0) - (-0.4) = -0.1, not -0.5
        self.assertAlmostEqual(snap["real_pnl_sol"], -0.1, places=6)

    def test_add_external_transfer_recomputes_immediately_not_just_on_next_poll(self):
        state = BotState()
        state.set_session_start_balance(0.10, 10.0)
        state.update_real_balance(0.20, 20.0)
        before = state.snapshot()["real_pnl_sol"]

        state.add_external_transfer(0.10634)

        after = state.snapshot()["real_pnl_sol"]
        self.assertNotAlmostEqual(before, after, places=4)

    def test_multiple_transfers_accumulate(self):
        state = BotState()
        state.set_session_start_balance(1.0, 150.0)
        state.add_external_transfer(0.1)
        state.add_external_transfer(-0.03)
        state.add_external_transfer(0.05)

        self.assertAlmostEqual(state.snapshot()["net_external_transfers_sol"], 0.12, places=6)

    def test_a_transfer_before_session_start_balance_is_captured_does_not_raise(self):
        # add_external_transfer only recomputes real_pnl_sol if both
        # session_start and current balance are already known - must not
        # crash if it fires before either lookup has completed
        state = BotState()
        state.add_external_transfer(0.1)  # must not raise
        self.assertIsNone(state.snapshot()["real_pnl_sol"])
        self.assertAlmostEqual(state.snapshot()["net_external_transfers_sol"], 0.1)


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
