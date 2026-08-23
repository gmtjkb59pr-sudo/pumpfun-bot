import unittest

from pumpfun_bot.price_ref import (
    extract_price_ref,
    extract_price_ref_for_field,
    extract_price_ref_with_field,
)


class ExtractPriceRefTests(unittest.TestCase):
    def test_prefers_market_cap_sol_over_other_fields(self):
        event = {"marketCapSol": 30.0, "vSolInBondingCurve": 5.0, "price": 0.0001}
        self.assertEqual(extract_price_ref(event), 30.0)

    def test_falls_back_through_the_priority_list(self):
        self.assertEqual(extract_price_ref({"vSolInBondingCurve": 5.0}), 5.0)
        self.assertEqual(extract_price_ref({"price": 0.0001}), 0.0001)
        self.assertEqual(extract_price_ref({"initialBuy": 1000.0}), 1000.0)

    def test_returns_none_when_nothing_present(self):
        self.assertIsNone(extract_price_ref({"mint": "MINT"}))

    def test_ignores_unparseable_values(self):
        self.assertIsNone(extract_price_ref({"marketCapSol": "not-a-number"}))

    def test_ignores_zero_and_falls_through(self):
        # a falsy 0 is treated as "not present", not a real reading of zero
        self.assertEqual(extract_price_ref({"marketCapSol": 0, "price": 0.0001}), 0.0001)


class ExtractPriceRefWithFieldTests(unittest.TestCase):
    def test_reports_which_field_supplied_the_value(self):
        ref, field = extract_price_ref_with_field({"marketCapSol": 30.0, "price": 0.0001})
        self.assertEqual(ref, 30.0)
        self.assertEqual(field, "marketCapSol")

    def test_reports_the_field_actually_used_on_fallback(self):
        ref, field = extract_price_ref_with_field({"price": 0.0001})
        self.assertEqual(ref, 0.0001)
        self.assertEqual(field, "price")

    def test_returns_none_none_when_nothing_present(self):
        self.assertEqual(extract_price_ref_with_field({"mint": "MINT"}), (None, None))


class ExtractPriceRefForFieldTests(unittest.TestCase):
    """The core of the -100% bug fix: once a position's field is
    established, only THAT field is ever re-extracted - never a fallback
    to a different, scale-incompatible one."""

    def test_extracts_only_the_given_field(self):
        event = {"marketCapSol": 30.0, "price": 0.0001}
        self.assertEqual(extract_price_ref_for_field(event, "marketCapSol"), 30.0)

    def test_returns_none_when_the_field_is_missing_even_if_others_present(self):
        event = {"price": 0.0001, "vSolInBondingCurve": 5.0}
        self.assertIsNone(extract_price_ref_for_field(event, "marketCapSol"))

    def test_returns_none_for_a_falsy_zero_value(self):
        self.assertIsNone(extract_price_ref_for_field({"marketCapSol": 0}, "marketCapSol"))

    def test_ignores_unparseable_values(self):
        self.assertIsNone(extract_price_ref_for_field({"marketCapSol": "bad"}, "marketCapSol"))


if __name__ == "__main__":
    unittest.main()
