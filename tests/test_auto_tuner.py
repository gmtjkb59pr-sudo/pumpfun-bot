import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pumpfun_bot.auto_tuner import (
    EXIT_MIN_SAMPLES,
    MIN_SAMPLES,
    AutoTuner,
    decide_adjustments,
    decide_exit_adjustments,
)


def make_bucket(count, median_pct_change):
    return {"count": count, "median_pct_change": median_pct_change, "win_rate_pct": 50.0}


def make_stats(overall, by_socials=None, by_liquidity=None, checkpoint_sec=300):
    return {
        "by_checkpoint": {
            str(checkpoint_sec): {
                "overall": overall,
                "by_socials": by_socials or {"true": make_bucket(0, None), "false": make_bucket(0, None)},
                "by_liquidity": by_liquidity or {},
            }
        }
    }


class DecideAdjustmentsTests(unittest.TestCase):
    def test_no_changes_when_checkpoint_missing(self):
        changes = decide_adjustments({}, current_min_liquidity_sol=5, current_require_socials=False)
        self.assertEqual(changes, [])

    def test_no_changes_below_min_sample_size(self):
        stats = make_stats(
            overall=make_bucket(10, 5.0),
            by_socials={
                "true": make_bucket(MIN_SAMPLES - 1, 50.0),
                "false": make_bucket(MIN_SAMPLES, 5.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=5, current_require_socials=False)
        self.assertEqual(changes, [])

    def test_proposes_require_socials_when_clearly_better(self):
        stats = make_stats(
            overall=make_bucket(30, 20.0),
            by_socials={
                "true": make_bucket(MIN_SAMPLES, 50.0),
                "false": make_bucket(MIN_SAMPLES, 5.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=5, current_require_socials=False)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "require_socials")
        self.assertEqual(changes[0]["to"], True)

    def test_does_not_repropose_require_socials_when_already_set(self):
        stats = make_stats(
            overall=make_bucket(30, 20.0),
            by_socials={
                "true": make_bucket(MIN_SAMPLES, 50.0),
                "false": make_bucket(MIN_SAMPLES, 5.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=5, current_require_socials=True)
        self.assertEqual(changes, [])

    def test_no_change_when_difference_below_margin(self):
        stats = make_stats(
            overall=make_bucket(30, 20.0),
            by_socials={
                "true": make_bucket(MIN_SAMPLES, 22.0),
                "false": make_bucket(MIN_SAMPLES, 20.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=5, current_require_socials=False)
        self.assertEqual(changes, [])

    def test_proposes_liquidity_tightening_for_best_performing_bucket(self):
        stats = make_stats(
            overall=make_bucket(50, 10.0),
            by_liquidity={
                "<5 SOL": make_bucket(MIN_SAMPLES, -20.0),
                "5-20 SOL": make_bucket(MIN_SAMPLES, 5.0),
                "20+ SOL": make_bucket(MIN_SAMPLES, 60.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=0, current_require_socials=True)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "min_liquidity_sol")
        self.assertEqual(changes[0]["to"], 20.0)

    def test_does_not_propose_liquidity_bucket_at_or_below_current_threshold(self):
        stats = make_stats(
            overall=make_bucket(50, 10.0),
            by_liquidity={
                "5-20 SOL": make_bucket(MIN_SAMPLES, 60.0),
            },
        )
        # already at 5.0, so the "5-20 SOL" bucket (threshold 5.0) wouldn't tighten anything
        changes = decide_adjustments(stats, current_min_liquidity_sol=5.0, current_require_socials=True)
        self.assertEqual(changes, [])

    def test_can_propose_both_changes_at_once(self):
        stats = make_stats(
            overall=make_bucket(50, 10.0),
            by_socials={
                "true": make_bucket(MIN_SAMPLES, 50.0),
                "false": make_bucket(MIN_SAMPLES, 5.0),
            },
            by_liquidity={
                "20+ SOL": make_bucket(MIN_SAMPLES, 60.0),
            },
        )
        changes = decide_adjustments(stats, current_min_liquidity_sol=0, current_require_socials=False)
        fields = {c["field"] for c in changes}
        self.assertEqual(fields, {"require_socials", "min_liquidity_sol"})


def make_exit_stats(timeout_count, timeout_median, timeout_win_rate):
    return {
        "exits": {
            "by_reason": {
                "timeout": {
                    "count": timeout_count,
                    "median_pct_change": timeout_median,
                    "win_rate_pct": timeout_win_rate,
                }
            }
        }
    }


class DecideExitAdjustmentsTests(unittest.TestCase):
    def test_no_change_below_min_samples(self):
        stats = make_exit_stats(EXIT_MIN_SAMPLES - 1, 30.0, 80.0)
        changes = decide_exit_adjustments(stats, current_take_profit_pct=50.0)
        self.assertEqual(changes, [])

    def test_no_change_when_win_rate_too_low(self):
        stats = make_exit_stats(EXIT_MIN_SAMPLES, 30.0, 55.0)
        changes = decide_exit_adjustments(stats, current_take_profit_pct=50.0)
        self.assertEqual(changes, [])

    def test_no_change_when_median_too_close_to_current_target(self):
        # only 5pp below 50 - below the EXIT_MARGIN_PCT of 10
        stats = make_exit_stats(EXIT_MIN_SAMPLES, 45.0, 80.0)
        changes = decide_exit_adjustments(stats, current_take_profit_pct=50.0)
        self.assertEqual(changes, [])

    def test_lowers_take_profit_when_evidence_is_strong(self):
        stats = make_exit_stats(EXIT_MIN_SAMPLES, 25.0, 70.0)
        changes = decide_exit_adjustments(stats, current_take_profit_pct=50.0)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["field"], "take_profit_pct")
        self.assertEqual(changes[0]["from"], 50.0)
        self.assertEqual(changes[0]["to"], 25.0)

    def test_never_proposes_below_sanity_floor(self):
        stats = make_exit_stats(EXIT_MIN_SAMPLES, 5.0, 90.0)  # below MIN_TAKE_PROFIT_PCT
        changes = decide_exit_adjustments(stats, current_take_profit_pct=50.0)
        self.assertEqual(changes, [])


def _make_tuner(**kwargs):
    defaults = dict(
        sniper_cfg=SimpleNamespace(require_socials=False),
        risk=MagicMock(cfg=SimpleNamespace(min_liquidity_sol=5.0)),
        alerter=MagicMock(send=AsyncMock()),
    )
    defaults.update(kwargs)
    return AutoTuner(**defaults)


class ApplyTests(unittest.TestCase):
    def test_routes_birdeye_movers_min_holder_count_to_its_own_config(self):
        birdeye_cfg = SimpleNamespace(min_holder_count=1)
        social_cfg = SimpleNamespace(min_holder_count=1)
        tuner = _make_tuner(social_watch_cfg=social_cfg, birdeye_movers_cfg=birdeye_cfg)

        asyncio.run(tuner._apply({
            "field": "birdeye_movers_min_holder_count", "from": 1, "to": 100, "reason": "test",
        }))

        self.assertEqual(birdeye_cfg.min_holder_count, 100)
        self.assertEqual(social_cfg.min_holder_count, 1)

    def test_plain_min_holder_count_still_routes_to_social_watch_only(self):
        birdeye_cfg = SimpleNamespace(min_holder_count=1)
        social_cfg = SimpleNamespace(min_holder_count=1)
        tuner = _make_tuner(social_watch_cfg=social_cfg, birdeye_movers_cfg=birdeye_cfg)

        asyncio.run(tuner._apply({
            "field": "min_holder_count", "from": 1, "to": 100, "reason": "test",
        }))

        self.assertEqual(social_cfg.min_holder_count, 100)
        self.assertEqual(birdeye_cfg.min_holder_count, 1)

    def test_birdeye_movers_change_is_a_noop_without_a_configured_target(self):
        tuner = _make_tuner()  # no birdeye_movers_cfg passed

        asyncio.run(tuner._apply({
            "field": "birdeye_movers_min_holder_count", "from": 1, "to": 100, "reason": "test",
        }))
        # no exception, no alert sent - unroutable change is dropped, not applied blindly
        tuner.alerter.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
