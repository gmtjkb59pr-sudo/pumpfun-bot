import unittest

from pumpfun_bot.auto_tuner import MIN_SAMPLES, decide_adjustments


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


if __name__ == "__main__":
    unittest.main()
