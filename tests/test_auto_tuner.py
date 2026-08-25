import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pumpfun_bot.auto_tuner import (
    EXIT_MIN_SAMPLES,
    MIN_SAMPLES,
    AutoTuner,
    decide_adjustments,
    decide_exit_adjustments,
    load_persisted_changes,
)


def _write_log(records: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for record in records:
        f.write(json.dumps(record) + "\n")
    f.close()
    return f.name


def _autotune_change(ts, field, from_, to, reason="test"):
    return {"type": "autotune_change", "ts": ts, "field": field, "from": from_, "to": to, "reason": reason}


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


class LoadPersistedChangesTests(unittest.TestCase):
    """User-requested 2026-08-25 ("fix" the restart-wipes-tuning gap):
    every applied change was always durably logged as an "autotune_change"
    record (for the dashboard feed) even before this existed - this reads
    that history back so AutoTuner can restore its own prior tightening
    across a restart instead of starting over from config.yaml every time."""

    def test_returns_the_latest_change_per_field(self):
        log_path = _write_log([
            _autotune_change(100, "min_liquidity_sol", 5.0, 20.0),
            _autotune_change(200, "require_socials", False, True),
            _autotune_change(300, "min_liquidity_sol", 20.0, 25.0),  # tightened again, later
        ])
        result = load_persisted_changes(log_path)
        self.assertEqual(result["min_liquidity_sol"]["to"], 25.0)
        self.assertEqual(result["require_socials"]["to"], True)

    def test_ignores_unrelated_activity_log_records(self):
        log_path = _write_log([
            {"type": "trade", "action": "buy", "mint": "MINT"},
            {"type": "exit", "mint": "MINT", "pct_change": 10.0},
            _autotune_change(100, "min_liquidity_sol", 5.0, 20.0),
        ])
        result = load_persisted_changes(log_path)
        self.assertEqual(list(result.keys()), ["min_liquidity_sol"])

    def test_missing_log_file_returns_empty_not_an_error(self):
        result = load_persisted_changes("/tmp/definitely-does-not-exist-auto-tuner.jsonl")
        self.assertEqual(result, {})

    def test_empty_log_returns_empty(self):
        log_path = _write_log([])
        result = load_persisted_changes(log_path)
        self.assertEqual(result, {})


class RestorePersistedChangesTests(unittest.TestCase):
    """Real bug found live 2026-08-25: this live-trading bot restarted
    every ~5-20 minutes all session (once per shipped fix) - AutoTuner's
    own 60s check interval likely fired plenty, but _apply() only ever
    mutated the running process's in-memory config, never anything
    durable, so every restart silently discarded whatever it had learned.
    restore_persisted_changes replays the SAME "autotune_change" history
    _apply() already logs, once at startup, before the live tuning loop
    begins - see AutoTuner.run()."""

    def test_restores_a_tightened_liquidity_threshold(self):
        log_path = _write_log([_autotune_change(100, "min_liquidity_sol", 5.0, 20.0)])
        risk = MagicMock(cfg=SimpleNamespace(min_liquidity_sol=5.0))
        tuner = _make_tuner(risk=risk, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        self.assertEqual(risk.cfg.min_liquidity_sol, 20.0)

    def test_restores_require_socials(self):
        log_path = _write_log([_autotune_change(100, "require_socials", False, True)])
        sniper_cfg = SimpleNamespace(require_socials=False)
        tuner = _make_tuner(sniper_cfg=sniper_cfg, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        self.assertTrue(sniper_cfg.require_socials)

    def test_restores_take_profit_pct_downward(self):
        log_path = _write_log([_autotune_change(100, "take_profit_pct", 50.0, 20.0)])
        outcome_tracker = SimpleNamespace(take_profit_pct=50.0)
        tuner = _make_tuner(outcome_tracker=outcome_tracker, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        self.assertEqual(outcome_tracker.take_profit_pct, 20.0)

    def test_restores_min_holder_count_for_social_watch_and_birdeye_separately(self):
        log_path = _write_log([
            _autotune_change(100, "min_holder_count", 1, 50),
            _autotune_change(100, "birdeye_movers_min_holder_count", 1, 75),
        ])
        social_cfg = SimpleNamespace(min_holder_count=1)
        birdeye_cfg = SimpleNamespace(min_holder_count=1)
        tuner = _make_tuner(social_watch_cfg=social_cfg, birdeye_movers_cfg=birdeye_cfg, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        self.assertEqual(social_cfg.min_holder_count, 50)
        self.assertEqual(birdeye_cfg.min_holder_count, 75)

    def test_never_loosens_when_the_current_config_is_already_stricter(self):
        # e.g. the user hand-edited config.yaml since the persisted change
        # was made - config.yaml's own value must win, never get relaxed
        # back down to an older, looser persisted value
        log_path = _write_log([_autotune_change(100, "min_liquidity_sol", 5.0, 10.0)])
        risk = MagicMock(cfg=SimpleNamespace(min_liquidity_sol=25.0))
        tuner = _make_tuner(risk=risk, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        self.assertEqual(risk.cfg.min_liquidity_sol, 25.0)

    def test_restoring_never_sends_an_alert_or_re_logs_a_change(self):
        # this is a silent replay of a decision already announced when it
        # first happened, not a new one
        log_path = _write_log([_autotune_change(100, "min_liquidity_sol", 5.0, 20.0)])
        risk = MagicMock(cfg=SimpleNamespace(min_liquidity_sol=5.0))
        tuner = _make_tuner(risk=risk, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())

        tuner.alerter.send.assert_not_called()

    def test_no_persisted_history_is_a_clean_no_op(self):
        log_path = _write_log([])
        risk = MagicMock(cfg=SimpleNamespace(min_liquidity_sol=5.0))
        tuner = _make_tuner(risk=risk, log_path=log_path)

        asyncio.run(tuner.restore_persisted_changes())  # must not raise

        self.assertEqual(risk.cfg.min_liquidity_sol, 5.0)
        tuner.alerter.send.assert_not_called()

    def test_run_restores_before_entering_the_live_tuning_loop(self):
        # confirms the wiring in run() itself, not just the method in
        # isolation - patches asyncio.sleep to raise after the first call
        # so the test doesn't actually wait, and to prove restore ran
        # BEFORE that first sleep
        log_path = _write_log([_autotune_change(100, "min_liquidity_sol", 5.0, 20.0)])
        risk = MagicMock(cfg=SimpleNamespace(min_liquidity_sol=5.0))
        tuner = _make_tuner(risk=risk, log_path=log_path)

        class _StopLoop(Exception):
            pass

        async def _fake_sleep(_seconds):
            raise _StopLoop

        with patch("pumpfun_bot.auto_tuner.asyncio.sleep", new=_fake_sleep):
            with self.assertRaises(_StopLoop):
                asyncio.run(tuner.run())

        self.assertEqual(risk.cfg.min_liquidity_sol, 20.0)


if __name__ == "__main__":
    unittest.main()
