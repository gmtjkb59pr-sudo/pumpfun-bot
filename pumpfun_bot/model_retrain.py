"""
Periodically retrains each per-strategy win-probability model from real
trade history, so they improve as real trades accumulate without a manual
`./venv/bin/python scripts/train_sniper_model.py` run - user-requested
2026-08-24: "build an AI in the bot so it learns more automatically".
Extended the same day to also retrain social_watch_model.py, a second
model built after sniper_model.py's own turned out to underperform its
baseline (see that module's docstring) - social_watch's richer feature set
(holder_count, top10_concentration_pct, market_cap_usd, launch-time bundle
detection) gets its own independent retrain cycle here.

Same conservative philosophy as auto_tuner.py, for EACH model:
- Only ever RETRAINS on more real data than before - never invents data,
  never touches config, never gates the buy decision itself (that's each
  strategy's own model_score_min_to_buy-equivalent, a config value the
  user sets - social_watch doesn't have one yet, sniper does).
- Skips the retrain entirely if the labeled row count hasn't grown since
  that model's own last retrain - no new data means nothing to learn.
- Every meaningful change (first model ever trained, or its beats-
  baseline status flips either direction) is logged AND alerted, so it's
  visible in the dashboard's live feed like anything else the bot does.
  A same-status retrain (still beating baseline, or still not) just logs
  quietly - not every 30-minute tick needs a chat message.
"""
from __future__ import annotations

import asyncio
import logging
from types import ModuleType

from . import social_watch_model, sniper_model
from .alerts import Alerter

logger = logging.getLogger("pumpfun_bot.model_retrain")

RETRAIN_INTERVAL_SEC = 1800


async def _maybe_retrain(
    model_module: ModuleType, label: str, alerter: Alerter, state: dict,
) -> None:
    """state carries last_n/last_beats_baseline ACROSS ticks for this one
    model - a plain dict (not instance attrs) so retrain_loop can hold one
    per model without a class just for two fields."""
    try:
        corrections = model_module.load_corrections()
        features, labels = model_module.load_labeled_dataset(corrections=corrections)
        n = len(features)
    except FileNotFoundError:
        return
    except Exception:  # noqa: BLE001
        logger.exception("Kon trainingsdata niet inlezen voor automatische %s-retrain.", label)
        return

    if n < model_module.MIN_TRAINING_ROWS or n == state["last_n"]:
        return

    try:
        result = model_module.retrain_and_save()
    except Exception:  # noqa: BLE001
        logger.exception("Automatische retrain van %s mislukt.", label)
        return
    if result is None:
        return

    state["last_n"] = result["n"]
    beats_baseline = result["beats_baseline"]
    status_changed = state["last_beats_baseline"] is not None and beats_baseline != state["last_beats_baseline"]
    first_model = state["last_beats_baseline"] is None
    state["last_beats_baseline"] = beats_baseline

    logger.info(
        "%s automatisch hertraind: N=%d, holdout accuracy=%.2f%% (baseline=%.2f%%, "
        "beats_baseline=%s)",
        label, result["n"], result["accuracy"] * 100, result["baseline"] * 100, beats_baseline,
    )

    if first_model or status_changed:
        verb = "beter dan" if beats_baseline else "NIET beter dan"
        message = (
            f"🧠 {label} automatisch hertraind op {result['n']} echte trades - "
            f"holdout accuracy {result['accuracy']:.1%} ({verb} baseline "
            f"{result['baseline']:.1%})."
        )
        logger.warning(message)
        await alerter.send(message)


async def retrain_loop(alerter: Alerter, interval_sec: int = RETRAIN_INTERVAL_SEC) -> None:
    models = (
        (sniper_model, "Sniper-model"),
        (social_watch_model, "Social-watch-model"),
    )
    states = {label: {"last_n": None, "last_beats_baseline": None} for _module, label in models}

    while True:
        await asyncio.sleep(interval_sec)
        for model_module, label in models:
            await _maybe_retrain(model_module, label, alerter, states[label])
