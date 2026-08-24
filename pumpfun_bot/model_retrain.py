"""
Periodically retrains sniper_model.py's win-probability model from the
real trade history, so it improves as real trades accumulate without a
manual `./venv/bin/python scripts/train_sniper_model.py` run - user-
requested 2026-08-24: "build an AI in the bot so it learns more
automatically".

Same conservative philosophy as auto_tuner.py:
- Only ever RETRAINS on more real data than before - never invents data,
  never touches config, never gates the buy decision itself (that's
  SniperConfig.model_score_min_to_buy, a config value the user sets).
- Skips the retrain entirely if the labeled row count hasn't grown since
  the last one - no new data means nothing to learn, no point re-running.
- Every meaningful change (first model ever trained, or its beats-
  baseline status flips either direction) is logged AND alerted, so it's
  visible in the dashboard's live feed like anything else the bot does.
  A same-status retrain (still beating baseline, or still not) just logs
  quietly - not every 30-minute tick needs a chat message.
"""
from __future__ import annotations

import asyncio
import logging

from . import sniper_model
from .alerts import Alerter

logger = logging.getLogger("pumpfun_bot.model_retrain")

RETRAIN_INTERVAL_SEC = 1800


async def retrain_loop(alerter: Alerter, interval_sec: int = RETRAIN_INTERVAL_SEC) -> None:
    last_n: int | None = None
    last_beats_baseline: bool | None = None

    while True:
        await asyncio.sleep(interval_sec)
        try:
            corrections = sniper_model.load_corrections()
            features, labels = sniper_model.load_labeled_dataset(corrections=corrections)
            n = len(features)
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            logger.exception("Kon trainingsdata niet inlezen voor automatische retrain.")
            continue

        if n < sniper_model.MIN_TRAINING_ROWS or n == last_n:
            continue

        try:
            result = sniper_model.retrain_and_save()
        except Exception:  # noqa: BLE001
            logger.exception("Automatische retrain van sniper_model mislukt.")
            continue
        if result is None:
            continue

        last_n = result["n"]
        beats_baseline = result["beats_baseline"]
        status_changed = last_beats_baseline is not None and beats_baseline != last_beats_baseline
        first_model = last_beats_baseline is None
        last_beats_baseline = beats_baseline

        logger.info(
            "Model automatisch hertraind: N=%d, holdout accuracy=%.2f%% (baseline=%.2f%%, "
            "beats_baseline=%s)",
            result["n"], result["accuracy"] * 100, result["baseline"] * 100, beats_baseline,
        )

        if first_model or status_changed:
            verb = "beter dan" if beats_baseline else "NIET beter dan"
            message = (
                f"🧠 Sniper-model automatisch hertraind op {result['n']} echte trades - "
                f"holdout accuracy {result['accuracy']:.1%} ({verb} baseline "
                f"{result['baseline']:.1%})."
            )
            logger.warning(message)
            await alerter.send(message)
