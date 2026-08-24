"""
Trains sniper_model.py's win-probability model from the real trade
history in data/activity_log.jsonl, and writes data/sniper_model.json.

Usage:
    ./venv/bin/python scripts/train_sniper_model.py

Thin CLI wrapper around sniper_model.retrain_and_save() - the same
function pumpfun_bot/model_retrain.py calls automatically in the
background loop. Kept as a standalone script for a manual, on-demand
retrain with visible output (e.g. right after a data-quality fix, like
the real-pnl correction).

Deliberately refuses to train (and says why) below
sniper_model.MIN_TRAINING_ROWS - a model "trained" on a handful of
examples is worse than no model at all, see sniper_model.py's module
docstring for the full reasoning on why this stays advisory/gated-only
rather than the sole buy decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pumpfun_bot import sniper_model  # noqa: E402


def main() -> None:
    if not sniper_model.ACTIVITY_LOG_PATH.exists():
        print(f"Geen {sniper_model.ACTIVITY_LOG_PATH} gevonden - niets om op te trainen.")
        return

    result = sniper_model.retrain_and_save()
    if result is None:
        print(
            f"Minder dan {sniper_model.MIN_TRAINING_ROWS} gelabelde rijen - te weinig om een "
            f"model te vertrouwen, stopt zonder te trainen. Blijft draaien met de bestaande "
            f"harde filters totdat er meer echte trades gelogd zijn."
        )
        return

    print(f"{result['n']} gelabelde real sniper trade(s) gevonden.")
    print(f"Train op {result['train_n']} rijen, holdout {result['holdout_n']} rijen.")
    print(
        f"Holdout accuracy: {result['accuracy']:.2%} "
        f"(baseline, altijd de meerderheidsklasse gokken: {result['baseline']:.2%})"
    )
    if not result["beats_baseline"]:
        print(
            "Model doet het NIET beter dan gewoon de meerderheidsklasse gokken - "
            "nog niet nuttig, maar toch opgeslagen zodat sniper.py de score kan "
            "blijven loggen/gebruiken voor verdere validatie."
        )
    print(f"Model opgeslagen naar {sniper_model.MODEL_PATH}")


if __name__ == "__main__":
    main()
