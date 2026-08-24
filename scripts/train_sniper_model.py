"""
Trains sniper_model.py's shadow-mode win-probability model from the real
trade history in data/activity_log.jsonl, and writes data/sniper_model.json.

Usage:
    ./venv/bin/python scripts/train_sniper_model.py

Deliberately refuses to train (and says why) below MIN_TRAINING_ROWS - a
model "trained" on a handful of examples is worse than no model at all,
see sniper_model.py's module docstring for the full reasoning on why this
stays shadow-mode/advisory rather than gating real buys.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pumpfun_bot.sniper_model import (  # noqa: E402
    WIN_MARGIN_PCT,
    build_creator_win_rates,
    extract_features,
    save_model,
    score_with_model,
    train_logistic_regression,
)

ACTIVITY_LOG_PATH = Path("data/activity_log.jsonl")
MIN_TRAINING_ROWS = 30
HOLDOUT_FRACTION = 0.2


def _load_labeled_dataset(activity_log_path: Path) -> tuple[list[list[float]], list[int]]:
    buys: dict[str, list[dict]] = defaultdict(list)
    exits_by_mint: dict[str, dict] = {}

    with activity_log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("type") == "trade" and record.get("action") == "buy"
                and record.get("strategy") == "sniper" and record.get("dry_run") is False
            ):
                buys[record["mint"]].append(record)
            elif record.get("type") == "exit" and record.get("dry_run") is False:
                exits_by_mint[record["mint"]] = record

    creator_win_rates = build_creator_win_rates(activity_log_path)

    features: list[list[float]] = []
    labels: list[int] = []
    for mint, mint_buys in buys.items():
        exit_record = exits_by_mint.get(mint)
        if exit_record is None or exit_record.get("pct_change") is None:
            continue
        meta = mint_buys[-1].get("meta") or {}
        feature_values = extract_features(meta, creator_win_rates)
        if feature_values is None:
            continue
        features.append(feature_values)
        labels.append(1 if exit_record["pct_change"] > WIN_MARGIN_PCT else 0)

    return features, labels


def main() -> None:
    if not ACTIVITY_LOG_PATH.exists():
        print(f"Geen {ACTIVITY_LOG_PATH} gevonden - niets om op te trainen.")
        return

    features, labels = _load_labeled_dataset(ACTIVITY_LOG_PATH)
    n = len(features)
    print(f"{n} gelabelde real sniper trade(s) gevonden (met initial_buy_pct beschikbaar).")

    if n < MIN_TRAINING_ROWS:
        print(
            f"Minder dan {MIN_TRAINING_ROWS} rijen - te weinig om een model te vertrouwen, "
            f"stopt zonder te trainen. Blijft draaien met de bestaande harde filters totdat "
            f"er meer echte trades gelogd zijn (initial_buy_pct wordt nu wel al gelogd voor "
            f"elke nieuwe live buy, zie sniper.py)."
        )
        return

    holdout_n = max(1, int(n * HOLDOUT_FRACTION))
    train_features, holdout_features = features[:-holdout_n], features[-holdout_n:]
    train_labels, holdout_labels = labels[:-holdout_n], labels[-holdout_n:]

    model = train_logistic_regression(train_features, train_labels)

    correct = 0
    for feature_values, label in zip(holdout_features, holdout_labels):
        predicted = 1 if score_with_model(model, feature_values) >= 0.5 else 0
        correct += predicted == label
    accuracy = correct / len(holdout_labels) if holdout_labels else 0.0
    baseline = max(sum(train_labels), len(train_labels) - sum(train_labels)) / len(train_labels)

    print(f"Train op {len(train_features)} rijen, holdout {len(holdout_features)} rijen.")
    print(f"Holdout accuracy: {accuracy:.2%} (baseline, altijd de meerderheidsklasse gokken: {baseline:.2%})")
    if accuracy <= baseline:
        print(
            "Model doet het NIET beter dan gewoon de meerderheidsklasse gokken - "
            "nog niet nuttig, maar toch opgeslagen zodat sniper.py's schaduwmodus "
            "de score kan blijven loggen voor verdere validatie."
        )

    save_model(model)
    print("Model opgeslagen naar data/sniper_model.json")


if __name__ == "__main__":
    main()
