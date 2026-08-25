"""
Plain-gradient-descent logistic regression, shared by sniper_model.py and
social_watch_model.py (and any future per-strategy win-probability model) -
extracted 2026-08-24 when building the second model, rather than
duplicating this math a second time. Deliberately pure Python (no numpy/
sklearn dependency) - see sniper_model.py's module docstring for why.

Each strategy owns its own feature extraction, dataset labeling, and
sentinel-default handling (those ARE strategy-specific - a "win" definition
and what raw signals exist at each strategy's own buy-decision moment
differ) - only the actual training/scoring math lives here.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("pumpfun_bot.logistic_regression")


def sigmoid(z: float) -> float:
    if z < -700:  # avoid math.exp overflow on a very negative z
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic_regression(
    features: list[list[float]], labels: list[int], feature_names: list[str],
    epochs: int = 500, lr: float = 0.1,
) -> dict:
    """Standardizes each feature first (mean 0, std 1) so a large-scale
    feature doesn't dominate the gradient purely from its units."""
    n = len(features)
    if n == 0:
        raise ValueError("Geen trainingsdata.")
    n_features = len(features[0])

    means = [sum(row[j] for row in features) / n for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((row[j] - means[j]) ** 2 for row in features) / n
        stds.append(math.sqrt(variance) or 1.0)  # avoid divide-by-zero for a constant feature

    normalized = [
        [(row[j] - means[j]) / stds[j] for j in range(n_features)]
        for row in features
    ]

    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for row, label in zip(normalized, labels):
            z = sum(w * x for w, x in zip(weights, row)) + bias
            error = sigmoid(z) - label
            for j in range(n_features):
                grad_w[j] += error * row[j]
            grad_b += error
        weights = [w - lr * (g / n) for w, g in zip(weights, grad_w)]
        bias -= lr * (grad_b / n)

    return {"weights": weights, "bias": bias, "means": means, "stds": stds, "features": list(feature_names)}


def score_with_model(model: dict, feature_values: list[float]) -> float:
    """Returns P(real win) in [0, 1]."""
    normalized = [
        (x - m) / s for x, m, s in zip(feature_values, model["means"], model["stds"])
    ]
    z = sum(w * x for w, x in zip(model["weights"], normalized)) + model["bias"]
    return sigmoid(z)


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load_model(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        logger.debug("Kon model niet lezen van %s.", path)
        return None
