"""
Shadow-mode win-probability scoring for sniper candidates.

User-requested: sketched out as a learned layer ON TOP of the existing
hard filters (duplicate-name, scam-social, activity-check, bundle-check,
initial-buy%) - not a replacement. Those filters encode specific,
evidence-backed scam patterns a statistical model trained on a few
hundred examples won't reliably relearn on its own.

SHADOW MODE ONLY right now: sniper.py computes and logs this score for
every real candidate that passes the existing filters, but does NOT gate
the buy decision on it. Two reasons:
1. Not enough labeled real trades yet to trust a model over the hard
   rules (a few hundred, not the 1000+ this would want).
2. Logging the score now, against the REAL outcome each trade eventually
   has, is exactly what builds the dataset needed to validate whether
   this model would actually help before ever letting it touch a real
   buy decision.

Deliberately pure Python (no numpy/sklearn dependency) - the feature
count and dataset size here don't need it, and it keeps this small enough
to read and audit directly, matching this bot's existing dependency-light
philosophy.

Feature set is intentionally thin, honestly reflecting what's actually
available at sniper's buy-decision moment (no watch window - speed is
sniper's whole edge, see sniper.py's own docs for why it can't check
holder_count/momentum before buying the way social_watch/moonshot_hunter
do):
- initial_buy_pct: creator's own buy % of supply in the creation tx
  itself - lower is safer, already computed (not logged) in
  sniper.py's _passes_filters
- liquidity_sol: kept even though largely constant across launches
  (confirmed live: 391 of ~401 real buys fell in the same 30-40 SOL
  band) - free to include, the model can learn to ignore it
- creator_win_rate: this creator wallet's historical real win rate from
  our own trade history (see wallet_reputation.py for the adjacent
  blocked-wallet tracking) - falls back to a neutral 0.5 prior for a
  creator with no history
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("pumpfun_bot.sniper_model")

MODEL_PATH = Path("data/sniper_model.json")

FEATURES = ("initial_buy_pct", "liquidity_sol", "creator_win_rate")

# a real trade must clear this real (post-fee) pct_change to count as a
# "win" label - covers the ~3.5%+ round-trip fee cost (see fees.py), not
# just "moved up at all"
WIN_MARGIN_PCT = 5.0

DEFAULT_CREATOR_WIN_RATE = 0.5


def build_creator_win_rates(activity_log_path: Path) -> dict[str, float]:
    """Real win rate per creator wallet, from every real (dry_run=false)
    sniper trade with a matched buy+exit in the activity log. Used both to
    build the creator_win_rate FEATURE for a past trade (when training)
    and to score a brand-new candidate (using history up to now)."""
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

    outcomes_by_creator: dict[str, list[bool]] = defaultdict(list)
    for mint, mint_buys in buys.items():
        exit_record = exits_by_mint.get(mint)
        if exit_record is None or exit_record.get("pct_change") is None:
            continue
        creator = (mint_buys[-1].get("meta") or {}).get("creator")
        if not creator:
            continue
        outcomes_by_creator[creator].append(exit_record["pct_change"] > WIN_MARGIN_PCT)

    return {
        creator: sum(wins) / len(wins)
        for creator, wins in outcomes_by_creator.items()
    }


def extract_features(meta: dict, creator_win_rates: dict[str, float]) -> list[float] | None:
    """None if a REQUIRED raw signal is missing - a candidate this
    incomplete isn't worth scoring (mirrors the hard filters' own
    fail-open stance elsewhere in this codebase)."""
    initial_buy_pct = meta.get("initial_buy_pct")
    liquidity_sol = meta.get("liquidity_sol")
    if initial_buy_pct is None or liquidity_sol is None:
        return None
    creator = meta.get("creator")
    creator_win_rate = creator_win_rates.get(creator, DEFAULT_CREATOR_WIN_RATE) if creator else DEFAULT_CREATOR_WIN_RATE
    return [float(initial_buy_pct), float(liquidity_sol), float(creator_win_rate)]


def _sigmoid(z: float) -> float:
    if z < -700:  # avoid math.exp overflow on a very negative z
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic_regression(
    features: list[list[float]], labels: list[int], epochs: int = 500, lr: float = 0.1,
) -> dict:
    """Plain-gradient-descent logistic regression, standardizing each
    feature first (mean 0, std 1) so a large-scale feature like
    liquidity_sol doesn't dominate the gradient purely from its units."""
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
            error = _sigmoid(z) - label
            for j in range(n_features):
                grad_w[j] += error * row[j]
            grad_b += error
        weights = [w - lr * (g / n) for w, g in zip(weights, grad_w)]
        bias -= lr * (grad_b / n)

    return {"weights": weights, "bias": bias, "means": means, "stds": stds, "features": list(FEATURES)}


def save_model(model: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        logger.debug("Kon sniper_model.json niet lezen.")
        return None


def score_with_model(model: dict, feature_values: list[float]) -> float:
    """Returns P(real win) in [0, 1]."""
    normalized = [
        (x - m) / s for x, m, s in zip(feature_values, model["means"], model["stds"])
    ]
    z = sum(w * x for w, x in zip(model["weights"], normalized)) + model["bias"]
    return _sigmoid(z)


def score(meta: dict, creator_win_rates: dict[str, float], model: dict | None = None) -> float | None:
    """None if no model is trained yet, or the candidate is missing a
    required raw feature - caller (sniper.py) treats either as "nothing to
    log", never as a reason to change the buy decision (shadow mode)."""
    model = model if model is not None else load_model()
    if model is None:
        return None
    feature_values = extract_features(meta, creator_win_rates)
    if feature_values is None:
        return None
    return score_with_model(model, feature_values)
