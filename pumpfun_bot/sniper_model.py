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
- activity_window_buy_count: real number of buys observed on the token's
  live trade stream during sniper's own pre-buy activity-check window
  (see sniper.py's _pre_buy_activity_check) - added 2026-08-24, user-
  requested, after the first trained model (on the 3 features above)
  came in BELOW the majority-class baseline (53.85% vs 57.55%). Root
  cause: liquidity_sol is nearly constant across launches and
  creator_win_rate defaults to the same neutral 0.5 for ~93% of rows
  (most creators launch only once), so the model had almost no real
  per-row variance to learn from. activity_window_buy_count is the one
  cheap, already-computed-but-previously-unlogged signal that varies
  meaningfully per candidate - a genuine "how much organic interest did
  this get" signal, not a near-constant. Falls back to
  DEFAULT_ACTIVITY_WINDOW_BUY_COUNT (a sentinel, NOT 0) for any row from
  before this field was logged, or any candidate where the activity
  check itself was disabled/skipped that run - distinguishing "not
  measured" from a genuinely-observed 0 buys.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("pumpfun_bot.sniper_model")

MODEL_PATH = Path("data/sniper_model.json")

FEATURES = ("initial_buy_pct", "liquidity_sol", "creator_win_rate", "activity_window_buy_count")

# a real trade must clear this real (post-fee) pct_change to count as a
# "win" label - covers the ~3.5%+ round-trip fee cost (see fees.py), not
# just "moved up at all"
WIN_MARGIN_PCT = 5.0

DEFAULT_CREATOR_WIN_RATE = 0.5

# sentinel, NOT 0 - a genuinely-observed 0 buys is real signal, "the check
# wasn't run/logged for this row" is a different thing entirely and must
# stay distinguishable (see module docstring)
DEFAULT_ACTIVITY_WINDOW_BUY_COUNT = -1.0


def _load_creator_outcomes(activity_log_path: Path) -> list[tuple[float, str, str, bool]]:
    """(buy_ts, mint, creator, is_win) for every real sniper trade with a
    matched buy+exit, chronologically unsorted - callers sort as needed.
    Shared by build_creator_win_rates (as-of-now, for live scoring) and
    build_point_in_time_creator_win_rates (as-of-each-row, for training -
    see that function's docstring for why these must NOT share one
    global dict)."""
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

    outcomes = []
    for mint, mint_buys in buys.items():
        exit_record = exits_by_mint.get(mint)
        if exit_record is None or exit_record.get("pct_change") is None:
            continue
        buy_record = mint_buys[-1]
        creator = (buy_record.get("meta") or {}).get("creator")
        if not creator:
            continue
        outcomes.append((buy_record["ts"], mint, creator, exit_record["pct_change"] > WIN_MARGIN_PCT))
    return outcomes


def build_creator_win_rates(activity_log_path: Path) -> dict[str, float]:
    """Real win rate per creator wallet, from EVERY real (dry_run=false)
    sniper trade with a matched buy+exit in the activity log, as of right
    now - for LIVE scoring only (sniper.py calls this right after a real
    buy, so "now" naturally can't include that trade's own not-yet-known
    outcome). Do NOT use this for building a TRAINING dataset - see
    build_point_in_time_creator_win_rates for why."""
    outcomes_by_creator: dict[str, list[bool]] = defaultdict(list)
    for _ts, _mint, creator, is_win in _load_creator_outcomes(activity_log_path):
        outcomes_by_creator[creator].append(is_win)
    return {
        creator: sum(wins) / len(wins)
        for creator, wins in outcomes_by_creator.items()
    }


def build_point_in_time_creator_win_rates(activity_log_path: Path) -> dict[str, float]:
    """Real bug found live 2026-08-23: training on build_creator_win_rates()
    (computed once from the FULL log) leaks each row's own label back into
    its creator_win_rate feature - confirmed live, 421 of 452 creators
    (93%) launched exactly ONE token, so their win rate is EXACTLY 0.0 or
    1.0, a verbatim copy of that single trade's own outcome. The model
    wasn't learning anything - it was trivially echoing the label back to
    itself for the vast majority of rows (the suspiciously clean ~0.02 vs
    ~0.98 score separation, and the inflated 86.54% holdout accuracy,
    were both artifacts of this leak, not real signal).

    Returns, per MINT, that mint's creator's win rate computed ONLY from
    trades that happened STRICTLY BEFORE it - exactly what a live buy
    decision would have had access to at that moment (matching
    build_creator_win_rates' live-scoring semantics, just replayed
    historically instead of frozen at "now"). Keyed by mint, not creator -
    the same creator can appear at multiple points in training, each
    occurrence needs its OWN point-in-time rate, not one shared value."""
    outcomes = sorted(_load_creator_outcomes(activity_log_path))
    history: dict[str, list[bool]] = defaultdict(list)
    rate_by_mint: dict[str, float] = {}
    for _ts, mint, creator, is_win in outcomes:
        prior = history[creator]
        rate_by_mint[mint] = (sum(prior) / len(prior)) if prior else DEFAULT_CREATOR_WIN_RATE
        history[creator].append(is_win)
    return rate_by_mint


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
    activity_window_buy_count = meta.get("activity_window_buy_count")
    if activity_window_buy_count is None:
        activity_window_buy_count = DEFAULT_ACTIVITY_WINDOW_BUY_COUNT
    return [
        float(initial_buy_pct), float(liquidity_sol), float(creator_win_rate),
        float(activity_window_buy_count),
    ]


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


def save_model(model: dict, path: Path | None = None) -> None:
    # resolved inside the body, not as a default-arg value - a default arg
    # is bound ONCE at function-definition time, so patching the module-
    # level MODEL_PATH afterward (as tests do) would silently have no
    # effect on it otherwise
    path = path if path is not None else MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))


def load_model(path: Path | None = None) -> dict | None:
    path = path if path is not None else MODEL_PATH
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
