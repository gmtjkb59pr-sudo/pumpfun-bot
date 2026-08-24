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
ACTIVITY_LOG_PATH = Path("data/activity_log.jsonl")
CORRECTIONS_PATH = Path("data/real_pnl_corrections.json")
MIN_TRAINING_ROWS = 30
HOLDOUT_FRACTION = 0.2

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


def load_corrections(path: Path | None = None) -> dict[str, dict]:
    """See scripts/audit_real_pnl.py's module docstring for the full bug
    story (user-reported 2026-08-24: "catecoin this is a false log",
    "wojakius also not correct") - keyed by sell tx_signature, each entry
    carries the TRUE on-chain pct_change for that exit, re-derived from
    the wallet's real SOL delta instead of the stale price-tick estimate
    _fetch_real_sol_delta silently fell back to before that fix. Missing
    entirely (file not yet generated) or missing a specific signature
    (audit script hasn't resolved it yet) both just mean "use the
    originally-logged pct_change" - never an error. Only matters for the
    HISTORICAL backlog from before that fix landed - a new real exit's own
    pct_change is already correct at the source now, nothing to correct."""
    path = path if path is not None else CORRECTIONS_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def load_labeled_dataset(
    activity_log_path: Path | None = None, corrections: dict[str, dict] | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Real bug found live 2026-08-23: using build_creator_win_rates()
    (one global, full-log snapshot) here leaked each row's own label back
    into its own creator_win_rate feature for any single-launch creator
    (421 of 452, 93%, in this dataset) - see
    build_point_in_time_creator_win_rates' docstring for the full story.
    Uses the point-in-time version instead, and sorts by buy_ts so the
    later train/holdout split is a genuine chronological forward-test,
    not a random shuffle that could still leak across nearby-in-time rows."""
    activity_log_path = activity_log_path if activity_log_path is not None else ACTIVITY_LOG_PATH
    corrections = corrections if corrections is not None else {}
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

    point_in_time_rates = build_point_in_time_creator_win_rates(activity_log_path)

    rows: list[tuple[float, list[float], int]] = []
    for mint, mint_buys in buys.items():
        exit_record = exits_by_mint.get(mint)
        if exit_record is None or exit_record.get("pct_change") is None:
            continue
        buy_record = mint_buys[-1]
        meta = buy_record.get("meta") or {}
        initial_buy_pct = meta.get("initial_buy_pct")
        liquidity_sol = meta.get("liquidity_sol")
        if initial_buy_pct is None or liquidity_sol is None:
            continue
        creator_win_rate = point_in_time_rates.get(mint, DEFAULT_CREATOR_WIN_RATE)
        activity_window_buy_count = meta.get("activity_window_buy_count")
        if activity_window_buy_count is None:
            activity_window_buy_count = DEFAULT_ACTIVITY_WINDOW_BUY_COUNT
        feature_values = [
            float(initial_buy_pct), float(liquidity_sol), float(creator_win_rate),
            float(activity_window_buy_count),
        ]
        correction = corrections.get(exit_record.get("tx_signature", ""))
        pct_change = correction["corrected_pct_change"] if correction is not None else exit_record["pct_change"]
        label = 1 if pct_change > WIN_MARGIN_PCT else 0
        rows.append((buy_record["ts"], feature_values, label))

    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows], [r[2] for r in rows]


def retrain_and_save(
    activity_log_path: Path | None = None,
    corrections_path: Path | None = None,
    model_path: Path | None = None,
) -> dict | None:
    """Trains a fresh model from the current real trade history and saves
    it, for BOTH scripts/train_sniper_model.py's manual CLI use and
    model_retrain.py's automatic periodic background use (user-requested
    2026-08-24: "build an AI in the bot so it learns more automatically").

    Returns None (and saves nothing) below MIN_TRAINING_ROWS - a model
    "trained" on a handful of examples is worse than no model at all. On
    success, returns a result dict: {"n", "train_n", "holdout_n",
    "accuracy", "baseline", "beats_baseline"} - callers decide what to log
    or alert, this function only trains/evaluates/saves."""
    activity_log_path = activity_log_path if activity_log_path is not None else ACTIVITY_LOG_PATH
    if not activity_log_path.exists():
        return None

    corrections = load_corrections(corrections_path)
    features, labels = load_labeled_dataset(activity_log_path, corrections)
    n = len(features)
    if n < MIN_TRAINING_ROWS:
        return None

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

    save_model(model, model_path)

    return {
        "n": n,
        "train_n": len(train_features),
        "holdout_n": len(holdout_features),
        "accuracy": accuracy,
        "baseline": baseline,
        "beats_baseline": accuracy > baseline,
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
