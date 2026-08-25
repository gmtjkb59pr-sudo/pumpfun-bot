"""
Shadow-mode win-probability scoring for social_watch candidates - a second
per-strategy model, user-requested 2026-08-24 after sniper_model.py's own
model was found to underperform its majority-class baseline (83.5% vs
87.5% accuracy on 458 real trades). Root cause traced to sniper's own
features carrying almost no real separation between winners and losers:
liquidity_sol is nearly constant across launches, and creator_win_rate
defaults to the same neutral 0.5 for ~93% of creators (most launch only
once) - see sniper_model.py's own module docstring for the full story.

social_watch tolerates real delay before buying (unlike sniper's zero-delay
design), so it already computes several signals sniper structurally can't:
holder_count, top10_concentration_pct, market_cap_usd, and (as of
2026-08-24) launch-time bundle/sniper-cluster detection (see
bundle_detection.py). Deliberately does NOT reuse sniper_model.py's
creator_win_rate feature - already shown weak there, and social_watch's
own meta doesn't log a creator field on the dry-run path anyway.

SHADOW MODE ONLY, same as sniper_model.py: computes and logs a score for
every real candidate, does NOT gate the buy decision. Two of the four
features here (top10_concentration_pct, market_cap_usd) were only STARTED
being logged 2026-08-24 (see social_watch.py's quality_meta fix) - every
row from before that fix gets the sentinel default for them, carrying zero
real information until enough post-fix trades accumulate. Honestly expect
this model to need real time before it can say anything useful - do not
treat an early "beats baseline: True" as proof it works with these two
features still mostly sentinel-valued.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from . import logistic_regression as _lr

logger = logging.getLogger("pumpfun_bot.social_watch_model")

MODEL_PATH = Path("data/social_watch_model.json")
ACTIVITY_LOG_PATH = Path("data/activity_log.jsonl")
CORRECTIONS_PATH = Path("data/real_pnl_corrections.json")
MIN_TRAINING_ROWS = 30
HOLDOUT_FRACTION = 0.2

FEATURES = ("holder_count", "top10_concentration_pct", "market_cap_usd", "launch_max_txs_in_one_slot")

# same reasoning as sniper_model.py's WIN_MARGIN_PCT - covers the ~3.5%+
# round-trip fee cost (see fees.py), not just "moved up at all". Kept as
# its own constant (not imported from sniper_model) so each model's
# threshold can be retuned independently if real evidence ever justifies it
WIN_MARGIN_PCT = 5.0

# real values are 0-100 (a percentage) - -1 is unambiguously "not logged
# for this row" (rows from before social_watch.py started logging this,
# 2026-08-24), never a genuine observation
DEFAULT_TOP10_CONCENTRATION_PCT = -1.0
# real market caps are always > 0 - same sentinel reasoning
DEFAULT_MARKET_CAP_USD = -1.0
# real slot-clustering counts are always >= 1 (the buy's own tx counts as
# at least one) - same sentinel reasoning; also covers rows from before
# bundle_detection.py existed (2026-08-24) or where the RPC lookup itself
# failed
DEFAULT_LAUNCH_MAX_TXS_IN_ONE_SLOT = -1.0


def load_corrections(path: Path | None = None) -> dict[str, dict]:
    """See sniper_model.py's identical function - same real-vs-tick
    correction data, shared across every strategy's trades."""
    path = path if path is not None else CORRECTIONS_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def extract_features(meta: dict) -> list[float] | None:
    """None if holder_count (the one field logged on every real
    social_watch buy since this strategy existed) is missing - a candidate
    this incomplete isn't worth scoring. The other three features fall back
    to their sentinel defaults instead of excluding the row, since two of
    them (top10_concentration_pct, market_cap_usd) genuinely weren't logged
    at all before 2026-08-24 - requiring them would throw away every row
    that predates that fix."""
    holder_count = meta.get("holder_count")
    if holder_count is None:
        return None
    top10_concentration_pct = meta.get("top10_concentration_pct")
    if top10_concentration_pct is None:
        top10_concentration_pct = DEFAULT_TOP10_CONCENTRATION_PCT
    market_cap_usd = meta.get("market_cap_usd")
    if market_cap_usd is None:
        market_cap_usd = DEFAULT_MARKET_CAP_USD
    launch_max_txs_in_one_slot = meta.get("launch_max_txs_in_one_slot")
    if launch_max_txs_in_one_slot is None:
        launch_max_txs_in_one_slot = DEFAULT_LAUNCH_MAX_TXS_IN_ONE_SLOT
    return [
        float(holder_count), float(top10_concentration_pct),
        float(market_cap_usd), float(launch_max_txs_in_one_slot),
    ]


def load_labeled_dataset(
    activity_log_path: Path | None = None, corrections: dict[str, dict] | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Same buy+exit matching pattern as sniper_model.py's identical
    function, filtered to strategy="social_watch" instead - see that
    module's docstring for why exits aren't ALSO filtered by strategy
    (a mint is only ever tracked by one strategy at a time, dedup'd
    elsewhere)."""
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
                and record.get("strategy") == "social_watch" and record.get("dry_run") is False
            ):
                buys[record["mint"]].append(record)
            elif record.get("type") == "exit" and record.get("dry_run") is False:
                exits_by_mint[record["mint"]] = record

    rows: list[tuple[float, list[float], int]] = []
    for mint, mint_buys in buys.items():
        exit_record = exits_by_mint.get(mint)
        if exit_record is None or exit_record.get("pct_change") is None:
            continue
        buy_record = mint_buys[-1]
        meta = buy_record.get("meta") or {}
        feature_values = extract_features(meta)
        if feature_values is None:
            continue
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
    """Same shape/contract as sniper_model.retrain_and_save - see that
    function's docstring. Returns None (saves nothing) below
    MIN_TRAINING_ROWS."""
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

    model = _lr.train_logistic_regression(train_features, train_labels, list(FEATURES))

    correct = 0
    for feature_values, label in zip(holdout_features, holdout_labels):
        predicted = 1 if _lr.score_with_model(model, feature_values) >= 0.5 else 0
        correct += predicted == label
    accuracy = correct / len(holdout_labels) if holdout_labels else 0.0
    baseline = max(sum(train_labels), len(train_labels) - sum(train_labels)) / len(train_labels)

    _lr.save_model(model, model_path if model_path is not None else MODEL_PATH)

    return {
        "n": n,
        "train_n": len(train_features),
        "holdout_n": len(holdout_features),
        "accuracy": accuracy,
        "baseline": baseline,
        "beats_baseline": accuracy > baseline,
    }


def load_model(path: Path | None = None) -> dict | None:
    return _lr.load_model(path if path is not None else MODEL_PATH)


def score(meta: dict, model: dict | None = None) -> float | None:
    """None if no model is trained yet, or the candidate is missing
    holder_count - caller (social_watch.py) treats either as "nothing to
    log", never as a reason to change the buy decision (shadow mode)."""
    model = model if model is not None else load_model()
    if model is None:
        return None
    feature_values = extract_features(meta)
    if feature_values is None:
        return None
    return _lr.score_with_model(model, feature_values)
