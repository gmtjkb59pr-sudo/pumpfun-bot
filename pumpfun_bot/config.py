"""Configuratie laden uit config.yaml + .env, met sane defaults en validatie."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RiskConfig:
    dry_run: bool = True
    max_sol_per_trade: float = 0.05
    max_sol_total_exposure: float = 0.3
    max_trades_per_hour: int = 10
    max_daily_loss_sol: float = 0.2
    default_slippage_pct: float = 10
    min_liquidity_sol: float = 5
    max_open_positions: int = 1000  # effectively unlimited unless set lower
    # hard kill-switch, not a per-trade limit: once the wallet's real SOL
    # balance (converted to USD via a live price) drops below this, the
    # whole bot stops - user-requested "stop before it gets to zero" floor.
    # 0 = disabled (default) so existing configs without this field never
    # unexpectedly enable a shutdown.
    min_wallet_balance_usd: float = 0
    # second hard kill-switch, distinct from the floor above: stops the bot
    # once the REAL (ground-truth wallet, not the fee-model estimate)
    # session loss reaches this many dollars - user-requested "stop after
    # losing $X", not "stop once the wallet is down to $Y". 0 = disabled.
    max_real_loss_usd: float = 0
    # user-requested, real incident 2026-08-23: max_sol_total_exposure is a
    # FIXED number - re-enabling sniper with a fixed cap larger than the
    # actual (much smaller, after a day of losses) wallet balance let it
    # spend the wallet down from 0.114 to 0.0012 SOL in about 90 seconds,
    # since nothing scaled the cap to what was actually left. This caps
    # total exposure at a PERCENTAGE of the wallet's real, live SOL balance
    # instead (see risk.py's update_live_balance()/can_trade()) - shrinks
    # automatically as the wallet shrinks, grows automatically after a
    # top-up, without needing a manual config edit either way. 0 = disabled
    # (falls back to max_sol_total_exposure alone, the old fixed-cap-only
    # behavior) so existing configs without this field are unaffected.
    max_exposure_pct_of_balance: float = 0


@dataclass
class TakeProfitLevel:
    """One rung of a scaled take-profit ladder: at `multiplier`x the entry
    price, sell `sell_pct`% of whatever's LEFT of the position at that
    moment (not of the original buy) - user-requested, ported from an
    earlier version of this bot's sniper strategy after
    scaled_exit_simulator.py's parallel counterfactual observer had been
    quietly modeling a similar idea (take partial profit, trail the rest)
    without ever being the REAL exit logic. See OutcomeTracker.track()'s
    take_profit_ladder docstring for how this replaces the single-shot
    take_profit_pct when non-empty."""
    multiplier: float
    sell_pct: float


@dataclass
class SniperConfig:
    enabled: bool = False
    min_token_age_seconds: int = 0
    max_token_age_seconds: int = 15
    require_socials: bool = False
    take_profit_pct: float = 50
    stop_loss_pct: float = 25
    trailing_activation_pct: float = 20
    trailing_stop_pct: float = 15
    # user-requested: scaled take-profit ladder - empty (default) keeps the
    # existing single-shot take_profit_pct behavior unchanged. See
    # TakeProfitLevel's docstring.
    take_profit_ladder: list = field(default_factory=list)

    # user-requested, ported from an earlier version of this bot's sniper -
    # scam/insider entry filters, all off (0/False) by default so enabling
    # sniper doesn't silently change its buy behavior.
    #
    # Free (no extra RPC call): rejects a launch where the creator's own
    # initialBuy already claims more than this % of total supply in the
    # SAME transaction as the token's creation - often a sign of a planned
    # dump. 0 = disabled.
    max_initial_buy_pct: float = 0
    # Costs one extra RPC round-trip (getTokenSupply + getTokenLargestAccounts)
    # per candidate that reaches this check. CAUTION, unlike the same field on
    # social_watch/birdeye_movers/coingecko_movers: those strategies only ever
    # evaluate this after a watch window or on an already-established token,
    # giving holder_concentration.py's SETTLING_DELAY_SEC time to pass first.
    # Sniper buys within seconds of launch with no such delay - essentially
    # all supply still sits with the bonding curve/deployer at that point
    # (confirmed live elsewhere in this codebase), so this will read close to
    # 100% for nearly every candidate unless you're prepared for that. 0 =
    # disabled.
    max_top10_concentration_pct: float = 0
    # Watches the token's live trade stream for bundle_check_window_ms right
    # after passing the filters above, and rejects it if more than
    # bundle_check_max_buys buys land in that window - a sign of coordinated
    # insider wallets all buying at once. Deliberately costs real time (the
    # whole window), trading sniper's speed advantage for this one extra
    # safety signal - off by default for that reason.
    enable_bundle_check: bool = False
    bundle_check_window_ms: int = 300
    bundle_check_max_buys: int = 5
    # user-requested, real finding 2026-08-23: 57.1% of everything sniper
    # bought went stale_price (zero real trade activity) within seconds -
    # min_liquidity_sol can't catch this, since every pump.fun launch
    # starts at essentially the same bonding-curve liquidity (confirmed
    # live: 391 of ~401 real buys all fell in the exact same 30-40 SOL
    # band, functionally a constant, not a signal). Reuses the SAME
    # bundle_check_window_ms trade-stream watch (see
    # _bundle_check_flags_bundle/_pre_buy_activity_check) to also reject a
    # candidate with FEWER than this many real buys in that window - the
    # opposite failure mode from bundle_check_max_buys, checked in the same
    # pass so this doesn't cost a second window's worth of real time. 0 =
    # disabled (default). Only takes effect once the window is actually
    # being watched (enable_bundle_check=true or this > 0).
    min_buys_in_window: int = 0


@dataclass
class SocialWatchConfig:
    """Unlike SniperConfig, which decides instantly off the launch event,
    this holds off and watches a candidate for watch_window_sec, polling its
    off-chain metadata every poll_interval_sec, and only buys if socials show
    up within that window - trading speed for a real, verifiable quality
    signal, since the launch event itself never carries socials directly."""
    enabled: bool = False
    watch_window_sec: int = 60
    poll_interval_sec: int = 10
    take_profit_pct: float = 50
    stop_loss_pct: float = 25
    trailing_activation_pct: float = 20
    trailing_stop_pct: float = 15
    # user-requested: scaled take-profit ladder - empty (default) keeps the
    # existing single-shot take_profit_pct behavior unchanged. See
    # TakeProfitLevel's docstring.
    take_profit_ladder: list = field(default_factory=list)
    # 0 = no filter. Only auto_tuner.py raises this (tighten-only, gated on
    # real sample size + margin) once there's evidence for a real threshold -
    # never hand-set this without evidence, see holder_count_tuning.py
    min_holder_count: int = 0
    # user-requested: skip candidates below this USD market cap - thin/tiny
    # market caps correlated with the worst stop-loss overshoots tonight
    # (a single sell can crater an illiquid curve 30-50% in one trade tick).
    # 0 = no filter.
    min_market_cap_usd: float = 0
    # user-requested: skip candidates whose top 10 holder accounts control
    # more than this % of supply - a manufactured/bundled launch concentrates
    # the float in a handful of wallets that can dump together, same failure
    # mode behind the deep stop-loss overshoots. Threshold matches Luminos's
    # (luminos.capital) own published "flag" band for this signal - see
    # holder_concentration.py for why we compute it ourselves instead of
    # querying them. 0 = no filter.
    max_top10_concentration_pct: float = 0
    # user-requested "movers"-style filter: only buy candidates already
    # showing positive 5m price momentum on DexScreener (see dexscreener.py
    # for why that's the ToS-clean data source, not pump.fun's own site).
    # Window shortened from the original 1h to 5m - user-requested, matches
    # how young these candidates actually are. False = no filter.
    require_positive_momentum_5m: bool = False
    # user-requested, evidence-based (real dry-run outcomes): losing trades
    # averaged 216% 5m momentum at buy vs 134% for winners, and win rate
    # drops sharply above ~100% (80% win at 0-50% momentum, ~25-31% above
    # 150%) - buying a token already up 100%+ correlates with buying near a
    # local top. 0 = no ceiling.
    max_price_change_5m_pct: float = 0
    # user-requested: own position budget, separate from the global
    # risk.max_open_positions shared pool - a burst of buys from another
    # strategy (e.g. birdeye_movers firing all at once on its slow poll)
    # can no longer crowd this strategy out entirely. Only enforced in
    # live mode - dry_run skips the open-positions check altogether so
    # every qualifying candidate can open a position for outcome data.
    max_open_positions: int = 5
    # user-requested: this strategy's own trade-size cap, independent of the
    # shared risk.max_sol_per_trade - 0 (default) falls back to that shared
    # value, unchanged behavior for anyone who hasn't set this explicitly
    trade_size_sol: float = 0.0


@dataclass
class BirdeyeMoversConfig:
    """Discovers already-existing tokens showing a real volume/price spike
    via Birdeye's trending API - the one gap social_watch structurally
    cannot cover, since it only ever sees BRAND NEW launches via
    PumpPortal's subscribeNewToken (see birdeye.py/birdeye_movers.py)."""
    enabled: bool = False
    api_key: str = ""
    # user-requested: Birdeye's free tier is ~1,000 calls/month (30,000 CU
    # at 30 CU/call) - this cadence keeps well within that even running
    # 24/7. Do not lower without checking the Birdeye pricing page first.
    poll_interval_sec: int = 2700
    trending_limit: int = 20
    # 0 = no filter, same convention as social_watch's equivalents
    min_holder_count: int = 0
    max_top10_concentration_pct: float = 0
    # SUPERSEDED, see is_pump_fun_mint() in birdeye_movers.py and
    # build_and_send_trade's pool="auto" default in pumpportal_client.py.
    # Originally lowered to 100,000 after two real candidates failed to buy
    # with a 400 - traced at the time to "migrated past the ~$69k bonding-
    # curve graduation point, pool='pump' has no route for it anymore".
    # Later disproven for genuine pump.fun tokens: confirmed live that a
    # real pump.fun-suffixed mint at ~$260M market cap builds a valid
    # transaction fine via pool="auto" (fails only under the old hardcoded
    # pool="pump"). The ACTUAL cause of those two original failures was
    # that neither mint was a real pump.fun token at all (Birdeye's
    # trending list is Solana-wide) - now caught by is_pump_fun_mint()
    # before this check ever runs. A blue-chip "category error" (SOL, PUMP,
    # wrapped ETH) is also already excluded there, since none of those end
    # in "pump". With both original justifications gone, market cap no
    # longer predicts buyability - 0 (disabled) by default. Set a positive
    # value here only if you want to deliberately avoid chasing tokens that
    # already pumped a lot (a "how far did it already run" filter, not a
    # "can we even buy this" one - those are different questions).
    max_market_cap_usd: float = 0
    take_profit_pct: float = 50
    stop_loss_pct: float = 25
    trailing_activation_pct: float = 20
    trailing_stop_pct: float = 15
    # user-requested: scaled take-profit ladder - empty (default) keeps the
    # existing single-shot take_profit_pct behavior unchanged. See
    # TakeProfitLevel's docstring.
    take_profit_ladder: list = field(default_factory=list)
    # user-requested: own position budget, separate from the global
    # risk.max_open_positions shared pool - confirmed live that a single
    # trending-list poll can return several qualifying tokens at once and
    # instantly claim the whole shared pool, crowding social_watch out.
    # Only enforced in live mode - dry_run skips the open-positions check
    # altogether so every qualifying candidate can open a position for
    # outcome data.
    max_open_positions: int = 5
    # user-requested: this strategy's own trade-size cap, independent of the
    # shared risk.max_sol_per_trade - 0 (default) falls back to that shared
    # value, unchanged behavior for anyone who hasn't set this explicitly
    trade_size_sol: float = 0.0


@dataclass
class CoinGeckoMoversConfig:
    """Same discovery niche as BirdeyeMoversConfig (already-existing tokens
    spiking, not brand-new launches) via CoinGecko's free Demo API plan
    instead - much higher budget (10,000 calls/month vs Birdeye's ~1,000),
    so this can poll far more often and react to genuinely short-window
    momentum (m5/m15/m30) instead of only a 24h lagging signal. See
    coingecko.py's module docstring for the API/budget details, and
    coingecko_movers.py for why the same is_pump_fun_mint() filter from
    birdeye_movers.py applies here too - CoinGecko's trending pools are
    Solana-wide, not pump.fun-specific, same as Birdeye's."""
    enabled: bool = False
    api_key: str = ""
    # user-requested: CoinGecko's free Demo plan is 100 calls/min, 10,000
    # calls/month - 300s (5 min) is 8,640 calls/month running 24/7,
    # comfortably under that with margin. Do not lower without checking
    # the actual monthly usage on the CoinGecko developer dashboard first.
    poll_interval_sec: int = 300
    trending_limit: int = 20
    # which of CoinGecko's price_change_percentage windows (m5/m15/m30/h1/
    # h6/h24) gates the buy decision - defaults to the shortest available,
    # matching social_watch's own evidence-based use of a 5-minute window
    momentum_window: str = "m5"
    min_holder_count: int = 0
    max_top10_concentration_pct: float = 0
    # SUPERSEDED - see BirdeyeMoversConfig.max_market_cap_usd's docstring
    # for the full reasoning. 0 (disabled) by default.
    max_market_cap_usd: float = 0
    take_profit_pct: float = 50
    stop_loss_pct: float = 25
    trailing_activation_pct: float = 20
    trailing_stop_pct: float = 15
    # user-requested: scaled take-profit ladder - empty (default) keeps the
    # existing single-shot take_profit_pct behavior unchanged. See
    # TakeProfitLevel's docstring.
    take_profit_ladder: list = field(default_factory=list)
    max_open_positions: int = 5
    trade_size_sol: float = 0.0


@dataclass
class MoonshotHunterConfig:
    """User-requested: a deliberately different, low-frequency bet - most
    strategies in this bot cut losses fast and lock in profit early
    (tight stop_loss/take_profit, 15-min max hold). This one instead aims
    at the rare 100-1000x pump.fun outlier, discovering candidates the
    SAME way social_watch does (watch new launches, wait for socials
    within watch_window_sec) but with much stricter/inverted filters and a
    completely different exit shape - wide stop-loss, a ladder that only
    starts taking (small) profit at huge multiples so most of the position
    keeps riding, and a hold time measured in days/weeks instead of
    minutes (see OutcomeTracker.track()'s max_hold_sec/
    stale_price_timeout_sec docstring for why the shared 15-min/10s
    defaults would kill this strategy at the first natural trading lull).

    Explicitly NOT expected to have a proven edge like social_watch's
    evidence-based filters - there's no reliable early signal for a true
    viral 1000x, this is a small, deliberately isolated lottery-ticket
    allocation, not a strategy to size like the others."""
    enabled: bool = False
    watch_window_sec: int = 60
    poll_interval_sec: int = 3
    # much stricter than social_watch's own bar - a real viral candidate
    # should be pulling in holders unusually fast, not just clearing the
    # minimum quality bar
    min_holder_count: int = 300
    max_top10_concentration_pct: float = 70
    # INVERTED from every other strategy's max_price_change_5m_pct ceiling
    # (which avoids buying something that already pumped) - here we
    # instead REQUIRE it, as the closest available proxy for "this is
    # already going viral right now". 0 = no floor (not recommended).
    min_price_change_5m_pct: float = 100
    stop_loss_pct: float = 70
    trailing_activation_pct: float = 300
    trailing_stop_pct: float = 35
    # small early rungs so most of the position keeps riding toward the
    # actual moonshot target - see TakeProfitLevel's docstring in config.py
    take_profit_ladder: list = field(default_factory=lambda: [
        TakeProfitLevel(multiplier=10, sell_pct=10),
        TakeProfitLevel(multiplier=50, sell_pct=15),
        TakeProfitLevel(multiplier=200, sell_pct=20),
    ])
    # 30 days - see OutcomeTracker.track()'s docstring for why the shared
    # 15-min MAX_HOLD_SEC constant doesn't apply here
    max_hold_sec: float = 2_592_000
    # 6 hours - much longer than the shared 10s STALE_PRICE_TIMEOUT_SEC,
    # but still a real backstop: a token with genuinely zero trades for 6
    # straight hours is very unlikely to still moonshot
    stale_price_timeout_sec: float = 21_600
    # "coin of the month" framing - one bet at a time, not a volume
    # strategy. Only enforced live - dry_run skips the open-positions
    # check like every other strategy here, to farm outcome data.
    max_open_positions: int = 1
    # user-requested: don't let this fire back-to-back bets on the same
    # bad day - a real cooldown between attempts, not just max_open_positions
    min_seconds_between_buys: float = 21_600
    # deliberately small, fixed, and separate from normal trade sizing -
    # this is capital you're fully prepared to lose entirely
    trade_size_sol: float = 0.02


@dataclass
class CopyTradeConfig:
    enabled: bool = False
    watched_wallets: list = field(default_factory=list)
    mirror_pct: float = 100
    max_copy_delay_ms: int = 3000
    # user-requested: copytrade's own trade-size cap, independent of the
    # shared risk.max_sol_per_trade - 0 (default) falls back to that shared
    # value, unchanged behavior for anyone who hasn't set this explicitly
    max_trade_sol: float = 0.0


@dataclass
class MarketMakerConfig:
    enabled: bool = False
    target_tokens: list = field(default_factory=list)
    grid_levels: int = 5
    grid_spacing_pct: float = 2
    order_size_sol: float = 0.01


@dataclass
class AppConfig:
    private_key: str
    rpc_http_url: str
    rpc_ws_url: str
    pumpportal_ws_url: str
    pumpportal_trade_api_url: str
    pumpportal_api_key: str
    risk: RiskConfig
    sniper: SniperConfig
    social_watch: SocialWatchConfig
    birdeye_movers: BirdeyeMoversConfig
    coingecko_movers: CoinGeckoMoversConfig
    moonshot_hunter: MoonshotHunterConfig
    copytrade: CopyTradeConfig
    market_maker: MarketMakerConfig
    alerts_console: bool
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_enabled: bool = True
    dashboard_port: int = 8765
    # user-requested: OKX Onchain OS credentials for the smart-money wallet
    # lister (see smart_money_wallets.py / list_smart_money_wallets.py) -
    # signed-request auth, needs all three. Empty by default so existing
    # configs without OKX access never break.
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""


def _parse_take_profit_ladder(raw_levels: list | None) -> list[TakeProfitLevel]:
    """Parses a strategy's own take_profit_ladder YAML list into
    TakeProfitLevel instances, sorted ascending by multiplier - the order
    OutcomeTracker's ladder logic assumes when checking which levels a
    price update has newly crossed. Missing/malformed - never guesses a
    default multiplier/sell_pct, a broken ladder entry should fail loudly
    (KeyError) rather than silently trade with a wrong threshold."""
    if not raw_levels:
        return []
    levels = [
        TakeProfitLevel(multiplier=lvl["multiplier"], sell_pct=lvl["sell_pct"])
        for lvl in raw_levels
    ]
    return sorted(levels, key=lambda lvl: lvl.multiplier)


def load_config(path: str = "config.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(
            f"Config bestand '{path}' niet gevonden. Kopieer config.example.yaml naar "
            f"config.yaml en vul je gegevens in."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    key_env_var = raw["wallet"]["private_key_env_var"]
    private_key = os.environ.get(key_env_var, "")
    if not private_key:
        sys.exit(
            f"Environment variable '{key_env_var}' is leeg. Zet je private key in .env "
            f"(zie .env.example). Deel dit bestand met NIEMAND."
        )

    risk_raw = raw.get("risk", {})
    risk = RiskConfig(
        dry_run=risk_raw.get("dry_run", True),
        max_sol_per_trade=risk_raw.get("max_sol_per_trade", 0.05),
        max_sol_total_exposure=risk_raw.get("max_sol_total_exposure", 0.3),
        max_trades_per_hour=risk_raw.get("max_trades_per_hour", 10),
        max_daily_loss_sol=risk_raw.get("max_daily_loss_sol", 0.2),
        default_slippage_pct=risk_raw.get("default_slippage_pct", 10),
        min_liquidity_sol=risk_raw.get("min_liquidity_sol", 5),
        max_open_positions=risk_raw.get("max_open_positions", 1000),
        min_wallet_balance_usd=risk_raw.get("min_wallet_balance_usd", 0),
        max_real_loss_usd=risk_raw.get("max_real_loss_usd", 0),
        max_exposure_pct_of_balance=risk_raw.get("max_exposure_pct_of_balance", 0),
    )

    strat_raw = raw.get("strategies", {})
    sniper_raw = strat_raw.get("sniper", {})
    sniper = SniperConfig(
        enabled=sniper_raw.get("enabled", False),
        min_token_age_seconds=sniper_raw.get("min_token_age_seconds", 0),
        max_token_age_seconds=sniper_raw.get("max_token_age_seconds", 15),
        require_socials=sniper_raw.get("require_socials", False),
        take_profit_pct=sniper_raw.get("take_profit_pct", 50),
        stop_loss_pct=sniper_raw.get("stop_loss_pct", 25),
        trailing_activation_pct=sniper_raw.get("trailing_activation_pct", 20),
        trailing_stop_pct=sniper_raw.get("trailing_stop_pct", 15),
        take_profit_ladder=_parse_take_profit_ladder(sniper_raw.get("take_profit_ladder")),
        max_initial_buy_pct=sniper_raw.get("max_initial_buy_pct", 0),
        max_top10_concentration_pct=sniper_raw.get("max_top10_concentration_pct", 0),
        enable_bundle_check=sniper_raw.get("enable_bundle_check", False),
        bundle_check_window_ms=sniper_raw.get("bundle_check_window_ms", 300),
        bundle_check_max_buys=sniper_raw.get("bundle_check_max_buys", 5),
        min_buys_in_window=sniper_raw.get("min_buys_in_window", 0),
    )

    sw_raw = strat_raw.get("social_watch", {})
    social_watch = SocialWatchConfig(
        enabled=sw_raw.get("enabled", False),
        watch_window_sec=sw_raw.get("watch_window_sec", 60),
        poll_interval_sec=sw_raw.get("poll_interval_sec", 10),
        take_profit_pct=sw_raw.get("take_profit_pct", 50),
        stop_loss_pct=sw_raw.get("stop_loss_pct", 25),
        trailing_activation_pct=sw_raw.get("trailing_activation_pct", 20),
        trailing_stop_pct=sw_raw.get("trailing_stop_pct", 15),
        min_holder_count=sw_raw.get("min_holder_count", 0),
        min_market_cap_usd=sw_raw.get("min_market_cap_usd", 0),
        max_top10_concentration_pct=sw_raw.get("max_top10_concentration_pct", 0),
        require_positive_momentum_5m=sw_raw.get("require_positive_momentum_5m", False),
        max_price_change_5m_pct=sw_raw.get("max_price_change_5m_pct", 0),
        max_open_positions=sw_raw.get("max_open_positions", 5),
        trade_size_sol=sw_raw.get("trade_size_sol", 0.0),
        take_profit_ladder=_parse_take_profit_ladder(sw_raw.get("take_profit_ladder")),
    )

    be_raw = strat_raw.get("birdeye_movers", {})
    be_key_var = be_raw.get("api_key_env_var", "BIRDEYE_API_KEY")
    birdeye_movers = BirdeyeMoversConfig(
        enabled=be_raw.get("enabled", False),
        api_key=os.environ.get(be_key_var, ""),
        poll_interval_sec=be_raw.get("poll_interval_sec", 2700),
        trending_limit=be_raw.get("trending_limit", 20),
        min_holder_count=be_raw.get("min_holder_count", 0),
        max_top10_concentration_pct=be_raw.get("max_top10_concentration_pct", 0),
        max_market_cap_usd=be_raw.get("max_market_cap_usd", 0),
        max_open_positions=be_raw.get("max_open_positions", 5),
        trade_size_sol=be_raw.get("trade_size_sol", 0.0),
        take_profit_pct=be_raw.get("take_profit_pct", 50),
        stop_loss_pct=be_raw.get("stop_loss_pct", 25),
        trailing_activation_pct=be_raw.get("trailing_activation_pct", 20),
        trailing_stop_pct=be_raw.get("trailing_stop_pct", 15),
        take_profit_ladder=_parse_take_profit_ladder(be_raw.get("take_profit_ladder")),
    )

    cg_raw = strat_raw.get("coingecko_movers", {})
    cg_key_var = cg_raw.get("api_key_env_var", "COINGECKO_API_KEY")
    coingecko_movers = CoinGeckoMoversConfig(
        enabled=cg_raw.get("enabled", False),
        api_key=os.environ.get(cg_key_var, ""),
        poll_interval_sec=cg_raw.get("poll_interval_sec", 300),
        trending_limit=cg_raw.get("trending_limit", 20),
        momentum_window=cg_raw.get("momentum_window", "m5"),
        min_holder_count=cg_raw.get("min_holder_count", 0),
        max_top10_concentration_pct=cg_raw.get("max_top10_concentration_pct", 0),
        max_market_cap_usd=cg_raw.get("max_market_cap_usd", 0),
        max_open_positions=cg_raw.get("max_open_positions", 5),
        trade_size_sol=cg_raw.get("trade_size_sol", 0.0),
        take_profit_pct=cg_raw.get("take_profit_pct", 50),
        stop_loss_pct=cg_raw.get("stop_loss_pct", 25),
        trailing_activation_pct=cg_raw.get("trailing_activation_pct", 20),
        trailing_stop_pct=cg_raw.get("trailing_stop_pct", 15),
        take_profit_ladder=_parse_take_profit_ladder(cg_raw.get("take_profit_ladder")),
    )

    mh_raw = strat_raw.get("moonshot_hunter", {})
    mh_defaults = MoonshotHunterConfig()
    moonshot_hunter = MoonshotHunterConfig(
        enabled=mh_raw.get("enabled", False),
        watch_window_sec=mh_raw.get("watch_window_sec", mh_defaults.watch_window_sec),
        poll_interval_sec=mh_raw.get("poll_interval_sec", mh_defaults.poll_interval_sec),
        min_holder_count=mh_raw.get("min_holder_count", mh_defaults.min_holder_count),
        max_top10_concentration_pct=mh_raw.get(
            "max_top10_concentration_pct", mh_defaults.max_top10_concentration_pct,
        ),
        min_price_change_5m_pct=mh_raw.get(
            "min_price_change_5m_pct", mh_defaults.min_price_change_5m_pct,
        ),
        stop_loss_pct=mh_raw.get("stop_loss_pct", mh_defaults.stop_loss_pct),
        trailing_activation_pct=mh_raw.get(
            "trailing_activation_pct", mh_defaults.trailing_activation_pct,
        ),
        trailing_stop_pct=mh_raw.get("trailing_stop_pct", mh_defaults.trailing_stop_pct),
        # unlike every other strategy's empty-ladder-means-no-ladder
        # convention, the ladder IS this strategy's whole exit design (see
        # MoonshotHunterConfig's docstring - there's no take_profit_pct
        # fallback field at all) - an omitted YAML value falls back to the
        # dataclass's own built-in ladder, not an empty one
        take_profit_ladder=(
            _parse_take_profit_ladder(mh_raw.get("take_profit_ladder"))
            or mh_defaults.take_profit_ladder
        ),
        max_hold_sec=mh_raw.get("max_hold_sec", mh_defaults.max_hold_sec),
        stale_price_timeout_sec=mh_raw.get(
            "stale_price_timeout_sec", mh_defaults.stale_price_timeout_sec,
        ),
        max_open_positions=mh_raw.get("max_open_positions", mh_defaults.max_open_positions),
        min_seconds_between_buys=mh_raw.get(
            "min_seconds_between_buys", mh_defaults.min_seconds_between_buys,
        ),
        trade_size_sol=mh_raw.get("trade_size_sol", mh_defaults.trade_size_sol),
    )

    ct_raw = strat_raw.get("copytrade", {})
    copytrade = CopyTradeConfig(
        enabled=ct_raw.get("enabled", False),
        watched_wallets=ct_raw.get("watched_wallets", []) or [],
        mirror_pct=ct_raw.get("mirror_pct", 100),
        max_copy_delay_ms=ct_raw.get("max_copy_delay_ms", 3000),
        max_trade_sol=ct_raw.get("max_trade_sol", 0.0),
    )

    mm_raw = strat_raw.get("market_maker", {})
    market_maker = MarketMakerConfig(
        enabled=mm_raw.get("enabled", False),
        target_tokens=mm_raw.get("target_tokens", []) or [],
        grid_levels=mm_raw.get("grid_levels", 5),
        grid_spacing_pct=mm_raw.get("grid_spacing_pct", 2),
        order_size_sol=mm_raw.get("order_size_sol", 0.01),
    )

    alerts_raw = raw.get("alerts", {})
    tg_raw = alerts_raw.get("telegram", {})
    tg_token_var = tg_raw.get("bot_token_env_var", "TELEGRAM_BOT_TOKEN")

    pp_raw = raw.get("pumpportal", {})
    pp_key_var = pp_raw.get("api_key_env_var", "PUMPPORTAL_API_KEY")

    dash_raw = raw.get("dashboard", {})

    return AppConfig(
        private_key=private_key,
        rpc_http_url=raw["rpc"]["http_url"],
        rpc_ws_url=raw["rpc"]["ws_url"],
        pumpportal_ws_url=pp_raw.get("ws_url", "wss://pumpportal.fun/api/data"),
        pumpportal_trade_api_url=pp_raw.get(
            "trade_api_url", "https://pumpportal.fun/api/trade-local"
        ),
        pumpportal_api_key=os.environ.get(pp_key_var, ""),
        risk=risk,
        sniper=sniper,
        social_watch=social_watch,
        birdeye_movers=birdeye_movers,
        coingecko_movers=coingecko_movers,
        moonshot_hunter=moonshot_hunter,
        copytrade=copytrade,
        market_maker=market_maker,
        alerts_console=alerts_raw.get("console", True),
        telegram_enabled=tg_raw.get("enabled", False),
        telegram_bot_token=os.environ.get(tg_token_var, ""),
        telegram_chat_id=tg_raw.get("chat_id", ""),
        dashboard_enabled=dash_raw.get("enabled", True),
        dashboard_port=dash_raw.get("port", 8765),
        okx_api_key=os.environ.get("OKX_API_KEY", ""),
        okx_secret_key=os.environ.get("OKX_SECRET_KEY", ""),
        okx_passphrase=os.environ.get("OKX_PASSPHRASE", ""),
    )
