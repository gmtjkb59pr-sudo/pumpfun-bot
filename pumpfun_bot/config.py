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
    # confirmed live: Birdeye's trending list (sorted by volumeUSD) surfaces
    # major/blue-chip tokens like SOL, wrapped ETH, and PUMP itself, not
    # just memecoins - "buying" SOL through a pump.fun-style trade is a
    # category error (SOL is the base currency, not a bonding-curve token),
    # so unlike the 0-disables-it filters above, this one defaults ON at a
    # level clearly above real memecoin market caps and clearly below any
    # major token's.
    max_market_cap_usd: float = 20_000_000
    take_profit_pct: float = 50
    stop_loss_pct: float = 25
    trailing_activation_pct: float = 20
    trailing_stop_pct: float = 15


@dataclass
class CopyTradeConfig:
    enabled: bool = False
    watched_wallets: list = field(default_factory=list)
    mirror_pct: float = 100
    max_copy_delay_ms: int = 3000


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
    copytrade: CopyTradeConfig
    market_maker: MarketMakerConfig
    alerts_console: bool
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    dashboard_enabled: bool = True
    dashboard_port: int = 8765


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
        max_market_cap_usd=be_raw.get("max_market_cap_usd", 20_000_000),
        take_profit_pct=be_raw.get("take_profit_pct", 50),
        stop_loss_pct=be_raw.get("stop_loss_pct", 25),
        trailing_activation_pct=be_raw.get("trailing_activation_pct", 20),
        trailing_stop_pct=be_raw.get("trailing_stop_pct", 15),
    )

    ct_raw = strat_raw.get("copytrade", {})
    copytrade = CopyTradeConfig(
        enabled=ct_raw.get("enabled", False),
        watched_wallets=ct_raw.get("watched_wallets", []) or [],
        mirror_pct=ct_raw.get("mirror_pct", 100),
        max_copy_delay_ms=ct_raw.get("max_copy_delay_ms", 3000),
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
        copytrade=copytrade,
        market_maker=market_maker,
        alerts_console=alerts_raw.get("console", True),
        telegram_enabled=tg_raw.get("enabled", False),
        telegram_bot_token=os.environ.get(tg_token_var, ""),
        telegram_chat_id=tg_raw.get("chat_id", ""),
        dashboard_enabled=dash_raw.get("enabled", True),
        dashboard_port=dash_raw.get("port", 8765),
    )
