"""
Wraps OKX's Onchain OS "Address Analysis API" - specifically the
address-tracker trades endpoint, which can poll OKX's own pre-curated
"smart money" wallet list (or a custom list of up to 20 wallets) for their
latest DEX trades. User-requested as a real copytrade signal source: OKX
already maintains and curates a smart-money wallet list, so this is a
ready-made feed rather than something this bot has to build from scratch
the way wallet_discovery.py does with Birdeye's per-token trader data.

Uses OKX's standard v5-style signed-request auth (the same scheme as
their main trading API, not a simple header key like Birdeye/DexScreener):
every request needs OK-ACCESS-KEY, OK-ACCESS-SIGN (a base64 HMAC-SHA256
signature over timestamp+method+requestPath+body, using the secret key),
OK-ACCESS-PASSPHRASE, and OK-ACCESS-TIMESTAMP (ISO 8601 UTC with
milliseconds). Requires a developer account + project + API key/secret/
passphrase from https://web3.okx.com/onchainos/dev-portal (user must
create this themselves - 60-day free trial, then a paid tier).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger("pumpfun_bot.okx_client")

OKX_BASE_URL = "https://web3.okx.com"
ADDRESS_TRACKER_TRADES_PATH = "/api/v6/dex/market/address-tracker/trades"

TRACKER_TYPE_SMART_MONEY = "1"
TRACKER_TYPE_KOL = "2"
TRACKER_TYPE_MULTI_ADDRESS = "3"


def _iso_timestamp() -> str:
    """OKX wants ISO 8601 UTC with millisecond precision, e.g.
    "2023-10-18T12:21:41.274Z" - matches their documented request example."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _sign(secret_key: str, timestamp: str, method: str, request_path: str, body: str = "") -> str:
    """OKX's standard v5 API signature: base64(HMAC-SHA256(secret,
    timestamp + method + requestPath + body)). Pure function, deterministic
    given its inputs - the part of this module most worth testing in
    isolation, since a subtly wrong signature fails auth with no useful
    error beyond "unauthorized"."""
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    digest = hmac.new(secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _auth_headers(api_key: str, secret_key: str, passphrase: str, method: str, request_path: str) -> dict:
    timestamp = _iso_timestamp()
    return {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": _sign(secret_key, timestamp, method, request_path),
        "OK-ACCESS-PASSPHRASE": passphrase,
        "OK-ACCESS-TIMESTAMP": timestamp,
    }


async def fetch_address_tracker_trades(
    api_key: str,
    secret_key: str,
    passphrase: str,
    tracker_type: str = TRACKER_TYPE_SMART_MONEY,
    wallet_address: str | None = None,
    trade_type: str = "0",
    chain_index: str = "501",  # Solana
    min_volume: float | None = None,
    max_volume: float | None = None,
    min_holders: int | None = None,
    min_market_cap: float | None = None,
    max_market_cap: float | None = None,
    min_liquidity: float | None = None,
    max_liquidity: float | None = None,
    timeout_sec: float = 10.0,
) -> list[dict] | None:
    """Returns the raw list of trade dicts (txHash/walletAddress/
    tokenSymbol/tokenContractAddress/marketCap/realizedPnlUsd/tradeType/
    tradeTime/...), or None if the lookup itself failed. wallet_address is
    required (comma-separated, up to 20) when tracker_type is
    TRACKER_TYPE_MULTI_ADDRESS - ignored otherwise."""
    params: dict[str, str] = {"trackerType": tracker_type, "tradeType": trade_type}
    if chain_index:
        params["chainIndex"] = chain_index
    if wallet_address:
        params["walletAddress"] = wallet_address
    if min_volume is not None:
        params["minVolume"] = str(min_volume)
    if max_volume is not None:
        params["maxVolume"] = str(max_volume)
    if min_holders is not None:
        params["minHolders"] = str(min_holders)
    if min_market_cap is not None:
        params["minMarketCap"] = str(min_market_cap)
    if max_market_cap is not None:
        params["maxMarketCap"] = str(max_market_cap)
    if min_liquidity is not None:
        params["minLiquidity"] = str(min_liquidity)
    if max_liquidity is not None:
        params["maxLiquidity"] = str(max_liquidity)

    query_string = urlencode(sorted(params.items()))
    request_path = f"{ADDRESS_TRACKER_TRADES_PATH}?{query_string}"
    headers = _auth_headers(api_key, secret_key, passphrase, "GET", request_path)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OKX_BASE_URL}{request_path}", headers=headers, timeout=timeout_sec,
            ) as resp:
                if resp.status != 200:
                    logger.debug("OKX address-tracker gaf status %d", resp.status)
                    return None
                data = await resp.json()
                if data.get("code") != "0":
                    logger.debug("OKX address-tracker antwoordde met code!=0: %s", data)
                    return None
                return data.get("data", {}).get("trades", [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kon OKX address-tracker trades niet ophalen: %s", exc)
        return None
